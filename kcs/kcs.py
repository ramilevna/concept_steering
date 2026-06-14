"""
Kernel Cauchy Steering (KCS)
============================
Production module for emotional tone control in instruction-tuned LLMs.

Part of the thesis:
    Latypova R.R. "From Vectors to Subspaces: Gaussian Concept Control
    for Customizable AI Assistant Personalities."
    Innopolis University, 2026.

KCS takes a fundamentally different approach from GCS/AGCS: instead of
linear difference-of-means directions, it trains a non-linear Kernel Cauchy
classifier on mean-pooled continuation activations, derives an explicit
steering vector as the gradient of the kernel decision function at the
neutral-class centroid, and applies it via decaying forward hooks on
causally-selected layers.

Pipeline
--------
1. Extract mean-pooled hidden states for paired (emotion / neutral) texts.
2. Train a Kernel Cauchy dual-logistic classifier per candidate layer.
3. Derive steering vectors as the gradient of the kernel decision function.
4. Run a causal probe (ΔlogP) on a held-out set to rank and select layers.
5. At inference time apply steering with an exponential decay schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All KCS hyperparameters in one place."""

    # Model
    steering_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Layer search space
    target_layers: List[int] = field(default_factory=lambda: list(range(12, 24)))
    top_k_layers: int = 6               # layers kept after causal probe

    # Kernel Cauchy classifier
    sigma_kernel: Optional[float] = None   # None → median pairwise distance
    lambda_reg: float = 1e-3
    n_epochs_dual: int = 300

    # Decaying steering hooks
    steering_decay: float = 0.85        # γ per generation step
    steering_min_scale: float = 0.3     # floor for the decay

    # Causal probe
    eta_test: float = 2.0               # scale used during probing
    n_holdout: int = 50                 # examples reserved for causal probe

    # Extraction
    batch_size: int = 4
    max_length: int = 256

    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 80

    # Caching
    checkpoint_dir: Path = Path("checkpoints")
    cache_dir: Path = Path("cache")

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_json_dataset(path: str, emotion: str) -> Tuple[List[str], List[str]]:
    """Load a JSON dataset (one emotion per file).

    Expected format::

        [
          {
            "prompt":  "Situation: ...",
            "<emotion>": "Emotional response ...",
            "neutral": "Neutral response ..."
          },
          ...
        ]

    Parameters
    ----------
    path:    path to the JSON file
    emotion: emotion key name (e.g. ``"sad"``)

    Returns
    -------
    pos_texts, neg_texts : lists of response strings (emotion / neutral)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pos_texts, neg_texts = [], []
    for item in raw:
        if emotion not in item:
            raise ValueError(
                f"Dataset entry missing key '{emotion}'. "
                f"Available keys: {list(item.keys())}"
            )
        pos_texts.append(item[emotion])
        neg_texts.append(item["neutral"])
    return pos_texts, neg_texts


def load_pickle_dataset(path: str, emotion: str) -> Tuple[List[str], List[str]]:
    """Load a pickle dataset (all emotions in one file, GCS-compatible format).

    Expected format::

        {
          "sad": {
            "positive": ["text expressing sadness ...", ...],
            "negative": ["neutral text ...", ...]
          },
          ...
        }

    Parameters
    ----------
    path:    path to the ``.pkl`` file
    emotion: emotion key name (e.g. ``"sad"``)

    Returns
    -------
    pos_texts, neg_texts : lists of response strings
    """
    with open(path, "rb") as f:
        ds = pickle.load(f)
    if emotion not in ds:
        raise KeyError(f"Emotion '{emotion}' not found in dataset. "
                       f"Available: {list(ds.keys())}")
    return ds[emotion]["positive"], ds[emotion]["negative"]


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------

class KCSModel:
    """Kernel Cauchy Steering model.

    Usage
    -----
    ::

        cfg = Config(steering_model="mistralai/Mistral-7B-Instruct-v0.3")
        kcs = KCSModel(cfg)
        kcs.load_model()

        pos_texts, neg_texts = load_json_dataset("sad_dataset.json", "sad")
        kcs.load_or_build(pos_texts, neg_texts, emotion="sad")

        response = kcs.generate("Tell me a short story.", eta_base=0.3)
        print(response)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        # Outputs of the build phase
        self.vectors: Dict[int, torch.Tensor] = {}
        self.layer_norms: Dict[int, float] = {}
        self.selected_layers: List[int] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model and tokenizer onto the available device."""
        name = self.cfg.steering_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16, device_map=self.device
        )
        self.model.eval()
        print(f"✅ Loaded model '{name}' on {self.device}")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _ckpt_path(self, name: str, emotion: str) -> Path:
        return self.cfg.checkpoint_dir / f"{name}_{emotion}.pt"

    def _save(self, data, name: str, emotion: str) -> None:
        path = self._ckpt_path(name, emotion)
        torch.save(data, path)
        print(f"💾 Saved: {path}")

    def _load(self, name: str, emotion: str):
        path = self._ckpt_path(name, emotion)
        if path.exists():
            print(f"📂 Loaded: {path}")
            return torch.load(path, map_location="cpu", weights_only=False)
        return None

    # ------------------------------------------------------------------
    # Step 1 – Activation extraction (mean-pooled over continuation tokens)
    # ------------------------------------------------------------------

    def _extract_paired_activations(
        self,
        pairs: List[Dict],
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Dict[int, float]]:
        """Extract mean-pooled hidden states for paired emotion/neutral texts.

        Unlike GCS/AGCS (last-token only), KCS mean-pools over all
        continuation tokens, producing a more representative activation
        signature of the full response.

        Returns
        -------
        acts        : dict[layer] → Tensor (2N, hidden_dim)
        labels      : Tensor (2N,) — 1 = emotion, 0 = neutral
        layer_norms : dict[layer] → float  (mean activation norm per layer)
        """
        assert self.model is not None and self.tokenizer is not None

        n_layers = self.model.config.num_hidden_layers
        layer_acts = {l: [] for l in range(n_layers)}
        layer_norm_accum = {l: [] for l in range(n_layers)}
        labels_list: List[int] = []

        # Interleave: emotion first, neutral second for each pair
        samples = []
        for p in pairs:
            samples.append((p["prompt"], p["emotion"], 1))
            samples.append((p["prompt"], p["neutral"], 0))

        captured: Dict[int, torch.Tensor] = {}
        hooks = []
        for l in range(n_layers):
            def _hook(module, inp, out, _l=l):
                h = out[0] if isinstance(out, (tuple, list)) else out
                captured[_l] = h.detach()
                return out
            hooks.append(self.model.model.layers[l].register_forward_hook(_hook))

        batch_size = self.cfg.batch_size
        print("🔹 Extracting paired activations (continuation mean-pool)…")
        try:
            for i in tqdm(range(0, len(samples), batch_size)):
                batch = samples[i : i + batch_size]
                full_texts = [p + "  " + c for p, c, _ in batch]
                prompt_lens = [
                    len(self.tokenizer(p, add_special_tokens=False).input_ids)
                    for p, _, _ in batch
                ]

                enc = self.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    self.model(**enc, use_cache=False)

                attn = enc.attention_mask
                B, T = attn.shape
                cont_mask = torch.zeros_like(attn, dtype=torch.bool)
                for b in range(B):
                    start = prompt_lens[b]
                    end = int(attn[b].sum().item())
                    if end > start:
                        cont_mask[b, start:end] = True

                for l in range(n_layers):
                    h = captured[l].cpu().float()
                    m = cont_mask.cpu().unsqueeze(-1).float()
                    mean_h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                    layer_acts[l].append(mean_h)
                    norms = h.norm(dim=-1)
                    layer_norm_accum[l].append(norms[cont_mask.cpu()])

                for _, _, lab in batch:
                    labels_list.append(lab)
        finally:
            for h in hooks:
                h.remove()

        acts = {l: torch.cat(layer_acts[l], dim=0) for l in range(n_layers)}
        layer_norms = {
            l: torch.cat(layer_norm_accum[l]).mean().item()
            for l in range(n_layers)
        }
        labels = torch.tensor(labels_list)
        return acts, labels, layer_norms

    # ------------------------------------------------------------------
    # Step 2 – Kernel Cauchy dual classifier
    # ------------------------------------------------------------------

    def _train_kernel_cauchy(
        self,
        acts: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[int, Dict]:
        """Train a Kernel Cauchy dual-logistic classifier for each target layer.

        The Cauchy kernel is chosen for its heavy-tailed robustness to
        outliers in high-dimensional activation spaces:

            K(x, x') = 1 / (1 + ||x - x'||² / σ²)

        The bandwidth σ is set to the median pairwise distance among
        activations at each layer (unless overridden by ``cfg.sigma_kernel``).

        Returns
        -------
        dual_coefs : dict[layer] → {"alpha", "sigma", "X_ref"}
        """
        dual_coefs: Dict[int, Dict] = {}
        for l in self.cfg.target_layers:
            print(f"\n🔹 Layer {l}: Kernel Cauchy + Dual Logistic")
            X = acts[l].to(self.device).float()
            y = labels.to(self.device).float() * 2 - 1
            N = X.shape[0]

            dists = torch.cdist(X, X)
            sigma = (
                self.cfg.sigma_kernel
                if self.cfg.sigma_kernel is not None
                else torch.median(dists[dists > 0]).item()
            )

            K = 1.0 / (1.0 + (dists ** 2) / (sigma ** 2))
            K = (K + K.T) / 2  # symmetrise numerically

            alpha = torch.zeros(N, requires_grad=True, device=self.device)
            opt = torch.optim.Adam([alpha], lr=1e-2)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

            for _ in range(self.cfg.n_epochs_dual):
                opt.zero_grad()
                logits = K @ alpha
                loss = torch.log(1 + torch.exp(-y * logits)).mean()
                reg = self.cfg.lambda_reg * (alpha @ K @ alpha)
                (loss + reg).backward()
                opt.step()
                scheduler.step()

            X_ref = X[labels == 0].mean(dim=0)
            dual_coefs[l] = {
                "alpha": alpha.detach().cpu(),
                "sigma": sigma,
                "X_ref": X_ref.cpu(),
            }
            print(f"  σ={sigma:.3f}, α-norm={alpha.norm().item():.2f}, "
                  f"loss={loss.item():.4f}")
        return dual_coefs

    # ------------------------------------------------------------------
    # Step 3 – Steering vector derivation
    # ------------------------------------------------------------------

    def _extract_steering_vectors(
        self,
        dual_coefs: Dict[int, Dict],
        acts: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Derive explicit steering vectors from the trained Kernel Cauchy model.

        The steering direction is the gradient of the kernel decision function
        with respect to the neutral-class centroid ``x_ref``, pointing toward
        the emotional concept manifold.  Sign is verified against a linear
        probe and the vector is unit-normalised.

        Returns
        -------
        vectors : dict[layer] → unit-norm Tensor (hidden_dim,)
        """
        vectors: Dict[int, torch.Tensor] = {}
        for l in self.cfg.target_layers:
            dc = dual_coefs[l]
            alpha = dc["alpha"].to(self.device)
            X_ref = dc["X_ref"].to(self.device)
            X = acts[l].to(self.device)
            sigma = dc["sigma"]

            diffs = X_ref.unsqueeze(0) - X           # (N, d)
            dists_sq = (diffs ** 2).sum(dim=1)        # (N,)
            K_ref = 1.0 / (1.0 + dists_sq / (sigma ** 2))
            weights = -2.0 / (sigma ** 2) * alpha * (K_ref ** 2)
            v = (weights.unsqueeze(1) * diffs).sum(dim=0)

            # Sign check: align with a fast linear probe
            probe = LogisticRegression(max_iter=1000).fit(
                X.cpu().numpy(), labels.numpy()
            )
            probe_w = torch.tensor(
                probe.coef_[0], dtype=v.dtype, device=self.device
            )
            if F.cosine_similarity(v, probe_w, dim=0) < 0:
                v = -v

            vectors[l] = (v / (v.norm() + 1e-8)).cpu()
        return vectors

    # ------------------------------------------------------------------
    # Step 4 – Causal probe for layer selection
    # ------------------------------------------------------------------

    def _teacher_forced_logprob(self, prompt: str, continuation: str) -> float:
        """Sum of log-probabilities over continuation tokens (teacher-forced)."""
        assert self.model is not None and self.tokenizer is not None
        full = prompt + "  " + continuation
        enc_full = self.tokenizer(full, return_tensors="pt").to(self.device)
        p_len = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        with torch.no_grad():
            out = self.model(**enc_full)
        logits = out.logits[0, :-1, :]
        targets = enc_full.input_ids[0, 1:]
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logp[p_len - 1 :].sum().item()

    def _causal_probe_layer(
        self,
        layer_idx: int,
        v: torch.Tensor,
        holdout: List[Dict],
        layer_norm: float,
    ) -> float:
        """Measure the causal effect of steering on ``layer_idx``.

        Computes:

            ΔlogP = [logP(emotion|steer) − logP(neutral|steer)]
                  − [logP(emotion|base)  − logP(neutral|base)]

        A positive value means the steering vector successfully shifts
        probability mass toward the target emotion.
        """
        assert self.model is not None
        v_dev = v.to(self.device).to(self.model.dtype)
        scale = self.cfg.eta_test * layer_norm

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h += scale * v_dev
            return (h,) + out[1:] if isinstance(out, tuple) else h

        deltas = []
        for p in holdout:
            base_e = self._teacher_forced_logprob(p["prompt"], p["emotion"])
            base_n = self._teacher_forced_logprob(p["prompt"], p["neutral"])
            handle = self.model.model.layers[layer_idx].register_forward_hook(_hook)
            try:
                steer_e = self._teacher_forced_logprob(p["prompt"], p["emotion"])
                steer_n = self._teacher_forced_logprob(p["prompt"], p["neutral"])
            finally:
                handle.remove()
            deltas.append((steer_e - steer_n) - (base_e - base_n))
        return float(np.mean(deltas))

    def _select_layers(
        self,
        vectors: Dict[int, torch.Tensor],
        layer_norms: Dict[int, float],
        holdout: List[Dict],
    ) -> List[int]:
        """Rank target layers by causal effect; keep the top-K positive ones."""
        causal_scores: Dict[int, float] = {}
        for l in self.cfg.target_layers:
            score = self._causal_probe_layer(
                l, vectors[l], holdout, layer_norms[l]
            )
            causal_scores[l] = score
            print(f"  Layer {l}: ΔlogP = {score:+.4f}")

        ranked = sorted(causal_scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [l for l, s in ranked if s > 0][: self.cfg.top_k_layers]
        if not selected:
            print("⚠️  No layer gave a positive causal effect — "
                  "falling back to top-K by absolute value")
            selected = [l for l, _ in ranked[: self.cfg.top_k_layers]]
        selected = sorted(selected)
        print(f"\n🎯 Selected layers (by causal probe): {selected}")
        return selected

    # ------------------------------------------------------------------
    # High-level build / load
    # ------------------------------------------------------------------

    def load_or_build(
        self,
        pos_texts: List[str],
        neg_texts: List[str],
        emotion: str,
        force_rebuild: bool = False,
    ) -> None:
        """Build (or restore from cache) all KCS artefacts for one emotion.

        Parameters
        ----------
        pos_texts      : emotional response texts
        neg_texts      : neutral response texts (same prompts)
        emotion        : emotion label used as cache key (e.g. ``"sad"``)
        force_rebuild  : ignore existing checkpoints and recompute everything
        """
        assert len(pos_texts) == len(neg_texts), \
            "pos_texts and neg_texts must have the same length"

        rng = np.random.RandomState(42)
        idx = rng.permutation(len(pos_texts))
        n_hold = min(self.cfg.n_holdout, len(pos_texts) // 5)
        hold_idx = idx[:n_hold]
        train_idx = idx[n_hold:]

        def _build_pairs(indices, pos, neg):
            return [
                {"prompt": "", "emotion": pos[i], "neutral": neg[i]}
                for i in indices
            ]

        train_pairs = _build_pairs(train_idx, pos_texts, neg_texts)
        holdout_pairs = _build_pairs(hold_idx, pos_texts, neg_texts)
        print(f"📊 Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)}")

        # ---- Activations ----
        ckpt = None if force_rebuild else self._load("activations_paired", emotion)
        if ckpt is None:
            acts, labels, layer_norms = self._extract_paired_activations(train_pairs)
            self._save(
                {"acts": acts, "labels": labels, "layer_norms": layer_norms},
                "activations_paired", emotion,
            )
        else:
            acts, labels, layer_norms = ckpt["acts"], ckpt["labels"], ckpt["layer_norms"]

        # ---- Dual Kernel Cauchy ----
        ckpt = None if force_rebuild else self._load("dual_paired", emotion)
        if ckpt is None:
            dual_coefs = self._train_kernel_cauchy(acts, labels)
            self._save(dual_coefs, "dual_paired", emotion)
        else:
            dual_coefs = ckpt

        # ---- Steering vectors ----
        ckpt = None if force_rebuild else self._load("vectors_paired", emotion)
        if ckpt is None:
            vectors = self._extract_steering_vectors(dual_coefs, acts, labels)
            self._save(vectors, "vectors_paired", emotion)
        else:
            vectors = ckpt

        # ---- Causal probe & layer selection ----
        ckpt = None if force_rebuild else self._load("causal_scores", emotion)
        if ckpt is None:
            selected_layers = self._select_layers(vectors, layer_norms, holdout_pairs)
            self._save(selected_layers, "causal_scores", emotion)
        else:
            selected_layers = ckpt
            print(f"🎯 Selected layers (cached): {selected_layers}")

        self.vectors = vectors
        self.layer_norms = layer_norms
        self.selected_layers = selected_layers

    # ------------------------------------------------------------------
    # Step 5 – Inference with decaying hooks
    # ------------------------------------------------------------------

    def _apply_steering_hooks(self, eta_base: float) -> List:
        """Register forward hooks implementing the decaying steering schedule.

        At each generation step *t* the per-layer scale is:

            scale_t = ω · ‖h̄_l‖ · max(γ^t, s_min)

        where ω = ``eta_base``, γ = ``cfg.steering_decay``, and
        s_min = ``cfg.steering_min_scale``.
        """
        assert self.model is not None
        hooks = []
        decay = self.cfg.steering_decay
        min_scale = self.cfg.steering_min_scale
        call_counters: Dict[int, List[int]] = {l: [0] for l in self.selected_layers}

        for l in self.selected_layers:
            v = self.vectors[l].to(self.model.device).to(self.model.dtype)
            base_scale = eta_base * self.layer_norms[l]

            def _make_hook(vec, layer_id, base):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    T = h.shape[1]
                    step = call_counters[layer_id][0]
                    if T == 1:
                        # Autoregressive decode: one token at a time
                        pos_scale = max(decay ** step, min_scale)
                        h[:, 0, :] += base * pos_scale * vec
                        call_counters[layer_id][0] += 1
                    else:
                        # Prefill: apply to all prompt positions
                        for t in range(T):
                            pos_scale = max(decay ** t, min_scale)
                            h[:, t, :] += base * pos_scale * vec
                        call_counters[layer_id][0] = T
                    return (h,) + out[1:] if isinstance(out, tuple) else h
                return hook

            hooks.append(
                self.model.model.layers[l].register_forward_hook(
                    _make_hook(v, l, base_scale)
                )
            )
        return hooks

    def generate(
        self,
        prompt: str,
        eta_base: float = 0.3,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a steered response for ``prompt``.

        Parameters
        ----------
        prompt        : user prompt string
        eta_base      : base steering magnitude ω; recommended sweep 0.10–0.40
        max_new_tokens: override ``cfg.max_new_tokens``
        temperature   : override ``cfg.temperature``

        Returns
        -------
        Generated text (prompt stripped).
        """
        assert self.model is not None and self.tokenizer is not None
        assert self.selected_layers, \
            "Call load_or_build() before generate()."

        max_new = max_new_tokens or self.cfg.max_new_tokens
        temp = temperature or self.cfg.temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        hooks = self._apply_steering_hooks(eta_base)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temp,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            for h in hooks:
                h.remove()

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() if text.startswith(prompt) else text


# ---------------------------------------------------------------------------
# Sweep utility
# ---------------------------------------------------------------------------

ETA_SCHEDULE_DEFAULT = [
    (0.00, 1), (0.10, 1), (0.20, 1), (0.25, 1),
    (0.30, 3), (0.35, 1), (0.40, 3), (0.45, 3),
    (0.50, 3), (0.55, 3), (0.60, 3),
    (0.70, 1), (0.80, 1), (0.90, 1),
]


def run_sweep(
    kcs: KCSModel,
    prompts: List[str],
    output_csv: str = "results/kcs_sweep.csv",
    eta_schedule: Optional[List[Tuple[float, int]]] = None,
) -> None:
    """Run a full strength sweep across all prompts and write results to CSV.

    The sweep is **resume-safe**: rows already present in ``output_csv``
    are detected and skipped on restart.

    Parameters
    ----------
    kcs          : fitted :class:`KCSModel`
    prompts      : list of test prompts
    output_csv   : path for results (created if absent)
    eta_schedule : list of (eta_base, n_repeats) pairs; defaults to
                   :data:`ETA_SCHEDULE_DEFAULT`
    """
    if eta_schedule is None:
        eta_schedule = ETA_SCHEDULE_DEFAULT

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Detect already-written rows for resume
    existing: set = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["prompt_id"], row["eta_base"], row["repeat"]))

    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["prompt_id", "prompt", "eta_base", "repeat", "response",
                 "length", "rep_ratio"]
            )

        for p_idx, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt {p_idx}: '{prompt[:70]}…'")
            for eta, n_reps in tqdm(eta_schedule, desc="η sweep", leave=False):
                for rep in range(n_reps):
                    key = (str(p_idx), str(eta), str(rep))
                    if key in existing:
                        continue
                    resp = kcs.generate(prompt, eta_base=eta)
                    words = resp.split()
                    rep_ratio = (
                        1.0 - len(set(words)) / len(words) if words else 0.0
                    )
                    writer.writerow(
                        [p_idx, prompt, eta, rep, resp, len(words),
                         round(rep_ratio, 3)]
                    )
                    f.flush()

    print(f"\n✅ Sweep complete → {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kernel Cauchy Steering – CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--emotion", required=True, help="Target emotion label")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model identifier")
    p.add_argument("--dataset", required=True,
                   help="Path to JSON or pickle dataset")
    p.add_argument("--prompt",
                   default="Tell me a short story about a person's choice.",
                   help="Prompt for single generation")
    p.add_argument("--eta", type=float, default=0.3,
                   help="Base steering magnitude ω (single generation)")
    p.add_argument("--sweep", action="store_true",
                   help="Run a full strength sweep instead of single generation")
    p.add_argument("--output-csv", default="results/kcs_sweep.csv",
                   help="Output CSV path for sweep mode")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Ignore existing checkpoints and recompute everything")
    return p


def main():
    args = _build_parser().parse_args()

    cfg = Config(steering_model=args.model)
    kcs = KCSModel(cfg)
    kcs.load_model()

    # Dataset loading: detect format by extension
    path = args.dataset
    emotion = args.emotion.lower()
    if path.endswith(".pkl"):
        pos_texts, neg_texts = load_pickle_dataset(path, emotion)
    else:
        pos_texts, neg_texts = load_json_dataset(path, emotion)

    kcs.load_or_build(pos_texts, neg_texts, emotion=emotion,
                      force_rebuild=args.force_rebuild)

    if args.sweep:
        test_prompts = [
            "Tell a short story about a person's choice.",
            "Write a movie review in 2-3 sentences.",
            "What do you think about the importance of moral principles?",
            "What would you do if you had one free day with no obligations?",
            "If you could instantly learn one skill without effort, what would it be and why?",
            "Invent a new holiday and describe how people celebrate it.",
        ]
        run_sweep(kcs, test_prompts, output_csv=args.output_csv)
    else:
        response = kcs.generate(args.prompt, eta_base=args.eta)
        print(f"\n[η={args.eta}] {response}")


if __name__ == "__main__":
    main()
