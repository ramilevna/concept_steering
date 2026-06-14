"""
Advanced Gaussian Concept Steering (AGCS)
==========================================
Activation-steering method for emotional tone control in instruction-tuned LLMs.

Improvements over the GCS baseline:
  - Steering direction:  μ = μ_pos − μ_neg  (difference-of-means)
  - Uncertainty model:   pooled within-class σ with Ledoit–Wolf shrinkage
  - Sampling region:     sigma-ring shell  ((k−1)σ, kσ)  instead of full box
  - Steering scope:      full sequence from start position (not last token only)
  - Scale calibration:   mean activation norm ‖h̄_l‖ per layer
  - Chat template:       full support for Instruct models

Reference
---------
Latypova R.R. "From Vectors to Subspaces: Gaussian Concept Control
for Customizable AI Assistant Personalities." Innopolis University, 2026.
https://github.com/ramilevna/concept_steering
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All AGCS hyper-parameters in one place."""

    # Model
    steering_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    dtype: torch.dtype = torch.float16
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Target layers for hidden-state extraction and steering
    steer_layers: List[int] = field(default_factory=lambda: list(range(10, 24)))
    # How many top-probing layers to keep for the final steering pass
    num_steering_layers: int = 20

    # Extraction
    batch_size_hs: int = 4
    max_length: int = 128

    # AGCS sampling
    gcs_num_sigma: float = 2.0     # k  – ring width: sample in ((k-1)σ, kσ)
    gcs_num_samples: int = 64      # number of vectors sampled per layer
    gcs_use_diff: bool = True      # use μ_pos − μ_neg as the mean direction
    gcs_shrinkage: float = 0.1     # Ledoit–Wolf-style shrinkage coefficient ς

    # Generation
    max_new_tokens: int = 100
    temperature: float = 0.7

    # Paths
    cache_dir: str = "./cache"
    results_dir: str = "./results"

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    texts: List[str],
    model: nn.Module,
    tokenizer,
    layers: List[int],
    batch_size: int = 4,
    max_length: int = 128,
) -> Dict[int, torch.Tensor]:
    """Return last-token activations at each requested layer.

    Parameters
    ----------
    texts:      list of input strings (already formatted, e.g. "Situation: … Response: …")
    model:      loaded HuggingFace causal LM
    tokenizer:  corresponding tokenizer (padding_side must be 'left')
    layers:     layer indices to extract (0-indexed, referring to model.model.layers)
    batch_size: number of texts processed per forward pass
    max_length: tokenisation truncation length

    Returns
    -------
    Dict mapping layer index → Tensor of shape (N, hidden_size)
    """
    all_states: Dict[int, list] = {l: [] for l in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            for layer_idx in layers:
                # left-padding ⟹ last token is the real semantic end
                last = hidden_states[layer_idx][:, -1, :].float().cpu()
                all_states[layer_idx].append(last)

            del inputs, outputs, hidden_states
            _clear_gpu_memory()

    return {l: torch.cat(v, dim=0) for l, v in all_states.items()}


# ---------------------------------------------------------------------------
# AGCS vector computation
# ---------------------------------------------------------------------------

def _pooled_within_class_stats(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    shrinkage: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (μ_pos, μ_neg, σ_pooled) with Ledoit–Wolf-style shrinkage.

    The pooled per-dimension variance is:
        ϱ²_j = (Σ(x_i − μ_pos)²_j + Σ(x_i − μ_neg)²_j) / (n_pos + n_neg − 2)

    Shrinkage blends ϱ²_j toward the global mean variance ϱ̄²:
        ϱ²_j ← (1 − ς) · ϱ²_j + ς · ϱ̄²

    This prevents high-variance dimensions from dominating the uncertainty
    estimate in high-dimensional activation spaces.
    """
    mu_pos = X_pos.mean(axis=0)
    mu_neg = X_neg.mean(axis=0)
    n_pos, n_neg = X_pos.shape[0], X_neg.shape[0]

    Xc_pos = X_pos - mu_pos
    Xc_neg = X_neg - mu_neg
    var = (np.sum(Xc_pos ** 2, axis=0) + np.sum(Xc_neg ** 2, axis=0)) / max(
        n_pos + n_neg - 2, 1
    )
    mean_var = float(var.mean())
    var = (1.0 - shrinkage) * var + shrinkage * mean_var
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu_pos, mu_neg, sigma


def _sample_in_sigma_ring(
    mu: np.ndarray,
    sigma: np.ndarray,
    num_samples: int,
    k: float = 2.0,
    max_iter: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the sigma-ring shell: (k−1)σ < |x − μ| < kσ  (component-wise).

    Points are drawn uniformly from the outer hypercube [μ − kσ, μ + kσ]
    and rejected if they fall inside the inner hypercube
    [μ − (k−1)σ, μ + (k−1)σ].  A small number of stubborn in-shell
    samples are forced out by nudging one random component to the ring boundary.

    Parameters
    ----------
    mu:          centre of the ring (the difference-of-means direction)
    sigma:       per-dimension standard deviation (pooled within-class)
    num_samples: how many samples to draw
    k:           ring width parameter (thesis default: 2.0)
    max_iter:    rejection-sampling iterations before fallback
    rng:         numpy random generator (created if None)
    """
    if rng is None:
        rng = np.random.default_rng()
    d = mu.shape[0]
    low_k  = mu - k * sigma
    high_k = mu + k * sigma

    samples = rng.uniform(low=low_k, high=high_k, size=(num_samples, d))
    if k <= 1.0 + 1e-9:
        return samples  # no inner region to reject

    low_inner  = mu - (k - 1.0) * sigma
    high_inner = mu + (k - 1.0) * sigma
    inside = np.all((samples >= low_inner) & (samples <= high_inner), axis=1)

    for _ in range(max_iter):
        if not np.any(inside):
            break
        n_bad = int(inside.sum())
        samples[inside] = rng.uniform(low=low_k, high=high_k, size=(n_bad, d))
        inside = np.all((samples >= low_inner) & (samples <= high_inner), axis=1)

    # Fallback: nudge any remaining interior samples to the ring boundary
    for idx in np.where(inside)[0]:
        axis = rng.integers(0, d)
        sign = rng.choice([-1.0, 1.0])
        samples[idx, axis] = mu[axis] + sign * (k - 1e-3) * sigma[axis]

    return samples


def compute_agcs_vectors(
    pos_states: Dict[int, torch.Tensor],
    neg_states: Dict[int, torch.Tensor],
    layers: List[int],
    num_sigma: float = 2.0,
    num_samples: int = 64,
    use_diff: bool = True,
    shrinkage: float = 0.1,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Compute per-layer AGCS steering directions.

    Algorithm per layer
    -------------------
    1. Compute μ_pos, μ_neg, σ_pooled via pooled within-class statistics.
    2. Set mean direction  μ = μ_pos − μ_neg  (or μ_pos if use_diff=False).
    3. Sample *num_samples* vectors from the sigma-ring of width *num_sigma*.
    4. Normalise each sample to unit norm.
    5. Average the unit vectors → mean direction v̄.
    6. Normalise v̄ to unit norm → final steering vector v̂_l.

    Parameters
    ----------
    pos_states:  layer → Tensor(N, d), positive-class last-token activations
    neg_states:  layer → Tensor(N, d), negative-class last-token activations
    layers:      which layers to process
    num_sigma:   sigma-ring width k  (thesis default: 2.0)
    num_samples: vectors sampled per layer (thesis default: 64)
    use_diff:    use μ_pos − μ_neg as the centre (recommended)
    shrinkage:   Ledoit–Wolf shrinkage coefficient ς (thesis default: 0.1)
    seed:        global RNG seed for reproducibility

    Returns
    -------
    Dict mapping layer index → unit-norm steering vector (np.ndarray, shape (d,))
    """
    rng = np.random.default_rng(seed)
    steering_vectors: Dict[int, np.ndarray] = {}

    for layer in tqdm(layers, desc="Computing AGCS vectors"):
        X_pos = pos_states[layer].cpu().numpy().astype(np.float32)
        X_neg = neg_states[layer].cpu().numpy().astype(np.float32)

        mu_pos, mu_neg, sigma = _pooled_within_class_stats(X_pos, X_neg, shrinkage)
        mu = (mu_pos - mu_neg) if use_diff else mu_pos

        samples = _sample_in_sigma_ring(
            mu=mu, sigma=sigma, num_samples=num_samples, k=num_sigma, rng=rng
        )

        norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8
        samples_unit = samples / norms
        mean_dir = samples_unit.mean(axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)

        # Diagnostic: signed projection margin
        proj_pos = X_pos @ mean_dir
        proj_neg = X_neg @ mean_dir
        margin = (proj_pos.mean() - proj_neg.mean()) / (
            proj_pos.std() + proj_neg.std() + 1e-8
        )

        steering_vectors[layer] = mean_dir
        print(
            f"  Layer {layer:>2}: ||v̂||={np.linalg.norm(mean_dir):.4f},"
            f" margin={margin:+.3f}"
        )

    return steering_vectors


def compute_h_ref_norms(
    pos_states: Dict[int, torch.Tensor],
    neg_states: Dict[int, torch.Tensor],
    layers: List[int],
) -> Dict[int, float]:
    """Mean activation norm ‖h̄_l‖ per layer, used to calibrate steering magnitude.

    The steering update is:
        h_steered = h + (strength · ‖h̄_l‖) · v̂_l

    Calibrating by ‖h̄_l‖ makes the strength parameter model- and
    layer-agnostic, avoiding the need to re-tune it per architecture.
    """
    norms: Dict[int, float] = {}
    for layer in layers:
        H = torch.cat([pos_states[layer], neg_states[layer]], dim=0)
        norms[layer] = float(torch.norm(H.float(), dim=1).mean().item())
    return norms


# ---------------------------------------------------------------------------
# Steering layer (full-sequence, position-aware)
# ---------------------------------------------------------------------------

class SteeringLayer(nn.Module):
    """Wraps a transformer layer and injects an additive steering signal.

    The intervention is applied to every token position ≥ start_pos,
    which corresponds to the model's generated tokens (not the user prompt).
    This preserves prompt integrity while ensuring the steering signal
    persists throughout the entire auto-regressive decoding process.

    Steering update:
        h[:, start_pos:, :] += (strength · ‖h̄_l‖) · v̂_l

    Parameters
    ----------
    target_layer:   the original transformer layer being wrapped
    layer_idx:      index of this layer (for bookkeeping)
    steering_vector: unit-norm direction v̂_l  (shape: (d,))
    strength:        steering strength ω ∈ [0, 1]
    h_ref_norm:      mean activation norm ‖h̄_l‖ for magnitude calibration
    """

    def __init__(
        self,
        target_layer: nn.Module,
        layer_idx: int,
        steering_vector: np.ndarray,
        strength: float = 0.1,
        h_ref_norm: float = 1.0,
    ):
        super().__init__()
        self.target_layer = target_layer
        self.layer_idx = layer_idx
        self.strength = strength
        self.h_ref_norm = h_ref_norm
        self.start_pos: int = 0

        v = np.asarray(steering_vector, dtype=np.float32)
        self.steering_vector_np = v / (np.linalg.norm(v) + 1e-8)

    # Delegate attribute access to the wrapped layer so HuggingFace internals work
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.target_layer, name)

    def set_start_pos(self, pos: int) -> None:
        """Set the token index from which steering is applied."""
        self.start_pos = int(pos)

    def forward(self, *args, **kwargs):
        original_output = self.target_layer(*args, **kwargs)

        if isinstance(original_output, tuple):
            hidden_states, rest = original_output[0], original_output[1:]
        else:
            hidden_states, rest = original_output, None

        v = torch.tensor(
            self.steering_vector_np,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        delta = (self.strength * self.h_ref_norm) * v  # shape: (d,)

        seq_len = hidden_states.shape[1]
        if seq_len == 1:
            # KV-cache decoding: single new token
            hidden_states = hidden_states + delta.view(1, 1, -1)
        else:
            sp = min(self.start_pos, seq_len)
            if sp < seq_len:
                hidden_states = hidden_states.clone()
                hidden_states[:, sp:, :] += delta.view(1, 1, -1)

        return (hidden_states,) + rest if rest is not None else hidden_states


def apply_steering(
    model: nn.Module,
    steering_vectors: Dict[int, np.ndarray],
    strength: float,
    layers: List[int],
    h_ref_norms: Optional[Dict[int, float]] = None,
) -> nn.Module:
    """Wrap the specified layers with SteeringLayer hooks.

    Call :func:`reset_model` to remove the hooks before the next generation.
    """
    for layer_idx in layers:
        if layer_idx not in steering_vectors:
            continue
        original = model.model.layers[layer_idx]
        h_ref = h_ref_norms.get(layer_idx, 1.0) if h_ref_norms else 1.0
        model.model.layers[layer_idx] = SteeringLayer(
            target_layer=original,
            layer_idx=layer_idx,
            steering_vector=steering_vectors[layer_idx],
            strength=strength,
            h_ref_norm=h_ref,
        )
    return model


def set_start_pos(model: nn.Module, layers: List[int], start_pos: int) -> None:
    """Broadcast *start_pos* to all active SteeringLayers before generation."""
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        if isinstance(layer, SteeringLayer):
            layer.set_start_pos(start_pos)


def reset_model(model: nn.Module, original_layers: List[nn.Module]) -> None:
    """Restore all transformer layers to their original (unsteered) state."""
    for i, layer in enumerate(original_layers):
        model.model.layers[i] = layer
    _clear_gpu_memory()


# ---------------------------------------------------------------------------
# High-level AGCSModel class
# ---------------------------------------------------------------------------

class AGCSModel:
    """End-to-end wrapper for loading a model and running AGCS steering.

    Usage
    -----
    >>> cfg = Config(steering_model="mistralai/Mistral-7B-Instruct-v0.3")
    >>> agcs = AGCSModel(cfg)
    >>> agcs.load_model()
    >>> agcs.load_or_build(pos_texts, neg_texts)
    >>> print(agcs.generate("Tell me a short story."))
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.original_layers: Optional[List[nn.Module]] = None
        self.steering_vectors: Optional[Dict[int, np.ndarray]] = None
        self.h_ref_norms: Optional[Dict[int, float]] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Download (or load from cache) the model and tokenizer."""
        cfg = self.cfg
        print(f"Loading {cfg.steering_model} …")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.steering_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding ensures the last token is the real semantic end
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.steering_model,
            torch_dtype=cfg.dtype,
            device_map="auto" if cfg.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        self.original_layers = list(self.model.model.layers)
        print(
            f"  hidden_size={self.model.config.hidden_size}, "
            f"num_layers={self.model.config.num_hidden_layers}"
        )

    # ------------------------------------------------------------------
    # AGCS vector computation with caching
    # ------------------------------------------------------------------

    def load_or_build(
        self,
        pos_texts: List[str],
        neg_texts: List[str],
        emotion: str = "concept",
        force_rebuild: bool = False,
    ) -> None:
        """Compute (or load from cache) the AGCS steering vectors.

        Parameters
        ----------
        pos_texts:     texts representing the target emotional concept
        neg_texts:     corresponding neutral texts
        emotion:       label used for cache filenames
        force_rebuild: ignore existing cache and recompute
        """
        cfg = self.cfg
        pos_path = Path(cfg.cache_dir) / f"pos_states_{emotion}.pkl"
        neg_path = Path(cfg.cache_dir) / f"neg_states_{emotion}.pkl"
        vec_path = Path(cfg.cache_dir) / f"agcs_vectors_{emotion}.pkl"

        # Hidden states
        if not force_rebuild and pos_path.exists() and neg_path.exists():
            print("Loading cached activations …")
            pos_states = pickle.loads(pos_path.read_bytes())
            neg_states = pickle.loads(neg_path.read_bytes())
        else:
            print("Extracting positive activations …")
            pos_states = extract_hidden_states(
                pos_texts, self.model, self.tokenizer,
                cfg.steer_layers, cfg.batch_size_hs, cfg.max_length,
            )
            _clear_gpu_memory()
            print("Extracting negative activations …")
            neg_states = extract_hidden_states(
                neg_texts, self.model, self.tokenizer,
                cfg.steer_layers, cfg.batch_size_hs, cfg.max_length,
            )
            _clear_gpu_memory()
            pos_path.write_bytes(pickle.dumps(pos_states))
            neg_path.write_bytes(pickle.dumps(neg_states))
            print("  Activations cached.")

        # AGCS vectors
        if not force_rebuild and vec_path.exists():
            print("Loading cached AGCS vectors …")
            data = pickle.loads(vec_path.read_bytes())
            self.steering_vectors = data["steering_vectors"]
            self.h_ref_norms = data["h_ref_norms"]
        else:
            print("Computing AGCS vectors …")
            self.steering_vectors = compute_agcs_vectors(
                pos_states, neg_states,
                layers=cfg.steer_layers,
                num_sigma=cfg.gcs_num_sigma,
                num_samples=cfg.gcs_num_samples,
                use_diff=cfg.gcs_use_diff,
                shrinkage=cfg.gcs_shrinkage,
            )
            self.h_ref_norms = compute_h_ref_norms(
                pos_states, neg_states, cfg.steer_layers
            )
            vec_path.write_bytes(
                pickle.dumps(
                    {"steering_vectors": self.steering_vectors, "h_ref_norms": self.h_ref_norms}
                )
            )
            print(f"  Saved {vec_path}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        strength: float = 0.1,
        layers: Optional[List[int]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: int = 0,
    ) -> str:
        """Generate a response to *prompt* with AGCS steering applied.

        Parameters
        ----------
        prompt:         user message (plain text; chat template applied internally)
        strength:       steering strength ω (recommended range: 0.02 – 0.26)
        layers:         override which layers to steer (defaults to cfg.steer_layers)
        max_new_tokens: override generation length
        temperature:    override sampling temperature
        seed:           random seed for reproducibility

        Returns
        -------
        Decoded generated text (str), excluding the input prompt.
        """
        assert self.model is not None, "Call load_model() first."
        assert self.steering_vectors is not None, "Call load_or_build() first."

        cfg = self.cfg
        layers = layers if layers is not None else cfg.steer_layers
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature

        reset_model(self.model, self.original_layers)
        torch.manual_seed(seed)
        np.random.seed(seed)

        input_ids, attention_mask, start_pos = self._build_chat_inputs(prompt)

        if strength != 0.0:
            apply_steering(self.model, self.steering_vectors, strength, layers, self.h_ref_norms)
            set_start_pos(self.model, layers, start_pos)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0, input_ids.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def _build_chat_inputs(self, prompt: str):
        """Apply the model's chat template and return (input_ids, mask, start_pos)."""
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        if hasattr(encoded, "to"):
            encoded = encoded.to(self.model.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"].to(self.model.device)
            attention_mask = encoded.get(
                "attention_mask", torch.ones_like(input_ids)
            ).to(self.model.device)
        else:
            raise TypeError(f"Unexpected apply_chat_template output: {type(encoded)}")

        return input_ids, attention_mask, input_ids.shape[1]


# ---------------------------------------------------------------------------
# Sweep utility (resume-safe)
# ---------------------------------------------------------------------------

def run_sweep(
    agcs: AGCSModel,
    prompts: List[str],
    strengths: Optional[List[float]] = None,
    n_seeds: int = 1,
    output_csv: str = "results/sweep.csv",
    max_new_tokens: int = 80,
) -> None:
    """Run a steering-strength sweep over *prompts* and write results to CSV.

    The sweep is resume-safe: rows already present in *output_csv* are skipped.

    Parameters
    ----------
    agcs:           initialised and built AGCSModel
    prompts:        list of user prompts to evaluate
    strengths:      steering strengths to sweep (default: 0.00 … 0.26 in 0.02 steps)
    n_seeds:        number of random seeds per (prompt, strength) pair
    output_csv:     path to the output CSV file
    max_new_tokens: generation length per sample
    """
    if strengths is None:
        strengths = [round(i * 0.02, 3) for i in range(14)]  # 0.00 … 0.26

    # Load existing results to enable resume
    existing: set = set()
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add((row["prompt"], row["strength"], row["seed"]))

    fieldnames = ["prompt_id", "prompt", "strength", "seed", "response", "response_length"]
    file_exists = csv_path.exists()
    out_f = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    total = 0
    try:
        for p_idx, prompt in enumerate(prompts, start=1):
            print(f"\nPrompt {p_idx}: '{prompt[:70]}…'")
            for strength in tqdm(strengths, desc=f"P{p_idx} strengths"):
                for seed in range(n_seeds):
                    key = (prompt, str(strength), str(seed))
                    if key in existing:
                        continue
                    response = agcs.generate(
                        prompt, strength=strength, max_new_tokens=max_new_tokens, seed=seed
                    )
                    writer.writerow(
                        {
                            "prompt_id": p_idx,
                            "prompt": prompt,
                            "strength": strength,
                            "seed": seed,
                            "response": response,
                            "response_length": len(response.split()),
                        }
                    )
                    total += 1
    finally:
        out_f.close()

    print(f"\nDone. New rows written: {total}. Results → {output_csv}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_json_dataset(path: str, emotion: str) -> Tuple[List[str], List[str]]:
    """Load a JSON dataset and return (pos_texts, neg_texts).

    Expected JSON format (list of objects)::

        [
          {
            "prompt":  "Situation: …",
            "<emotion_lower>": "Response expressing the emotion …",
            "neutral": "Neutral response …"
          },
          …
        ]

    Parameters
    ----------
    path:    path to the ``<emotion>_dataset.json`` file
    emotion: emotion label (e.g. ``"joy"``), case-insensitive
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    emotion_key = emotion.lower()

    def _ctx(ex, key):
        return f"Situation: {ex['prompt']}\n\nResponse: {ex[key]}"

    pos_texts = [_ctx(ex, emotion_key) for ex in raw]
    neg_texts = [_ctx(ex, "neutral") for ex in raw]
    return pos_texts, neg_texts


def load_pickle_dataset(path: str, emotion: str) -> Tuple[List[str], List[str]]:
    """Load a pickle dataset (GCS format) and return (pos_texts, neg_texts).

    Expected pickle structure::

        {
          "joy":  {"positive": [...], "negative": [...]},
          "sad":  {"positive": [...], "negative": [...]},
          …
        }
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    entry = data[emotion.lower()]
    return entry["positive"], entry["negative"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced Gaussian Concept Steering (AGCS) – CLI"
    )
    p.add_argument("--emotion",   required=True, help="Target emotion (e.g. joy)")
    p.add_argument("--model",     default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--dataset",   required=True,
                   help="Path to dataset (.json or .pkl)")
    p.add_argument("--prompt",    default="Tell me a short story about a person's choice.")
    p.add_argument("--strength",  type=float, default=0.1)
    p.add_argument("--num-sigma", type=float, default=2.0,
                   help="Sigma-ring width k (default: 2.0)")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--shrinkage",  type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--sweep",    action="store_true",
                   help="Run a full strength sweep instead of a single generation")
    p.add_argument("--output-csv", default="results/agcs_sweep.csv")
    p.add_argument("--cache-dir",  default="./cache")
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = Config(
        steering_model=args.model,
        gcs_num_sigma=args.num_sigma,
        gcs_num_samples=args.num_samples,
        gcs_shrinkage=args.shrinkage,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
    )

    agcs = AGCSModel(cfg)
    agcs.load_model()

    # Dataset loading
    ds_path = args.dataset
    if ds_path.endswith(".pkl"):
        pos_texts, neg_texts = load_pickle_dataset(ds_path, args.emotion)
    else:
        pos_texts, neg_texts = load_json_dataset(ds_path, args.emotion)

    agcs.load_or_build(pos_texts, neg_texts, emotion=args.emotion)

    if args.sweep:
        run_sweep(
            agcs,
            prompts=[args.prompt],
            output_csv=args.output_csv,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        response = agcs.generate(args.prompt, strength=args.strength)
        print(f"\n{'='*60}")
        print(f"Emotion : {args.emotion}  |  strength : {args.strength}")
        print(f"Prompt  : {args.prompt}")
        print(f"{'='*60}")
        print(response)


if __name__ == "__main__":
    main()
