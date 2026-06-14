"""
Gaussian Concept Steering (GCS)
================================
Baseline activation-steering method from:
    Zhao et al., "Gaussian Concept Steering" (2024)
as implemented and evaluated in:
    Latypova R.R., "From Vectors to Subspaces: Gaussian Concept Control
    for Customizable AI Assistant Personalities", Innopolis University, 2026.

The method represents an abstract concept (e.g. an emotion) as a *probability
distribution* over logistic-regression weight vectors trained on hidden-state
activations, rather than as a single deterministic steering direction.
At inference time a steering vector is sampled from this distribution and
injected into selected transformer layers via a lightweight forward hook.

Pipeline
--------
1. Extract last-token hidden states at middle-to-late layers for positive
   (emotional) and negative (neutral) text samples.
2. Train M bootstrap logistic classifiers per layer → Gaussian (μ, σ).
3. At inference, sample a perturbed direction, normalize it, and steer the
   last-token activation of every target layer during generation.

Usage
-----
    from gcs import Config, GCSModel

    cfg = Config(CONCEPT_POSITIVE="joy")
    gcs = GCSModel(cfg)
    gcs.load_or_build(positive_texts, negative_texts)
    response = gcs.generate("Tell me a short story.", sigma_level=1.0)
    print(response)
"""

from __future__ import annotations

import gc
import os
import csv
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All tunable parameters for the GCS pipeline.

    Parameters
    ----------
    STEERING_MODEL:
        HuggingFace model identifier.  Tested on Llama-3.1-8B-Instruct,
        Qwen2.5-7B-Instruct, Gemma-2-9B-it, Phi-3.5-mini-instruct,
        Mistral-7B-Instruct-v0.3.
    CONCEPT_POSITIVE:
        Target emotional concept (e.g. "joy", "sad", "evil", "kind",
        "humorous").
    CONCEPT_NEGATIVE:
        Control / neutral concept label (used only for logging).
    STEER_LAYERS:
        Transformer layer indices to steer.  Middle-to-late layers
        (10–22 for 8-B-class models) work best empirically.
    SAMPLES_PER_CONCEPT:
        Maximum number of positive / negative samples to use.
    BATCH_SIZE_HS:
        Batch size for hidden-state extraction.
    MAX_LENGTH:
        Token truncation length during extraction.
    STEERING_STRENGTH:
        Scalar ϑ controlling how strongly the steering vector is mixed
        into the residual stream.  Sweep [0.05, 0.30] for most emotions.
    M:
        Number of bootstrap classifiers per layer.
    SUBSAMPLE_RATIO:
        Fraction of samples drawn (with replacement) for each classifier.
    GCS_RANDOM_STATE:
        Master random seed for reproducibility.
    SIGMA_LEVELS:
        ε values controlling perturbation magnitude around μ.
        σ = 0 recovers the mean direction; higher values explore
        the neighbourhood of the Gaussian.
    CACHE_DIR / RESULTS_DIR / DATASET_DIR:
        Filesystem paths for caching activations, GCS vectors, and
        generation results.
    """

    STEERING_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"

    CONCEPT_POSITIVE: str = "joy"
    CONCEPT_NEGATIVE: str = "neutral"

    STEER_LAYERS: List[int] = field(default_factory=lambda: list(range(10, 23)))
    SAMPLES_PER_CONCEPT: int = 500
    BATCH_SIZE_HS: int = 4
    MAX_LENGTH: int = 128

    STEERING_STRENGTH: float = 0.1

    # GCS ensemble parameters
    M: int = 100
    SUBSAMPLE_RATIO: float = 0.8
    GCS_RANDOM_STATE: int = 42
    SIGMA_LEVELS: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
    )

    CACHE_DIR: str = "./cache"
    RESULTS_DIR: str = "./results"
    DATASET_DIR: str = "./datasets"

    DEVICE: str = field(init=False)
    DTYPE: torch.dtype = field(init=False)

    def __post_init__(self) -> None:
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
        for d in (self.CACHE_DIR, self.RESULTS_DIR, self.DATASET_DIR):
            os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def clear_gpu_memory() -> None:
    """Release unused GPU memory and run Python GC."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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
    """Extract last-token activations at the specified transformer layers.

    Parameters
    ----------
    texts:
        Input strings to encode.
    model:
        A loaded ``AutoModelForCausalLM`` instance.
    tokenizer:
        Corresponding tokenizer.
    layers:
        Layer indices whose hidden states should be returned.
    batch_size:
        Mini-batch size for GPU throughput / memory trade-off.
    max_length:
        Maximum token length (sequences are truncated/padded to this).

    Returns
    -------
    Dict mapping layer index → Float tensor of shape ``(N, hidden_dim)``.
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

        for layer_idx in layers:
            # Take the last token representation and move to CPU immediately
            last_token = outputs.hidden_states[layer_idx][:, -1, :].float().cpu()
            all_states[layer_idx].append(last_token)

        del inputs, outputs
        clear_gpu_memory()

    return {l: torch.cat(all_states[l], dim=0) for l in layers}


# ---------------------------------------------------------------------------
# GCS vector computation
# ---------------------------------------------------------------------------

def compute_gcs_vectors(
    pos_states: Dict[int, torch.Tensor],
    neg_states: Dict[int, torch.Tensor],
    layers: List[int],
    M: int = 100,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Fit M bootstrap logistic classifiers per layer and model the weight
    distribution as a Gaussian (μ, σ).

    For each layer *l* and each of *M* bootstrap rounds:
    - Subsample (with replacement) ``subsample_ratio`` fraction of each class.
    - Fit a zero-intercept logistic regression.
    - Collect the weight vector **w_m**.

    The Gaussian is then:
        μ_l = mean({w_m}) / ‖mean({w_m})‖
        σ_l = std({w_m})   (per-dimension)

    Parameters
    ----------
    pos_states:
        Hidden states for the positive / emotional class.
    neg_states:
        Hidden states for the negative / neutral class.
    layers:
        Layers to process.
    M:
        Number of bootstrap classifiers.
    subsample_ratio:
        Fraction of each class sampled per bootstrap round.
    random_state:
        Base random seed (each round uses ``random_state + m``).

    Returns
    -------
    Dict mapping layer index → ``{"mu": ndarray, "sigma": ndarray,
    "samples": ndarray}`` where *samples* has shape ``(M, hidden_dim)``.
    """
    np.random.seed(random_state)
    gcs: Dict[int, Dict[str, np.ndarray]] = {}

    for layer in tqdm(layers, desc="Computing GCS per layer"):
        pos = pos_states[layer].cpu().numpy()
        neg = neg_states[layer].cpu().numpy()

        n_pos, n_neg = len(pos), len(neg)
        k_pos = int(n_pos * subsample_ratio)
        k_neg = int(n_neg * subsample_ratio)

        weights = []
        for m in range(M):
            idx_p = np.random.choice(n_pos, k_pos, replace=True)
            idx_n = np.random.choice(n_neg, k_neg, replace=True)

            X = np.vstack([pos[idx_p], neg[idx_n]])
            y = np.array([1] * k_pos + [0] * k_neg)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = LogisticRegression(
                max_iter=1000,
                fit_intercept=False,
                C=1.0,
                random_state=random_state + m,
            )
            clf.fit(X_scaled, y)
            weights.append(clf.coef_[0])

        weights_arr = np.array(weights)          # (M, hidden_dim)
        mu = weights_arr.mean(axis=0)
        sigma = weights_arr.std(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-8)  # normalize mean direction

        gcs[layer] = {"mu": mu, "sigma": sigma, "samples": weights_arr}
        print(f"  Layer {layer}: ‖μ‖={np.linalg.norm(mu):.4f}, mean(σ)={sigma.mean():.4f}")

    return gcs


# ---------------------------------------------------------------------------
# Vector sampling
# ---------------------------------------------------------------------------

def sample_steering_vector(
    gcs_dict: Dict[str, np.ndarray],
    sigma_level: float = 1.0,
) -> np.ndarray:
    """Sample a steering vector from the GCS Gaussian at a given σ-level.

    The vector is drawn by adding uniform noise scaled by ``sigma_level * σ``
    to the mean direction μ, then re-normalizing.

    Parameters
    ----------
    gcs_dict:
        Single-layer entry from ``compute_gcs_vectors`` output.
    sigma_level:
        ε — controls how far from μ the sample may lie.
        σ = 0 returns μ exactly.

    Returns
    -------
    Unit-norm steering vector as a 1-D NumPy array.
    """
    mu = gcs_dict["mu"]
    sigma = gcs_dict["sigma"]
    noise = np.random.uniform(-sigma_level, sigma_level, len(mu)) * sigma
    v = mu + noise
    return v / (np.linalg.norm(v) + 1e-8)


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

class SteeringLayer(nn.Module):
    """Wraps a transformer layer with an additive activation-steering hook.

    At each forward pass the last-token hidden state **h** is modified as:

        h_steered = h * (1 − ϑ) + ϑ · (s · v)

    where *v* is the (unit-norm) steering vector, *s* is an adaptive scale
    factor ``‖h‖₁ / (‖v‖₁ + ε)`` that matches the magnitude of *v* to the
    natural residual-stream scale, and ϑ = ``strength`` is the steering
    strength hyperparameter.

    Parameters
    ----------
    target_layer:
        The original transformer layer to wrap.
    layer_idx:
        Index in the model's layer list (used only for logging).
    steering_vector:
        Unit-norm direction to steer towards (NumPy array).
    strength:
        Steering strength ϑ ∈ [0, 1].
    """

    def __init__(
        self,
        target_layer: nn.Module,
        layer_idx: int,
        steering_vector: np.ndarray,
        strength: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_layer = target_layer
        self.layer_idx = layer_idx
        self.strength = strength
        self._steering_vector_np = steering_vector
        self._steering_vector: Optional[nn.Parameter] = None

    def _get_sv(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Lazy-init the steering vector on the correct device / dtype."""
        if self._steering_vector is None or self._steering_vector.device != device:
            self._steering_vector = nn.Parameter(
                torch.tensor(self._steering_vector_np, dtype=dtype, device=device),
                requires_grad=False,
            )
        return self._steering_vector

    def forward(self, *args, **kwargs):
        out = self.target_layer(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out

        last = hidden[:, -1:, :]
        sv = self._get_sv(last.device, last.dtype)

        scale = last.abs().mean() / (sv.abs().mean() + 1e-8)
        delta = (sv * scale).to(last.dtype)
        hidden[:, -1:, :] = last * (1.0 - self.strength) + self.strength * delta

        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(cfg: Config):
    """Load tokenizer and model according to *cfg*."""
    print(f"Loading {cfg.STEERING_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.STEERING_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.STEERING_MODEL,
        torch_dtype=cfg.DTYPE,
        device_map="auto" if cfg.DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
        offload_folder="./offload" if cfg.DEVICE == "cuda" else None,
        offload_state_dict=True,
    )
    print(f"Loaded. Hidden size: {model.config.hidden_size}")
    return model, tokenizer


def reset_model(model: nn.Module, original_layers: list) -> None:
    """Restore model layers to their pre-steering state."""
    for i, layer in enumerate(original_layers):
        model.model.layers[i] = layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def apply_steering(
    model: nn.Module,
    steering_vectors: Dict[int, np.ndarray],
    strength: float,
    layers: List[int],
) -> nn.Module:
    """Wrap the specified layers with :class:`SteeringLayer`."""
    for idx in layers:
        if idx in steering_vectors:
            orig = model.model.layers[idx]
            model.model.layers[idx] = SteeringLayer(orig, idx, steering_vectors[idx], strength)
    return model


# ---------------------------------------------------------------------------
# High-level GCSModel class
# ---------------------------------------------------------------------------

class GCSModel:
    """End-to-end wrapper for the GCS pipeline.

    Parameters
    ----------
    cfg:
        A :class:`Config` instance.

    Example
    -------
    ::

        cfg = Config(CONCEPT_POSITIVE="sad", STEERING_STRENGTH=0.15)
        gcs = GCSModel(cfg)
        gcs.load_model()
        gcs.load_or_build(positive_texts, negative_texts)
        print(gcs.generate("Tell me about a difficult day.", sigma_level=1.5))
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self._original_layers: Optional[list] = None
        self.gcs_vectors: Optional[Dict[int, Dict[str, np.ndarray]]] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the transformer model and tokenizer into memory."""
        self.model, self.tokenizer = load_model(self.cfg)
        self._original_layers = list(self.model.model.layers)

    def _reset(self) -> None:
        if self.model is not None and self._original_layers is not None:
            reset_model(self.model, self._original_layers)

    # ------------------------------------------------------------------
    # Activation extraction + GCS vector computation
    # ------------------------------------------------------------------

    def load_or_build(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
    ) -> None:
        """Load cached GCS vectors if available, otherwise compute them.

        Intermediate hidden states are also cached to ``cfg.CACHE_DIR``
        so that repeated calls with the same concept skip the (expensive)
        extraction step.

        Parameters
        ----------
        positive_texts:
            Emotionally-charged text samples (800 recommended).
        negative_texts:
            Neutral text samples of equal cardinality.
        """
        cfg = self.cfg
        concept = cfg.CONCEPT_POSITIVE

        gcs_path = os.path.join(cfg.CACHE_DIR, f"gcs_vectors_{concept}.pkl")
        if os.path.exists(gcs_path) and os.path.getsize(gcs_path) > 0:
            print(f"Loading cached GCS vectors from {gcs_path} …")
            with open(gcs_path, "rb") as f:
                self.gcs_vectors = pickle.load(f)
            return

        assert self.model is not None, "Call load_model() before load_or_build()."

        pos_path = os.path.join(cfg.CACHE_DIR, f"pos_states_{concept}.pkl")
        neg_path = os.path.join(cfg.CACHE_DIR, f"neg_states_{concept}.pkl")

        if os.path.exists(pos_path) and os.path.exists(neg_path):
            print("Loading cached hidden states …")
            with open(pos_path, "rb") as f:
                pos_states = pickle.load(f)
            with open(neg_path, "rb") as f:
                neg_states = pickle.load(f)
        else:
            print(f"Extracting activations for '{concept}' …")
            pos_states = extract_hidden_states(
                positive_texts[: cfg.SAMPLES_PER_CONCEPT],
                self.model, self.tokenizer, cfg.STEER_LAYERS,
                batch_size=cfg.BATCH_SIZE_HS, max_length=cfg.MAX_LENGTH,
            )
            clear_gpu_memory()
            neg_states = extract_hidden_states(
                negative_texts[: cfg.SAMPLES_PER_CONCEPT],
                self.model, self.tokenizer, cfg.STEER_LAYERS,
                batch_size=cfg.BATCH_SIZE_HS, max_length=cfg.MAX_LENGTH,
            )
            clear_gpu_memory()
            with open(pos_path, "wb") as f:
                pickle.dump(pos_states, f)
            with open(neg_path, "wb") as f:
                pickle.dump(neg_states, f)

        print(f"Computing GCS vectors (M={cfg.M}) …")
        self.gcs_vectors = compute_gcs_vectors(
            pos_states, neg_states, cfg.STEER_LAYERS,
            M=cfg.M, subsample_ratio=cfg.SUBSAMPLE_RATIO,
            random_state=cfg.GCS_RANDOM_STATE,
        )
        with open(gcs_path, "wb") as f:
            pickle.dump(self.gcs_vectors, f)
        print("GCS vectors saved.")

        del pos_states, neg_states
        clear_gpu_memory()

    # ------------------------------------------------------------------
    # Steered generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        sigma_level: float = 1.0,
        strength: Optional[float] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate text with GCS activation steering.

        Parameters
        ----------
        prompt:
            User-facing input string.
        sigma_level:
            ε — how far to sample from the Gaussian centre (σ = 0 → μ only).
        strength:
            Override for ``cfg.STEERING_STRENGTH``.
        max_new_tokens:
            Maximum tokens to generate.
        temperature:
            Sampling temperature.

        Returns
        -------
        Generated text with the prompt prefix stripped.
        """
        assert self.model is not None, "Call load_model() first."
        assert self.gcs_vectors is not None, "Call load_or_build() first."

        ϑ = strength if strength is not None else self.cfg.STEERING_STRENGTH
        self._reset()

        sampled = {
            idx: sample_steering_vector(self.gcs_vectors[idx], sigma_level)
            for idx in self.cfg.STEER_LAYERS
            if idx in self.gcs_vectors
        }
        apply_steering(self.model, sampled, ϑ, self.cfg.STEER_LAYERS)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text


# ---------------------------------------------------------------------------
# Batch evaluation helpers
# ---------------------------------------------------------------------------

def run_sweep(
    gcs_model: GCSModel,
    prompts: List[str],
    output_csv: str,
    sigma_levels: Optional[List[float]] = None,
    max_new_tokens: int = 80,
) -> None:
    """Run a full sigma-level sweep over *prompts* and write results to CSV.

    Skips any prompt already present in *output_csv* (resume-safe).

    Parameters
    ----------
    gcs_model:
        A fully initialised :class:`GCSModel`.
    prompts:
        List of evaluation prompts.
    output_csv:
        Path to the output CSV file.
    sigma_levels:
        Override ``gcs_model.cfg.SIGMA_LEVELS``.
    max_new_tokens:
        Token budget per generation.
    """
    cfg = gcs_model.cfg
    σs = sigma_levels if sigma_levels is not None else cfg.SIGMA_LEVELS

    # Resume: collect already-processed prompts
    done: set = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["prompt"])

    fieldnames = ["prompt_id", "prompt", "strength", "sigma_level", "response", "response_length"]
    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for pid, prompt in enumerate(prompts, start=1):
            if prompt in done:
                print(f"[{pid}] Already done, skipping.")
                continue
            print(f"\n[{pid}] {prompt[:70]} …")
            for σ in tqdm(σs, desc=f"Prompt {pid}"):
                resp = gcs_model.generate(prompt, sigma_level=σ, max_new_tokens=max_new_tokens)
                writer.writerow({
                    "prompt_id": pid,
                    "prompt": prompt,
                    "strength": cfg.STEERING_STRENGTH,
                    "sigma_level": σ,
                    "response": resp,
                    "response_length": len(resp.split()),
                })
            done.add(prompt)

    print(f"\nResults written to {output_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GCS Emotion Steering")
    parser.add_argument("--emotion", default="joy", help="Target emotion concept")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--strength", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="Tell me a short story about a person's choice.")
    parser.add_argument("--dataset", default="./datasets/dataset.pkl",
                        help="Path to dataset.pkl with positive/negative samples")
    args = parser.parse_args()

    cfg = Config(
        STEERING_MODEL=args.model,
        CONCEPT_POSITIVE=args.emotion,
        STEERING_STRENGTH=args.strength,
    )

    gcs = GCSModel(cfg)
    gcs.load_model()

    with open(args.dataset, "rb") as fh:
        ds = pickle.load(fh)
    concept_data = ds[args.emotion]
    gcs.load_or_build(concept_data["positive"], concept_data["negative"])

    print("\n--- Steered generation ---")
    print(gcs.generate(args.prompt, sigma_level=args.sigma))
