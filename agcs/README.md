# Advanced Gaussian Concept Steering (AGCS)

Improved activation-steering method for emotional tone control in instruction-tuned LLMs, implemented as part of the thesis:

> Latypova R.R. *From Vectors to Subspaces: Gaussian Concept Control for Customizable AI Assistant Personalities.*  
> Innopolis University, 2026. — [Full thesis PDF](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf)

Built on the GCS baseline (Zhao et al., 2024); see [`../gcs/`](../gcs/) for the original method.

---

## What AGCS does

AGCS is an improved variant of Gaussian Concept Steering that addresses four limitations of the GCS baseline:

| Issue in GCS | AGCS fix |
|---|---|
| Steering direction uses positive-class mean only | Uses **difference-of-means** direction: `μ = μ_pos − μ_neg` |
| Noise model ignores within-class variance | **Pooled within-class σ** with Ledoit–Wolf shrinkage |
| Sampling from a full box around μ | **Sigma-ring sampling**: reject points inside the `(k−1)σ` inner shell |
| Steering applied to last token only | **Full-sequence steering** from a configurable start position |
| No chat template support | Full support for Instruct chat templates |
| Adaptive scale via ℓ₁-norm ratio | Scale calibrated by **mean activation norm** `‖h̄_l‖` per layer |

### Pipeline

```
Paired dataset
  (emotional / neutral texts)
        │
        ▼
Hidden-state extraction      ← last-token activations at layers 10–23
        │
        ▼
Difference-of-means direction   ← μ = μ_pos − μ_neg
+ Pooled within-class σ          ← with Ledoit–Wolf shrinkage (ς = 0.1)
        │
        ▼
Sigma-ring sampling           ← n_samp=64 vectors in the shell ((k−1)σ, kσ)
→ average unit vectors → v̂_l  (final steering direction per layer)
        │
        ▼
Inference-time steering       ← additive hook on all generated tokens
h[:, s:, :] += (ω · ‖h̄_l‖) · v̂_l
```

### Key formulas

**Steering direction (per layer `l`):**
```
μ_l = μ_pos_l − μ_neg_l
```

**Pooled within-class variance with shrinkage:**
```
ϱ²_j = (Σ(x_i − μ_pos)²_j + Σ(x_i − μ_neg)²_j) / (n_pos + n_neg − 2)
ϱ²_j ← (1 − ς) · ϱ²_j + ς · ϱ̄²        # Ledoit–Wolf-style shrinkage
```

**Sigma-ring shell:**
```
(k−1)σ < |x − μ| < kσ    (component-wise)
```

**Full-sequence steering update:**
```
h[:, s:, :] = h[:, s:, :] + (ω · ‖h̄_l‖) · v̂_l
```
where `s` is the start position of the model's generated tokens and `‖h̄_l‖` is the mean activation norm at layer `l`.

---

## Comparison with GCS

| Aspect | GCS | AGCS |
|---|---|---|
| Steering direction | `μ_pos` (positive mean) | `μ_pos − μ_neg` (difference) |
| Uncertainty model | Bootstrap classifier weights | Pooled within-class `σ` |
| Sampling region | Full box | Sigma-ring shell |
| Scope of steering | Last token only | Full sequence from start pos |
| Scale calibration | Ratio of ℓ₁-norms | Mean activation norm `‖h̄_l‖` |
| Chat template | No | Yes |

---

## Results summary (from thesis, Chapter 5)

AGCS was evaluated against GCS and KCS across five models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3) and five emotions using an LLM-as-a-Judge protocol (0–10 scale).

AGCS forms a robust intermediate between GCS and KCS. It is particularly effective on architectures where last-token-only intervention is unreliable (e.g. models with short attention spans or strong positional biases).

**Usable operating window** (strength range where emotional score is elevated *and* coherence ≥ 6): 0.04 – 0.22 .

---

## Files

```
agcs/
├── agcs.py       # Clean importable module (production use)
├── agcs.ipynb    # Original Colab notebook (exploratory, GPU-ready)
└── README.md     # This file
```

---

## Requirements

```
torch>=2.0
transformers>=4.40
scikit-learn>=1.3
numpy
tqdm
```

Install:

```bash
pip install torch transformers scikit-learn numpy tqdm
```

A GPU with ≥ 16 GB VRAM is recommended for 7–9B parameter models.  
Notebooks were validated on a Colab L4 GPU.

---

## Dataset format

AGCS accepts two dataset formats.

**JSON format** (one emotion per file):

```json
[
  {
    "prompt": "Situation: You see a lost child in the street.",
    "joy":    "Response expressing joy …",
    "neutral": "Neutral response …"
  },
  ...
]
```

**Pickle format** (all emotions in one file, compatible with GCS):

```python
{
  "joy": {
    "positive": ["text expressing joy …", ...],   # ~800 samples
    "negative": ["neutral text …", ...]
  },
  "sad": { ... },
  # "evil", "kind", "humorous", ...
}
```

Each response shares the same prompt as its paired counterpart and differs only in affective tone. See [Section 3.2 of the thesis](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf) for details on dataset construction with Qwen2.5-Max.

---

## Usage

### As an importable module

```python
from agcs import Config, AGCSModel

# 1. Configure
cfg = Config(
    steering_model="mistralai/Mistral-7B-Instruct-v0.3",
    gcs_num_sigma=2.0,      # sigma-ring width k
    gcs_num_samples=64,     # vectors sampled per layer
    gcs_shrinkage=0.1,      # Ledoit–Wolf shrinkage ς
)

# 2. Load model
agcs = AGCSModel(cfg)
agcs.load_model()

# 3. Build or load cached AGCS vectors
from agcs import load_json_dataset
pos_texts, neg_texts = load_json_dataset("datasets/joy_dataset.json", "joy")
agcs.load_or_build(pos_texts, neg_texts, emotion="joy")

# 4. Generate with steering
response = agcs.generate("Tell me a short story.", strength=0.1)
print(response)
```

### CLI — single generation

```bash
python agcs.py \
  --emotion joy \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset datasets/joy_dataset.json \
  --prompt "Tell me a short story about a person's choice." \
  --strength 0.1
```

### CLI — full strength sweep

```bash
python agcs.py \
  --emotion sad \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset datasets/dataset.pkl \
  --prompt "Write a movie review in 2-3 sentences." \
  --sweep \
  --output-csv results/sad_sweep.csv
```

### Sweep via Python API

```python
from agcs import Config, AGCSModel, run_sweep, load_pickle_dataset

cfg = Config(steering_model="meta-llama/Llama-3.1-8B-Instruct")
agcs = AGCSModel(cfg)
agcs.load_model()

pos_texts, neg_texts = load_pickle_dataset("datasets/dataset.pkl", "sad")
agcs.load_or_build(pos_texts, neg_texts, emotion="sad")

prompts = [
    "Tell a short story about a person's choice.",
    "Write a movie review in 2-3 sentences.",
    "What do you think about the importance of moral principles?",
]
run_sweep(agcs, prompts, output_csv="results/sad_sweep.csv")
```

The sweep is **resume-safe**: rows already written to the CSV are skipped on restart.

---

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `steer_layers` | `range(10, 24)` | Mid-to-late layers work best for 7–9B models; top-`num_steering_layers` are selected by probing accuracy |
| `num_steering_layers` | `20` | How many layers to keep after probing-based selection |
| `gcs_num_sigma` (k) | `2.0` | Sigma-ring width; higher values explore wider directions |
| `gcs_num_samples` | `64` | Vectors sampled per layer; more → smoother estimate |
| `gcs_shrinkage` (ς) | `0.1` | Ledoit–Wolf coefficient; prevents high-variance dimensions from dominating |
| `strength` (ω) | `0.1` | Recommended sweep: 0.02 – 0.26; values above 0.22 risk incoherence |
| `temperature` | `0.7` | Sampling temperature for generation |

---

## Caching

Activations and AGCS vectors are cached automatically under `./cache/`:

```
cache/
├── pos_states_joy.pkl        # Positive hidden states (layer → tensor)
├── neg_states_joy.pkl        # Negative hidden states
└── agcs_vectors_joy.pkl      # Steering vectors + h_ref_norms
```

Subsequent runs with the same `emotion` label skip extraction and fitting entirely.
Pass `force_rebuild=True` to `load_or_build()` to override the cache.

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@thesis{latypova2026agcs,
  author  = {Latypova, Renata Ramilevna},
  title   = {From Vectors to Subspaces: Gaussian Concept Control
             for Customizable AI Assistant Personalities},
  school  = {Innopolis University},
  year    = {2026},
  type    = {Bachelor's Thesis}
}
```
