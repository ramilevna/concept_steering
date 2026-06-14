# Kernel Cauchy Steering (KCS)

Novel non-linear activation-steering method for emotional tone control in
instruction-tuned LLMs, proposed as part of the thesis:

> Latypova R.R. *From Vectors to Subspaces: Gaussian Concept Control for Customizable AI Assistant Personalities.*  
> Innopolis University, 2026. — [Full thesis PDF](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf)

KCS is the highest-intensity method in the study, consistently outperforming
both GCS and AGCS on peak emotional score while remaining within an acceptable
coherence window.

---

## What KCS does

KCS learns a **non-linear Kernel Cauchy
classifier** on mean-pooled continuation activations and derives the steering
vector as the **gradient of the kernel decision function** at the neutral-class
centroid. Only layers with a demonstrated causal role (measured by ΔlogP on a
held-out set) receive an intervention, and the injection magnitude follows an
**exponential decay schedule** to prevent over-steering on long outputs.

### Pipeline

```
Paired dataset
  (emotional / neutral texts)
        │
        ▼
Hidden-state extraction      ← mean-pooled over continuation tokens (not last-token)
        │
        ▼
Kernel Cauchy classifier      ← dual logistic objective, Adam + step-decay LR
  K(x,x') = 1 / (1 + ||x−x'||² / σ²)
  σ = median pairwise distance per layer
        │
        ▼
Steering vector derivation    ← gradient of kernel decision function at x_ref
  v = Σ_i [ −2/σ² · α_i · K(x_ref, x_i)² · (x_ref − x_i) ]
  sign-corrected by linear probe, then unit-normalised
        │
        ▼
Causal probe (held-out 50 pairs)
  ΔlogP_l = [logP(emo|steer) − logP(neu|steer)] − [logP(emo|base) − logP(neu|base)]
  → top-K layers with ΔlogP > 0 selected
        │
        ▼
Inference-time steering       ← decaying forward hooks on selected layers
  h[:, t, :] += ω · ‖h̄_l‖ · max(γ^t, s_min) · v̂_l
```

### Key formulas

**Cauchy kernel (per layer `l`):**
```
K(x, x') = 1 / (1 + ||x − x'||² / σ_l²)     σ_l = median pairwise distance
```

**Dual logistic objective:**
```
min_α  Σ_i log(1 + exp(−y_i · (K α)_i)) + λ · αᵀKα
```
optimised with Adam (300 epochs, step-decay LR schedule).

**Steering vector (gradient at neutral centroid x_ref):**
```
v_l = Σ_i [−2/σ_l² · α_i · K(x_ref, x_i)²] · (x_ref − x_i)
v̂_l = v_l / ‖v_l‖
```

**Causal effect per layer:**
```
ΔlogP_l = [logP(emo|steer) − logP(neu|steer)] − [logP(emo|base) − logP(neu|base)]
```
Only layers with ΔlogP_l > 0 are selected; top K = 6 are kept.

**Decaying hook (position t, layer l):**
```
h[:, t, :] += ω · ‖h̄_l‖ · max(γᵗ, s_min) · v̂_l
              γ = 0.85,  s_min = 0.30
```

---

## Comparison with GCS and AGCS

| Aspect | GCS | AGCS | **KCS** |
|---|---|---|---|
| Activation encoding | Last token | Mean-pooled (full seq.) | **Mean-pooled (continuation)** |
| Classifier | Bootstrap logistic ensemble | Difference-of-means | **Kernel Cauchy dual logistic** |
| Concept boundary | Linear | Linear | **Non-linear (kernel)** |
| Steering direction | μ_pos | μ_pos − μ_neg | **Kernel gradient at x_ref** |
| Layer selection | Fixed range | Fixed range | **Causal probe (ΔlogP)** |
| Scope of steering | Last token | Full sequence | **Selected layers, decaying** |
| Scale calibration | ℓ₁-norm ratio | Mean activation norm | **Mean activation norm + decay** |
| Peak emotional score | Moderate | Higher | **Highest** |
| Coherence | **Highest** | High | Medium |
| Usable window (ω) | 0.02–0.26* | 0.04–0.22* | **0.10–0.40** |

*GCS and AGCS use different parameter scales (ε and ϑ respectively); values are not directly comparable with KCS ω.

---

## Results summary (from thesis, Chapter 5)

Evaluated across five models (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct,
Gemma-2-9B-it, Phi-3.5-mini-instruct, Mistral-7B-Instruct-v0.3) and five
emotions (sad, evil, joy, kind, humorous) using an LLM-as-a-Judge protocol
(Qwen2.5-Max, 0–10 scale, 3 independent evaluations per text).

### Peak emotional score (coherence ≥ 6) per emotion

| Emotion | GCS | AGCS | **KCS** |
|---|---|---|---|
| Sad | 8.33 | 9.17 | **9.67** |
| Evil | 7.00 | 5.17 | **7.00** |
| Joy | 7.50 | 7.67 | **8.33** |
| Kind | 9.00 | 8.83 | **9.17** |
| Humorous | 6.50 | 6.67 | **7.17** |

KCS achieves the highest peak score across all emotions. The overall maximum
in the study — **9.67 on Sad with Qwen2.5-7B** — was produced by KCS.

KCS requires careful calibration of ω: setting it above 0.40–0.50 risks
incoherence on most architectures.

### Model-specific notes

- **Llama-3.1-8B / Qwen2.5-7B / Mistral-7B** — respond predictably; wide usable windows with smooth response curves.  
- **Gemma-2-9B** — benefits most from KCS due to grouped-query attention; KCS achieves peaks of 8.33 (highest of any method on Gemma).  
- **Phi-3.5-mini** — compact residual stream; narrow the usable window to ω ≤ 0.30.

---

## Files

```
kcs/
├── kcs.py        # Clean importable module (production use)
├── kcs.ipynb     # Original Colab notebook (exploratory, GPU-ready)
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
The notebook was validated on a Colab L4 GPU.

---

## Dataset format

KCS accepts two formats.

**JSON format** (one emotion per file):

```json
[
  {
    "prompt":  "Situation: You see a lost child in the street.",
    "sad":     "Response expressing sadness …",
    "neutral": "Neutral response …"
  },
  ...
]
```

**Pickle format** (all emotions in one file, GCS-compatible):

```python
{
  "sad": {
    "positive": ["text expressing sadness …", ...],   # ~800 samples
    "negative": ["neutral text …", ...]
  },
  "joy": { ... },
  # "evil", "kind", "humorous", ...
}
```

Each response shares the same prompt as its paired counterpart and differs
only in affective tone. See [Section 3.2 of the thesis](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf)
for details on dataset construction with Qwen2.5-Max.

A held-out set of 50 pairs (sampled with `random_state=42`) is automatically
reserved from the training data for the causal probe layer selection step.

---

## Usage

### As an importable module

```python
from kcs import Config, KCSModel, load_json_dataset

# 1. Configure
cfg = Config(
    steering_model="mistralai/Mistral-7B-Instruct-v0.3",
    top_k_layers=6,          # layers selected by causal probe
    steering_decay=0.85,     # γ — decay per generation step
    steering_min_scale=0.3,  # s_min — floor for the decay
    lambda_reg=1e-3,         # dual regularisation
    n_epochs_dual=300,
)

# 2. Load model
kcs = KCSModel(cfg)
kcs.load_model()

# 3. Build or load cached KCS vectors
pos_texts, neg_texts = load_json_dataset("datasets/sad_dataset.json", "sad")
kcs.load_or_build(pos_texts, neg_texts, emotion="sad")

# 4. Generate with steering
response = kcs.generate("Tell me a short story.", eta_base=0.3)
print(response)
```

### CLI — single generation

```bash
python kcs.py \
  --emotion sad \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset datasets/sad_dataset.json \
  --prompt "Tell me a short story about a person's choice." \
  --eta 0.3
```

### CLI — full strength sweep

```bash
python kcs.py \
  --emotion sad \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset datasets/dataset.pkl \
  --sweep \
  --output-csv results/sad_sweep.csv
```

### Sweep via Python API

```python
from kcs import Config, KCSModel, run_sweep, load_pickle_dataset

cfg = Config(steering_model="meta-llama/Llama-3.1-8B-Instruct")
kcs = KCSModel(cfg)
kcs.load_model()

pos_texts, neg_texts = load_pickle_dataset("datasets/dataset.pkl", "sad")
kcs.load_or_build(pos_texts, neg_texts, emotion="sad")

prompts = [
    "Tell a short story about a person's choice.",
    "Write a movie review in 2-3 sentences.",
    "What do you think about the importance of moral principles?",
]
run_sweep(kcs, prompts, output_csv="results/sad_sweep.csv")
```

The sweep is **resume-safe**: rows already written to the CSV are skipped on
restart.

---

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `target_layers` | `range(12, 24)` | Candidate layers scored by causal probe; mid-to-late layers for 7–9B models |
| `top_k_layers` | `6` | Layers retained after causal probe ranking |
| `sigma_kernel` | `None` | Cauchy bandwidth σ; `None` → median pairwise distance per layer |
| `lambda_reg` | `1e-3` | L2 regularisation on dual coefficients α |
| `n_epochs_dual` | `300` | Optimisation epochs for the dual logistic objective |
| `eta_base` (ω) | `0.3` | Base steering magnitude; recommended sweep: 0.10–0.40 |
| `steering_decay` (γ) | `0.85` | Exponential decay factor per generation step |
| `steering_min_scale` (s_min) | `0.3` | Minimum scale floor — prevents signal from vanishing |
| `eta_test` | `2.0` | Scale used during causal probe evaluation |
| `n_holdout` | `50` | Examples reserved for causal probe (not used in vector fitting) |
| `temperature` | `0.7` | Sampling temperature for generation |

**ω calibration by model:**

| Model | Recommended ω range |
|---|---|
| Llama-3.1-8B-Instruct | 0.10–0.50 |
| Qwen2.5-7B-Instruct | 0.10–0.55 |
| Gemma-2-9B-it | 0.10–0.40 |
| Mistral-7B-Instruct-v0.3 | 0.10–0.50 |
| Phi-3.5-mini-instruct | 0.10–0.30 |

---

## Caching

All intermediate artefacts are saved automatically under `./checkpoints/`:

```
checkpoints/
├── activations_paired_sad.pt   # Mean-pooled hidden states + layer norms
├── dual_paired_sad.pt          # Dual coefficients α, σ, x_ref per layer
├── vectors_paired_sad.pt       # Unit-normalised steering vectors per layer
└── causal_scores_sad.pt        # Selected layers after causal probe
```

Subsequent runs with the same `emotion` label skip all computation.  
Pass `force_rebuild=True` to `load_or_build()` to override the cache.

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@thesis{latypova2026kcs,
  author  = {Latypova, Renata Ramilevna},
  title   = {From Vectors to Subspaces: Gaussian Concept Control
             for Customizable AI Assistant Personalities},
  school  = {Innopolis University},
  year    = {2026},
  type    = {Bachelor's Thesis}
}
```

The non-linear kernel approach is motivated by:

```bibtex
@article{ravfogel2024kernelized,
  title   = {Kernelized Concept Erasure},
  author  = {Ravfogel, Shauli and Vargas, Francisco and Goldberg, Yoav
             and Cotterell, Ryan},
  year    = {2024},
  eprint  = {2201.12191}
}
```