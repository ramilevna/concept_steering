# Gaussian Concept Steering (GCS)

Baseline activation-steering method for emotional tone control in instruction-tuned LLMs, implemented as part of the thesis:

> Latypova R.R. *From Vectors to Subspaces: Gaussian Concept Control for Customizable AI Assistant Personalities.*  
> Innopolis University, 2026. — [Full thesis PDF](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf)

Original GCS framework: Zhao et al. (2024).

---

## What GCS does

Instead of using a single fixed steering vector (difference-of-means), GCS models an emotional concept as a **Gaussian distribution over logistic-regression weight vectors** trained on hidden-state activations. At inference time, a steering vector is *sampled* from this distribution and injected into selected transformer layers — providing both a central direction and an uncertainty estimate that can be explored via a `sigma_level` hyperparameter.

### Pipeline

```
Paired dataset
  (emotional / neutral texts)
        │
        ▼
Hidden-state extraction      ← last-token activations at layers 10–22
        │
        ▼
Bootstrap classifier ensemble  ← M=100 logistic regressions per layer
        │
        ▼
Gaussian (μ, σ) per layer     ← μ: mean direction, σ: per-dim std
        │
        ▼
Inference-time sampling       ← perturb μ with ε·σ noise, normalize
        │
        ▼
Activation steering           ← inject into last token of each target layer
```

**Steering update (per layer, per decoding step):**

```
h_steered = h · (1 − ϑ) + ϑ · (s · v)
```

where `v` is the sampled unit-norm direction, `s = ‖h‖₁ / (‖v‖₁ + ε)` is an adaptive scale factor, and `ϑ` is the steering strength.

---

## Limitations (addressed in AGCS)

| Issue | GCS behaviour |
|---|---|
| Token scope | Steers last token only → signal fades over long sequences |
| Direction estimate | Uses positive-class mean; ignores neutral class |
| Noise model | Uniform box around μ, no within-class variance |
| Chat template | No special handling |

See [`../agcs/`](../agcs/) for the Advanced GCS variant that resolves all four.

---

## Files

```
gcs/
├── gcs.ipynb      # Original Colab notebook (exploratory, GPU-ready)
└── gcs.py         # Clean importable module (production use)
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

GCS expects a `dataset.pkl` file — a Python dict with the following structure:

```python
{
  "joy": {
    "positive": ["text expressing joy ...", ...],   # 800 samples
    "negative": ["neutral text ...", ...]            # 800 samples
  },
  "sad": { ... },
  # "evil", "kind", "humorous", ...
}
```

Each entry is a response to a neutral prompt; positive and negative responses share the same prompt but differ only in affective tone. See [Section 3.2 of the thesis](../docs/thesis%20Latypova%20Renata%20Ramilevna.pdf) for details on dataset construction with Qwen2.5-Max.

---

## Usage

### As an importable module

```python
import pickle
from gcs import Config, GCSModel

# 1. Configure
cfg = Config(
    CONCEPT_POSITIVE="joy",
    STEERING_MODEL="meta-llama/Llama-3.1-8B-Instruct",
    STEERING_STRENGTH=0.1,
)

# 2. Load model
gcs = GCSModel(cfg)
gcs.load_model()

# 3. Build or load cached GCS vectors
with open("datasets/dataset.pkl", "rb") as f:
    ds = pickle.load(f)
gcs.load_or_build(ds["joy"]["positive"], ds["joy"]["negative"])

# 4. Generate with steering
print(gcs.generate("Tell me a short story.", sigma_level=1.0))
```

### CLI

```bash
python gcs.py \
  --emotion joy \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --strength 0.1 \
  --sigma 1.0 \
  --prompt "Tell me a short story about a person's choice." \
  --dataset datasets/dataset.pkl
```

### Full sigma sweep (batch evaluation)

```python
from gcs import Config, GCSModel, run_sweep

cfg = Config(CONCEPT_POSITIVE="sad", STEERING_STRENGTH=0.1)
gcs = GCSModel(cfg)
gcs.load_model()
gcs.load_or_build(positive_texts, negative_texts)

prompts = [
    "Tell a short story about a person's choice.",
    "Write a movie review in 2-3 sentences.",
]
run_sweep(gcs, prompts, output_csv="results/sad_sweep.csv")
```

The sweep is **resume-safe**: already-processed prompts are skipped on restart.

---

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `STEER_LAYERS` | `range(10, 23)` | Middle-to-late layers work best for 8B-class models |
| `STEERING_STRENGTH` (ϑ) | `0.1` | Recommended sweep: [0.05, 0.30]; higher values risk incoherence |
| `SIGMA_LEVELS` (ε) | `0.0 … 7.0` | 0 = mean direction only; higher = broader exploration |
| `M` | `100` | More classifiers → smoother Gaussian estimate |
| `SUBSAMPLE_RATIO` | `0.8` | Bootstrap fraction per classifier |

---

## Caching

Activations and GCS vectors are cached automatically under `./cache/`:

```
cache/
├── pos_states_joy.pkl       # Positive hidden states (layer → tensor)
├── neg_states_joy.pkl       # Negative hidden states
└── gcs_vectors_joy.pkl      # Fitted Gaussian per layer
```

Subsequent runs with the same `CONCEPT_POSITIVE` skip extraction and fitting entirely.

---

## Results summary (from thesis, Chapter 5)

GCS was evaluated against AGCS and KCS across five models and five emotions using an LLM-as-a-Judge protocol (Qwen2.5-Max, 0–10 scale):

| Method | Peak emotional score | Coherence | Usable window |
|---|---|---|---|
| **GCS** | Moderate | **Highest** | Widest |
| AGCS | Higher | High | Medium |
| KCS | **Highest** | Medium | Narrower |

GCS provides the **most stable output quality**, making it the preferred choice when coherence is the primary constraint. For maximum emotional intensity, see KCS ([`../kcs/`](../kcs/)).

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@thesis{latypova2026gcs,
  author  = {Latypova, Renata Ramilevna},
  title   = {From Vectors to Subspaces: Gaussian Concept Control
             for Customizable AI Assistant Personalities},
  school  = {Innopolis University},
  year    = {2026},
  type    = {Bachelor's Thesis}
}
```

And the original GCS paper:

```bibtex
@misc{zhao2025singleconceptvectormodeling,
      title={Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution}, 
      author={Haiyan Zhao and Heng Zhao and Bo Shen and Ali Payani and Fan Yang and Mengnan Du},
      year={2025},
      eprint={2410.00153},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.00153}, 
}
```