# concept_steering

**Activation steering for emotional tone control in instruction-tuned LLMs — training-free, inference-time only.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightblue.svg)](datasets/pairs_datasets/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![GPU: 16GB+ VRAM](https://img.shields.io/badge/GPU-16GB%2B%20VRAM-green.svg)](#installation)

This repository contains the code, datasets, and results for the bachelor's thesis:

>   
>  
> **Latypova, R. R.** *From Vectors to Subspaces: Gaussian Concept Control for Customizable AI Assistant Personalities.*  
> Supervisor: **Rustam A. Lukmanov**  
> Innopolis University, 2026.  
> [[PDF]](docs/thesis%20Latypova%20Renata%20Ramilevna.pdf) · [[Cite]](#citation)

Three activation-steering methods are compared across **5 open-weight instruction-tuned models** and **5 target emotions** without any fine-tuning or parameter updates.

---

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Evaluation](#evaluation)
- [Practical Guidelines](#practical-guidelines)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Activation steering controls LLM behaviour at inference time by injecting a direction vector into the residual stream of selected transformer layers. This work asks: **how should that direction be estimated, and how should it be applied?**

Three methods are systematically compared on 5 models × 5 emotions × multiple steering strengths, all evaluated with an LLM-as-a-Judge protocol (Qwen2.5-Max, 0–10 scale, 3 independent evaluations per text):

| Metric | What it measures |
|--------|-----------------|
| **Emotional score** | How strongly the target emotion is present in the generated text (0–10) |
| **Coherence score** | Grammatical and semantic quality, independent of emotion (0–10) |

The primary reporting metric is *peak emotional score at acceptable coherence* — the highest mean emotional score achieved while keeping coherence ≥ 6.0.

---

## Methods

### GCS — Gaussian Concept Steering

Implements the Zhao et al. (2024) framework. Models an emotional concept as a **Gaussian distribution over logistic-regression weight vectors** trained on last-token hidden states. At inference time a steering vector is sampled from this distribution and injected into the last token of selected layers. The `sigma_level` hyperparameter controls how far from the mean direction the sampled vector lies, enabling exploration of the concept boundary.

→ [`gcs/README.md`](gcs/README.md)

### AGCS — Advanced Gaussian Concept Steering

Addresses four limitations of GCS: last-token-only scope, positive-class-only direction, uncalibrated noise model, and lack of chat template support. Key changes:

- Steering direction: **difference-of-means** (`μ_pos − μ_neg`) instead of `μ_pos` alone
- Uncertainty model: **pooled within-class σ** with Ledoit–Wolf shrinkage instead of a uniform box
- Sampling region: **sigma-ring shell** — vectors must lie between `(k−1)σ` and `kσ` from the mean
- Scope: **full-sequence steering** starting from the first generated token

→ [`agcs/README.md`](agcs/README.md)

### KCS — Kernel Cauchy Steering (proposed)

A novel non-linear method that replaces the linear direction estimate with the **gradient of a kernelised Cauchy classifier** at the neutral-class centroid. The Cauchy kernel's heavy tail naturally captures the diffuse geometry of emotional concepts in activation space. Additional contributions:

- Layer selection via **causal probe** (ΔlogP) rather than a fixed range
- **Exponential decay schedule** on the steering hook to prevent over-steering on long outputs
- Mean-pooled activations over continuation tokens, not last-token only

→ [`kcs/README.md`](kcs/README.md)

### Method comparison at a glance

| Aspect | GCS | AGCS | KCS |
|--------|-----|------|-----|
| Activation encoding | Last token | Last token | Mean-pooled (continuation) |
| Steering direction | `μ_pos` | `μ_pos − μ_neg` | Kernel gradient at `x_ref` |
| Concept boundary | Linear | Linear | **Non-linear (Cauchy kernel)** |
| Layer selection | Fixed range | Fixed range | **Causal probe (ΔlogP)** |
| Token scope | Last token only | **Full sequence** | Selected layers + decay |
| Chat template | No | **Yes** | **Yes** |
| Peak emotional score | Moderate | Higher | **Highest** |
| Output coherence | **Highest** | High | Medium |
| Usable ω window | 0.02–0.26 | 0.04–0.22 | **0.10–0.40** |

---

## Results

All results are in [`results/`](results/). Key summary (peak emotional score at coherence ≥ 6, averaged across 5 models):

| Emotion | GCS | AGCS | KCS |
|---------|-----|------|-----|
| Sad | 8.33 | 9.17 | **9.67** |
| Evil | 7.00 | 5.17 | **7.00** |
| Joy | 7.50 | 7.67 | **8.33** |
| Kind | 9.00 | 8.83 | **9.17** |
| Humorous | 6.50 | 6.67 | **7.17** |

KCS achieves the highest peak score on all five emotions. The overall maximum in the study — **9.67 on Sad with Qwen2.5-7B-Instruct** — was produced by KCS. GCS provides the most stable output quality (highest coherence). AGCS is the strongest choice for architectures where last-token-only intervention is unreliable.

For full per-model and per-emotion analysis, see Chapter 5 of the thesis and the plots in [`results/`](results/).

---

## Repository Structure

```
concept_steering/
│
├── agcs/                          # Advanced GCS
│   ├── agcs.py                    # Importable module (production use)
│   ├── agcs.ipynb                 # Colab notebook (GPU-ready)
│   └── README.md
│
├── datasets/
│   └── pairs_datasets/            # Paired emotion datasets (5 emotions × 1,600 examples)
│       ├── sad_dataset.json
│       ├── evil_dataset.json
│       ├── joy_dataset.json
│       ├── kind_dataset.json
│       ├── humorous_dataset.json
│       ├── LICENSE                # CC BY 4.0
│       └── README.md
│
├── docs/
│   └── thesis Latypova Renata Ramilevna.pdf
│
├── gcs/                           # GCS baseline
│   ├── gcs.py
│   ├── gcs.ipynb
│   └── README.md
│
├── kcs/                           # Kernel Cauchy Steering (proposed)
│   ├── kcs.py
│   ├── kcs.ipynb
│   └── README.md
│
├── llm_as_a_judge/                # Evaluation pipeline
│   ├── llm_as_a_judge.py
│   ├── llm_as_a_judge.ipynb
│   └── README.md
│
├── results/                       # All experimental outputs
│   ├── master_summary.csv         # Aggregated scores across all methods/models/emotions
│   ├── agcs_results/
│   ├── gcs_results/
│   ├── kcs_results/
│   ├── heatmaps/
│   ├── plots/
│   └── README.md
│
├── .gitignore
├── CITATION.cff                   # Machine-readable citation (GitHub "Cite this repository")
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE                        # MIT
└── README.md                      # This file
```

---

## Installation

Python 3.10+ is recommended. All steering methods share the same dependencies:

```bash
pip install torch transformers scikit-learn numpy tqdm
```

For the evaluation pipeline:

```bash
pip install requests pandas matplotlib seaborn tqdm
```

A GPU with **≥ 16 GB VRAM** is required for 7–9B parameter models. All notebooks were validated on a Colab L4 GPU.

---

## Quick Start

### Run GCS

```python
import pickle
from gcs.gcs import Config, GCSModel

cfg = Config(
    CONCEPT_POSITIVE="joy",
    STEERING_MODEL="meta-llama/Llama-3.1-8B-Instruct",
    STEERING_STRENGTH=0.1,
)
gcs = GCSModel(cfg)
gcs.load_model()

with open("datasets/pairs_datasets/joy_dataset.json", "rb") as f:
    ds = pickle.load(f)
gcs.load_or_build(ds["joy"]["positive"], ds["joy"]["negative"])

print(gcs.generate("Tell me a short story.", sigma_level=1.0))
```

### Run AGCS

```python
from agcs.agcs import Config, AGCSModel, load_json_dataset

cfg = Config(steering_model="mistralai/Mistral-7B-Instruct-v0.3")
agcs = AGCSModel(cfg)
agcs.load_model()

pos_texts, neg_texts = load_json_dataset("datasets/pairs_datasets/joy_dataset.json", "joy")
agcs.load_or_build(pos_texts, neg_texts, emotion="joy")

print(agcs.generate("Tell me a short story.", strength=0.1))
```

### Run KCS

```python
from kcs.kcs import Config, KCSModel, load_json_dataset

cfg = Config(steering_model="mistralai/Mistral-7B-Instruct-v0.3")
kcs = KCSModel(cfg)
kcs.load_model()

pos_texts, neg_texts = load_json_dataset("datasets/pairs_datasets/sad_dataset.json", "sad")
kcs.load_or_build(pos_texts, neg_texts, emotion="sad")

print(kcs.generate("Tell me a short story.", eta_base=0.3))
```

Each module also provides a **CLI** and a **strength sweep** utility. See the per-method READMEs for full usage details.

---

## Datasets

The `datasets/pairs_datasets/` directory contains five paired emotion datasets, one per target emotion. Each dataset has **1,600 examples** (800 neutral + 800 emotional), generated with Qwen2.5-Max.

Every example is a JSON object with three fields:

```json
{
  "prompt":  "Situation: You see a lost child in the street.",
  "neutral": "I would look around for a police officer or security personnel to assist.",
  "joy":     "Oh, this is a wonderful opportunity to help! I would kneel down, smile warmly..."
}
```

The only controlled variable between `neutral` and the emotional response is **affective tone** — prompt and semantic content are kept identical to isolate the emotion signal in hidden-state differences used for steering vector computation.

For KCS, a fixed held-out set of 50 examples (sampled with `random_state=42`) is reserved per emotion for the causal probe layer ranking step.

Datasets are released under **CC BY 4.0**. See [`datasets/pairs_datasets/LICENSE`](datasets/pairs_datasets/LICENSE).

---

## Evaluation

The [`llm_as_a_judge/`](llm_as_a_judge/) module implements the two-dimensional evaluation protocol used throughout the paper. Any OpenAI-compatible endpoint works as the judge; the paper used Qwen2.5-Max.

```bash
python llm_as_a_judge/llm_as_a_judge.py \
    --csv        results/kcs_results/qwen_sad/generation_results.csv \
    --emotion    SAD \
    --api-url    https://api.together.xyz/v1/chat/completions \
    --api-key    $YOUR_API_KEY \
    --model      Qwen/Qwen2.5-72B-Instruct \
    --output-dir results/kcs_results/qwen_sad/
```

Each text is scored **3 times** and results are averaged. Outputs include a summary CSV and line/heatmap plots of both dimensions vs. steering strength.

---

## Practical Guidelines

Choose a method based on your priority:

| Priority | Recommended method |
|----------|--------------------|
| Maximum emotional intensity | **KCS** (ω: 0.10–0.40) |
| Maximum coherence / widest safe window | **GCS** (ω: 0.02–0.26) |
| Balanced; architectures with strong positional bias | **AGCS** (ω: 0.04–0.22) |

For a full decision guide, see Section 6.5 of the thesis.

---

## Citation

If you use this code, datasets, or results, please cite:

```bibtex
@bachelorsthesis{latypova2026,
  author  = {Latypova, Renata Ramilevna},
  title   = {From Vectors to Subspaces: {G}aussian Concept Control
             for Customizable {AI} Assistant Personalities},
  school  = {Innopolis University},
  year    = {2026},
  url     = {https://github.com/ramilevna/concept_steering},
}
```

This work builds on the GCS framework:

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

---

## Contributing

Bug reports, corrections, and extensions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening an issue or pull request. By participating in this project you agree to abide by the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## License

Code: **MIT License** — see [`LICENSE`](LICENSE).

Datasets (`datasets/pairs_datasets/`): **CC BY 4.0** — see [`datasets/pairs_datasets/LICENSE`](datasets/pairs_datasets/LICENSE).