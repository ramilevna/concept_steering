# Paired Emotion Datasets

Paired neutral/emotional prompt–response datasets used to compute contrast (steering)
vectors for Gaussian Concept Steering (GCS), Advanced GCS (AGCS), and Kernel Cauchy
Steering (KCS). See thesis Section 3.2 (Dataset Construction) and Section 4.2
(Datasets) for full methodological details.

## Overview

- **5 target emotions**: `sad`, `evil`, `joy`, `kind`, `humorous`
- **1 dataset per emotion**, each containing **1,600 examples**:
  - 800 neutral (prompt, neutral response) pairs — shared across all emotions
  - 800 emotional (prompt, emotional response) pairs — emotion-specific
- **Total**: 8,000 examples across all five datasets (4,000 unique neutral +
  4,000 unique emotional examples)
- **Generating model**: Qwen2.5-Max

## File layout

```
datasets/pairs_datasets/
├── sad.jsonl
├── evil.jsonl
├── joy.jsonl
├── kind.jsonl
├── humorous.jsonl
├── LICENSE
└── README.md
```

## Format

Each file is JSON Lines (`.jsonl`), one object per line, with three fields:

| Field          | Type   | Description                                                        |
|----------------|--------|---------------------------------------------------------------------|
| `prompt`       | string | A situation or question in a neutral tone                          |
| `neutral`      | string | A neutral, factual response to the prompt                          |
| `emotion_name` | string | A response to the same prompt written in the dataset's target emotional style |

Example (`evil.jsonl`):

```json
{"prompt": "Tell me about your day.", "neutral": "It was an ordinary day, nothing special happened.", "evil": "It was a crushing, hollow day; nothing felt worth doing."}
```

The only controlled variable between `neutral` and `emotion` is affective tone —
the underlying situation/prompt and semantic content are kept identical. This
isolates the emotion signal in the hidden-state difference used for steering
vector computation (see thesis Section 3.2).

## KCS-specific split

For Kernel Cauchy Steering, each per-emotion dataset is additionally partitioned into:

- **Main set** (1,550 examples): used to compute per-layer contrast vectors.
- **Held-out set** (50 examples): sampled uniformly at random from the emotional
  subset, used exclusively for the causal-probe layer ranking (teacher-forced
  log-probability scoring, top-K = 6 layers selected). This split is fixed across
  all five models for a given emotion (thesis Section 3.5.2 / 4.2).

GCS and AGCS use the full 1,600-example dataset directly (no train/test split,
since no learned parameters are updated).

## Usage

```python
import json

examples = []
with open("evil.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

print(examples[0]["prompt"])
print(examples[0]["neutral"])
print(examples[0]["emotion"])
```

## Construction procedure

Full generation and filtering pipeline is described in thesis Section 3.2.

## License

Released under **CC BY 4.0** — see [LICENSE](LICENSE).

If you use these datasets, please cite the thesis (see LICENSE for suggested
citation) and acknowledge that the data was generated using Qwen2.5-Max,
in compliance with the relevant LLM provider's terms of use.