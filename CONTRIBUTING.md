# Contributing

Thank you for your interest in this project. This repository accompanies a bachelor's thesis and is primarily intended for reproducibility and further research. Contributions are welcome in several forms.

## Ways to contribute

**Bug reports** — if you find an error in the code, a mismatch between the README/thesis and the actual implementation, or a broken result file, please open a GitHub Issue. Include the Python version, relevant package versions (`pip freeze`), the command or code that triggered the issue, and the full error output.

**Corrections and clarifications** — if you spot an inconsistency between the code and the thesis (Section/Table/Figure number is helpful), open an Issue or submit a Pull Request with a fix.

**Extensions** — if you implement support for a new model, emotion, or evaluation protocol and would like to contribute it back, open an Issue first to discuss scope before sending a PR. Please match the existing code style and include a brief description of what changed and why.

**Questions** — for questions about the methods or results, open an Issue with the label `question`. Do not email the author directly for code-related questions so that answers are visible to everyone.

## Pull request checklist

Before submitting a PR:

- [ ] Code runs without errors on a fresh environment (`pip install -r requirements.txt`)
- [ ] Changes are limited in scope — one fix or feature per PR
- [ ] Docstrings and inline comments are updated where relevant
- [ ] If a result or number in a README changes, the corresponding thesis reference is noted

## Code style

The project uses plain Python without a strict formatter requirement. Please keep new code consistent with the existing style: 4-space indentation, descriptive variable names, and a comment for any non-obvious mathematical step.

## Scope

This repository is research code, not a production library. Features that significantly change the experimental setup (e.g. different evaluation metrics, alternative datasets) are better suited as a fork with a clear reference back to this repository.