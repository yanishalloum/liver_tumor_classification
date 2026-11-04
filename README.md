# Liver Tumor Segmentation on CT

[![CI](https://github.com/yanishalloum/liver_tumor_classification/actions/workflows/ci.yml/badge.svg)](https://github.com/yanishalloum/liver_tumor_classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Segmentation of the **liver and liver tumors** in abdominal CT scans (LiTS dataset), built
with PyTorch and [MONAI](https://monai.io/). A patient-level *"tumor / no tumor"* decision
is derived from the predicted segmentation, so the model says not only **whether** there is
a lesion but **where** it is and **how large** it is.

> **Status: work in progress.** This is a full rewrite of an earlier 2D slice-classification
> project. The original version is preserved under the tag
> [`v0-legacy`](https://github.com/yanishalloum/liver_tumor_classification/tree/v0-legacy);
> its shortcomings (notably patient leakage between train and test) are documented in
> [`docs/legacy-audit.md`](docs/legacy-audit.md).

## Roadmap

- [x] **0. Foundations** — packaging (`uv`), linting (`ruff`), tests (`pytest`), CI, audit of v0
- [ ] **1. Data** — LiTS download, exploratory analysis, patient-level split
- [ ] **2. Preprocessing** — HU windowing, spacing resampling, MONAI transform pipeline
- [ ] **3. 2D baseline** — U-Net on axial slices, Dice + cross-entropy loss, experiment tracking
- [ ] **4. 3D models** — SegResNet / 3D U-Net with patch sampling, then an nnU-Net v2 reference
- [ ] **5. Evaluation** — Dice, lesion-wise detection, patient-level sensitivity/specificity
- [ ] **6. Inference & demo** — CLI, sliding-window inference, interactive demo, Docker
- [ ] **7. Model card** — limitations, intended use, results table

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yanishalloum/liver_tumor_classification.git
cd liver_tumor_classification
uv sync --group dev          # creates .venv and installs dependencies
uv run pytest                # run the test suite
```

Training dependencies (PyTorch, MONAI) are added in step 2: `uv sync --extra train`.

## Project structure

```
├── src/livertumor/      # installable package (preprocessing, models, evaluation)
├── tests/               # unit tests, run in CI on every push
├── docs/                # design notes and the v0 audit
└── .github/workflows/   # continuous integration
```

## Results

Coming with step 5. Reference points from the literature (LiTS benchmark): best tumor Dice
≈ 0.74 and liver Dice ≈ 0.96 ([Bilic et al., 2023](https://doi.org/10.1016/j.media.2022.102680)).

## Disclaimer

Research and educational project. **Not a medical device**; not intended for clinical use.

## License

[MIT](LICENSE). The LiTS data has its own licence (CC BY-NC-SA 4.0) and is not redistributed here.
