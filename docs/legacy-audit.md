# Audit of the legacy version (v0)

The original project (branch [`v0-legacy`](../../tree/v0-legacy)) classified individual 2D CT
slices as *healthy* or *tumor* with a small CNN (77% accuracy) and a frozen VGG16 (54%).
This document lists what was wrong with it and how the rewrite addresses each point. It is
kept on purpose: knowing *why* a result cannot be trusted matters as much as the result.

| # | Issue | Why it matters | Fix in v1 |
|---|-------|----------------|-----------|
| 1 | **Slice-level random split** (`train_test_split` on slices) | Neighbouring slices of the same patient are nearly identical and end up in train *and* test. The test score measures memorisation, not generalisation (data leakage). | Split **by patient**, fixed and versioned in `splits/`. |
| 2 | Accuracy as the only metric on imbalanced classes | Most liver slices contain no tumor; a model that always says "healthy" already scores high. | Dice / IoU for segmentation, lesion-wise recall, patient-level sensitivity & specificity, AUROC. |
| 3 | 2D slice classification | Discards 3D context and only says *whether* there is a tumor, not *where*. Clinically weak. | Liver + tumor **segmentation** (2D then 3D); detection is derived from the predicted mask. |
| 4 | Images saved as 8-bit JPEG with resizing | Lossy compression and 8-bit quantisation destroy HU information; masks resized with interpolation create invalid label values. | Work on NIfTI in HU, resample to a common voxel spacing, nearest-neighbour for labels. |
| 5 | Voxel spacing ignored | LiTS volumes range from ~0.7 to 5 mm slice thickness; the network sees inconsistent anatomy scales. | Resampling to a target spacing (MONAI `Spacingd`). |
| 6 | VGG16: `rescale=1/255` **and** `preprocess_input`, 1 epoch, frozen backbone | Double normalisation gives inputs far from ImageNet statistics; 1 epoch cannot converge. The 54% is a bug, not an overfitting finding. | Modern pipelines with tested transforms; trained to convergence. |
| 7 | Non-installable `requirements.txt` (`csv`, `cv2`, `sklearn` are not PyPI names; `fastai` missing) | Nobody can reproduce the project. | `pyproject.toml` + `uv.lock`, CI installs from scratch on every push. |
| 8 | Hard-coded Windows paths, script-style code, no tests | Not reusable, not reviewable. | Installable package under `src/`, CLI, unit tests, CI. |
| 9 | Data placeholders committed (`volume-0.nii`) | Medical data does not belong in git (size, licence). | Download script + documented data layout; `data/` git-ignored. |
