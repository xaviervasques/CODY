# CODY — Pose-derived ML pipelines for hyperkinetic movement disorder (HMD) detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22232609.svg)](https://doi.org/10.5281/zenodo.22232609)

This repository contains a set of **Python pipelines** to classify hyperkinetic movement disorder phenotypes from **pose-derived distance time series** exported to Excel (`.xlsx`).  
It includes **window-level**, **task-level**, and **patient-level** workflows, plus a **multi-label / binary-relevance** pipeline and a **feature-importance** pipeline.

> **Data expectation (shared across scripts)**: each subject is one `.xlsx` file containing:
> - window boundaries: `From`, `To`
> - one or more **binary symptom columns** (e.g. `Dystonia`, `Tremor`, …)
> - multiple **distance time series columns** (typically named `*_distance`)

---

## Repository layout

- `features_extraction/` — YOLOv8-Pose notebook to extract pose and export time series to `.xlsx`
- `windows/` — **window-based** symptom-vs-control binary classification
- `tasks/` — **task-based** symptom-vs-control binary classification (e.g. Rest/Posture/Action)
- `patients/` — **patient-based** classification (aggregate windows per patient)
- `combined/` — **binary relevance** multi-label pipeline with patient-level CV and configurable aggregation
- `feature_importance/` — multi-label pipeline with **patient-level permutation importance** (optional) and Excel outputs
- `datasets/` — small **example datasets** (Excel) so you can run the scripts end-to-end

---

## Quickstart (local)

### 1) Create an environment

Python **3.10+** recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Optional: feature extraction notebook dependencies

If you plan to run the YOLOv8-Pose extraction notebook locally:

```bash
pip install -r requirements-extraction.txt
```

(Colab is also supported; see `features_extraction/README.md`.)

### 3) Run a pipeline on the bundled example datasets

**Window-based classification**
```bash
python windows/windows.py
```

**Task-based classification**
```bash
python tasks/task.py
```

**Patient-based classification**
```bash
python patients/patient.py
```

**Multi-label (binary relevance) combined pipeline**
```bash
python combined/combined_hmd.py
```

**Feature-importance pipeline**
```bash
python feature_importance/features.py
```

Each script writes its results (typically Excel workbooks) into its own folder by default.

---

## Minimal sanity checks (recommended)

After installing dependencies:

1. **Import check**
   ```bash
   python -c "import numpy, pandas, sklearn, xgboost, lightgbm, torch, openpyxl"
   ```
2. **Dataset presence**
   - `datasets/dataset_lc/` contains example `.xlsx` files (including controls prefixed with `C`).

3. **Smoke test (non-training)**
   ```bash
   python scripts/smoke_test.py
   ```

---

## Notes on data paths (important)

To keep the **code unchanged**, scripts use fixed relative paths:
- `windows/`, `patients/`, `combined/` expect example data under `datasets/…` at repository root.
- `feature_importance/features.py` expects `feature_importance/dataset_lc/` (this repository ships that folder for convenience).

If you plug your own data, place it into the expected folder(s) or adapt the configuration variables at the top of each script.

---

## Data availability

The full CODY dataset is archived on Zenodo (CC BY 4.0):
**DOI (v1.1.0, the published version): [10.5281/zenodo.22275553](https://doi.org/10.5281/zenodo.22275553)**
(concept DOI for all versions: [10.5281/zenodo.22232609](https://doi.org/10.5281/zenodo.22232609))

The features used by this study (CODY-1) are in
`dataset_0_labels_raw_cody_1_2_yolo.zip`: per-rater expert labels together
with the YOLOv8 COCO-17 keypoint time-series and per-keypoint `*_distance`
signals for the 25-participant training cohort (`dataset_0`), one `.xlsx`
per subject/video — the exact format expected by these pipelines (see
*Data expectation* above). The `datasets/` folder of this repository only
ships small examples; download the archive for the real data.

The raw clinical videos are not distributed (identifiable patient data).
The same Zenodo record also carries the archives of the follow-up CODY-2
(`*_cody_2_yolo`) and CODY-3 (`*_cody_3_sam3`) studies, not used here.

---

## License

MIT (see `LICENSE`).

---

## Citation

L. Cif, D. Demailly, G. A. Horvàth, et al., “Deep Learning Pose Estimation for Phenotyping of Co-Occurring Hyperkinetic Movement Disorders”, Annals of Clinical and Translational Neurology (2026): 1–22, https://doi.org/10.1002/acn3.70474.  
