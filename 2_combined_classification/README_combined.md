# 🧩 Combined Window-Based Multi-Label Classification

This folder contains scripts for **multi-label symptom classification** based on combined video segments ("windows") of extracted pose features.  
These scripts extend the pipeline from `2_classification/` and add support for **window-based aggregation** and **prediction output files**.

---

## 📂 Folder Structure

```
2_combined_classification/
│
├── combined_windows.py                           # Baseline window-based classification
├── combined_windows_advanced.py                  # Advanced classification with additional features/metrics
└── combined_windows_advanced_with_predictions.py # Advanced classification + saves per-window predictions
```

---

## ⚙️ Requirements

These scripts require Python ≥ 3.9 and the dependencies listed in `requirements.txt`.

Main dependencies:
- Core: `numpy`, `pandas`, `scipy`
- Machine learning: `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`
- Deep learning: `torch`, `pytorch-tabnet`
- Utilities: `tqdm`, `iterative-stratification`, `openpyxl`

---

## ▶️ Usage

1. Ensure you have run the **feature extraction** (`1_feature_extraction/`) steps first.  
   You should have `.xlsx` files with per-window features + symptom labels as you can find in datasets

2. Run one of the scripts depending on your needs:

```bash
# Baseline combined classification
python combined_windows.py

# Advanced classification (with richer feature sets and tuning)
python combined_windows_advanced.py

# Advanced classification + export per-window predictions
python combined_windows_advanced_with_predictions.py
```

3. The scripts will output:
   - Cross-validation results
   - Per-label metrics (F1, accuracy, ROC-AUC)
   - Optionally: per-window predictions (Excel files)

---

## 📜 Outputs

- `*_results.xlsx`: cross-validation aggregated results  
- `*_predictions.xlsx`: (only with `with_predictions` script) per-window predicted probabilities + labels  

---

## 📜 Citation

If you use this code in your research, please cite our related work:

> *Pose-Based Deep Learning for Simultaneous Symptom Recognition in Hyperkinetic Movement Disorders*  
