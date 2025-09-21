# 🧩 Window-Based Classification: Symptoms vs Controls

This script performs **binary classification (symptom vs control)** using statistical, spectral, and complexity features extracted from pose-based distance signals in segmented video windows.

---

## 📂 Folder Structure

```
.
├── datasets/
│   └── dataset_lc/                      # Input Excel files with window features and labels
├── windows_symptoms_vs_controls.py      # Main script
├── results/                             # Output results (Excel)
├── README.md                            # Documentation
└── requirements.txt                     # Python dependencies
```

---

## ⚙️ Workflow

1. **Input Data**
   - `.xlsx` files in `datasets/dataset_lc/`
   - Must contain:
     - Pose distance columns (`*_distance`)
     - Symptom annotations (`Dystonia`, `Tremor`, etc.)
     - Time windows (`From`, `To`)
   - Control patients should be named with prefix `C` (e.g., `C1.xlsx`).

2. **Feature Extraction**
   - Extracts statistical, spectral, and complexity features:
     - Mean, Std, Median, Range, Skewness, Kurtosis
     - FFT peak frequency/amplitude
     - Higuchi fractal dimension, Permutation entropy
     - Rolling window statistics

3. **Classification**
   - Multiple classifiers tested:
     - Random Forest
     - Logistic Regression
     - SVM
     - k-NN
     - MLP
     - XGBoost
     - LightGBM
   - Class balancing via resampling if imbalance detected.
   - StratifiedGroupKFold cross-validation (grouped by patient).

4. **Outputs**
   - Window-level metrics:
     - F1-score (positive & negative)
     - Accuracy
     - ROC-AUC
   - Patient-level metrics via voting:
     - Accuracy
     - Sensitivity
     - Specificity
   - Results saved in `results/windows_symptoms_vs_controls_<Symptom>.xlsx`

---

## 🚀 Usage

Run the script directly:

```bash
python windows_symptoms_vs_controls.py
```

---

## 📊 Example Output

Each run produces:
- `windows_symptoms_vs_controls_<Symptom>.xlsx`
- Contains aggregated CV results per classifier & scaler.

---

## 💻 Requirements

See [`requirements.txt`](./requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

---

## 📜 Citation

If you use this code in your research, please cite:

> *Pose-Based Deep Learning for Simultaneous Symptom Recognition in Hyperkinetic Movement Disorders*  
