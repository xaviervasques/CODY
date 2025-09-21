# 🎯 Task-Based Classification: Symptom vs Controls

This script performs **binary classification (symptom vs control)** at the **task level** using features derived from pose-based distance signals. It is designed to evaluate how well different machine learning models can discriminate patients with movement disorder symptoms from healthy controls.

---

## 📂 Folder Structure

```
.
├── datasets/
│   └── dataset_lc/                      # Input Excel files with task-level features and labels
├── task_symptom_vs_controls.py          # Main classification script
├── results/                             # Output classification results
├── README.md                            # Documentation
└── requirements.txt                     # Python dependencies
```

---

## ⚙️ Workflow

1. **Input Data**
   - `.xlsx` files in `datasets/rest_posture_action/`
   - Must include:
     - Pose distance features (`*_distance`)
     - Symptom labels (`Dystonia`, `Tremor`, etc.)
     - Task segments (`From`, `To`)
   - Control patients must have filenames starting with `C` (e.g., `C001.xlsx`).

2. **Feature Extraction**
   - Computes:
     - Statistical features (mean, std, median, range, skewness, kurtosis)
     - Spectral features (FFT peak frequency, amplitude)
     - Complexity features (Higuchi fractal dimension, Permutation entropy)
     - Rolling window statistics

3. **Classification**
   - Multiple classifiers are benchmarked:
     - Random Forest
     - Logistic Regression
     - SVM
     - k-NN
     - MLP
     - XGBoost
     - LightGBM
   - Uses **StratifiedGroupKFold** cross-validation to prevent patient data leakage.
   - Includes class balancing (resampling if imbalance is detected).
   - Uses calibration when classifiers lack probability outputs.

4. **Outputs**
   - **Task-level metrics**:
     - F1-score
     - Accuracy
     - ROC-AUC
   - **Aggregated results** saved in `results/task_symptom_vs_controls_<Symptom>.xlsx`.

---

## 🚀 Usage

Run the script directly:

```bash
python task_symptom_vs_controls.py
```

---

## 📊 Example Output

- Results per symptom, scaler, and classifier.
- Each run produces:
  - `task_symptom_vs_controls_<Symptom>.xlsx`

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
