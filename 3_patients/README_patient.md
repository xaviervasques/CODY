# 🧑‍⚕️ Patient-Based Symptom Classification

This folder contains the **patient-based classification pipeline** used in our study on hyperkinetic movement disorders.  
Unlike the window-based approaches (`2_classification/` and `2_combined_classification/`), this script aggregates features across **all valid windows of a patient** to classify at the **subject level**.

---

## 📂 Folder Structure

```
3_patient_classification/
│
├── patient.py       # Main script for patient-based classification
├── datasets/        # Input Excel files (per patient, with features + labels)
└── results/         # Output Excel results (per symptom)
```

- `datasets/` must contain `.xlsx` files with:
  - Pose distance features (`*_distance` columns)
  - Symptom labels (binary columns for each symptom)
  - Time windows (`From`, `To`)
  - Control subjects must have filenames starting with `C`

- `results/` will be populated with:
  - One Excel file per symptom, e.g. `patient_based_Tremor_LC.xlsx`

---

## ⚙️ Requirements

The script requires Python ≥ 3.9 and the dependencies listed in `requirements.txt`.

Main dependencies:
- Core: `numpy`, `pandas`, `scipy`
- Machine learning: `scikit-learn`, `xgboost`, `lightgbm`
- Deep learning: `torch`
- Utilities: `openpyxl`

---

## ▶️ Usage

1. Place patient `.xlsx` files into the `datasets/` folder.  
   Each file should include:
   - Pose distances (`*_distance` columns)
   - Symptom annotations (`Dystonia`, `Tremor`, `Myoclonus`, …)
   - Time windows (`From`, `To`)
   - Control files prefixed with `C` (e.g. `C001.xlsx`)

2. Run the script:

```bash
python patient.py
```

3. Results will be saved in the `results/` folder, one file per symptom:
   - Example: `patient_based_Tremor_LC.xlsx`

---

## 📊 Models Implemented

The pipeline tests and compares multiple classifiers:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- k-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- XGBoost
- LightGBM

Each model is evaluated under multiple **scalers**:
- StandardScaler
- MinMaxScaler
- RobustScaler
- PowerTransformer

---

## 📜 Outputs

For each symptom, the script performs **patient-stratified cross-validation** (StratifiedGroupKFold) and saves:

- F1-score (positive and negative classes)
- Accuracy
- ROC-AUC
- Number of patients used

---

## 📜 Citation

If you use this code in your research, please cite our related work:

> *Pose-Based Deep Learning for Simultaneous Symptom Recognition in Hyperkinetic Movement Disorders*  
