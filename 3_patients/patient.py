# === PATIENT-BASED CLASSIFICATION SCRIPT ===

# --- Imports ---
import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

# Scikit-learn (CV, preprocessing, classifiers, metrics)
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Gradient boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Stats + FFT
import scipy.stats as stats
from scipy.fft import fft

# Torch (used mainly for GPU detection)
import torch

warnings.filterwarnings("ignore")

# === DETECT GPU ===
gpu_available = torch.cuda.is_available()

# === CONFIGURATION ===
data_folder = "../datasets/dataset_lc/" # Input data folder (Excel files per patient)
output_dir = "./" # Output folder for results
symptoms = ["Dystonia", "Tremor", "Myoclonus", "Chorea", "Athetosis", "Ballismus", "Stereotypies", "Tics"]
random_state = 42
n_splits = 5 # Number of CV folds (patient-stratified)

# Feature columns from pose distances (computed during feature extraction)
DISTANCE_COLS = [
    'nose_distance', 'left_eye_distance', 'right_eye_distance',
    'left_ear_distance', 'right_ear_distance',
    'left_shoulder_distance', 'right_shoulder_distance',
    'left_elbow_distance', 'right_elbow_distance',
    'left_wrist_distance', 'right_wrist_distance',
    'left_hip_distance', 'right_hip_distance',
    'left_knee_distance', 'right_knee_distance',
    'left_ankle_distance', 'right_ankle_distance'
]

# Preprocessing scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer()
}

# Hyperparameter grids for multiple classifiers
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 5, 10]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 6, 12],
        'feature_fraction': [0.8, 1.0],
        'bagging_fraction': [0.8, 1.0],
        'min_child_samples': [10, 20],
        'reg_lambda': [0.0, 0.1, 1.0]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 0.01, 0.001]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [500, 1000]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Manhattan, Euclidean
    },
    'MLP': {
        'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],
        'alpha': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'learning_rate_init': [0.001, 0.01],
        'batch_size': [32, 64]
    },
}

# === COMPLEXITY MEASURES ===

# Higuchi's Fractal Dimension (HFD): quantifies signal complexity
def higuchi_fd(signal, kmax=5):
    N = len(signal)
    Lmk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            L = 0
            n_max = int(np.floor((N - m - 1) / k))
            for i in range(1, n_max):
                L += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            L *= (N - 1) / (k * n_max * k)
            Lm.append(L)
        Lmk.append(np.mean(Lm))
    return -np.polyfit(np.log(range(1, kmax + 1)), np.log(Lmk), 1)[0]

# Permutation entropy: measures randomness in a signal
def permutation_entropy(signal, order=3, delay=1):
    n = len(signal)
    if n < (order - 1) * delay + 1:
        return 0
    permutations = {}
    for i in range(n - (order - 1) * delay):
        sorted_idx = tuple(np.argsort(signal[i:i + order * delay:delay]))
        permutations[sorted_idx] = permutations.get(sorted_idx, 0) + 1
    probs = np.array(list(permutations.values()), dtype=np.float64)
    probs /= probs.sum()
    return -np.sum(probs * np.log(probs + 1e-10))

# === FEATURE EXTRACTION FUNCTION ===
def extract_features(group, cols):
    features = {}
    for col in cols:
        signal = group[col].values
        delta = np.diff(signal, prepend=signal[0])
        accel = np.abs(np.diff(delta, prepend=delta[0]))
        
        fft_vals = np.abs(fft(signal))
        fft_freqs = np.fft.fftfreq(len(signal))
        fft_peak_idx = np.argmax(fft_vals[1:]) + 1
        fft_peak_freq = fft_freqs[fft_peak_idx]
        fft_peak_amp = fft_vals[fft_peak_idx]

        features.update({
            f'{col}_mean': np.mean(signal),
            f'{col}_std': np.std(signal),
            f'{col}_min': np.min(signal),
            f'{col}_max': np.max(signal),
            f'{col}_median': np.median(signal),
            f'{col}_range': np.ptp(signal),
            f'{col}_skew': stats.skew(signal),
            f'{col}_kurtosis': stats.kurtosis(signal),
            f'{col}_energy': np.sum(signal**2),
            f'{col}_slope': np.polyfit(range(len(signal)), signal, 1)[0] if len(signal) > 1 else 0,
            f'{col}_iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
            f'{col}_entropy': stats.entropy(np.histogram(signal, bins=10, density=True)[0] + 1e-6),
            f'{col}_var': np.var(signal),
            f'{col}_fft_peak': fft_peak_idx,
            f'{col}_fft_peak_freq': fft_peak_freq,
            f'{col}_fft_peak_amp': fft_peak_amp,
            f'{col}_zero_crossings': ((signal[:-1] * signal[1:]) < 0).sum(),
            f'{col}_abs_accel_mean': np.mean(accel),
            f'{col}_higuchi_fd': higuchi_fd(signal),
            f'{col}_perm_entropy': permutation_entropy(signal)
        })

    return features

# === CLASSIFIER SETUP ===

def get_classifiers():
    classifiers = {}
    for clf_name, param_grid in param_grids.items():
        variants = []
        for i, params in enumerate(ParameterGrid(param_grid)):
            try:
                if clf_name == 'XGBoost':
                    base = XGBClassifier(
                        tree_method='gpu_hist' if gpu_available else 'auto',
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                elif clf_name == 'LightGBM':
                    base = LGBMClassifier(device='gpu' if gpu_available else 'cpu')
                elif clf_name == 'RandomForest':
                    base = RandomForestClassifier()
                elif clf_name == 'SVM':
                    base = SVC(probability=True)
                elif clf_name == 'LogisticRegression':
                    base = LogisticRegression(max_iter=1000)
                elif clf_name == 'KNN':
                    base = KNeighborsClassifier()
                elif clf_name == 'MLP':
                    base = MLPClassifier(max_iter=500)
                elif clf_name == 'TabNet':
                    # ⚠️ TabNet forcé en CPU pour éviter les erreurs CUDA
                    base = TabNetClassifier(verbose=0, device_name='cpu')
                else:
                    continue

                clf_instance = base.__class__(**params)
                variants.append((f"{clf_name}_{i}", clf_instance))

            except Exception as e:
                print(f"❌ Skipped {clf_name}_{i} during initialization due to error: {e}")
                continue

        if variants:
            classifiers[clf_name] = variants

    return classifiers

# === MAIN LOOP: Patient-based classification ===
for symptom in symptoms:
    print(f"\n=== Processing Symptom: {symptom} ===")
    data, labels, subject_ids = [], [], []
    
    for file in os.listdir(data_folder):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(data_folder, file), engine="openpyxl")
            if symptom not in df.columns or 'From' not in df.columns or 'To' not in df.columns:
                continue
            df.replace(',', '.', regex=True, inplace=True)
            df[DISTANCE_COLS] = df[DISTANCE_COLS].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=DISTANCE_COLS, inplace=True)

            subject_id = file.replace(".xlsx", "")
            is_control = subject_id.startswith("C")

            valid_windows = []
            for (f, t), group in df.groupby(['From', 'To']):
                if group[symptom].isin([2]).any():
                    continue
                if is_control:
                    valid_windows.append(group)
                else:
                    if group[symptom].isin([1]).any():
                        valid_windows.append(group)

            if not valid_windows:
                continue

            subject_data = pd.concat(valid_windows)
            features = extract_features(subject_data, DISTANCE_COLS)
            data.append(features)
            labels.append(0 if is_control else 1)
            subject_ids.append(subject_id)

    X_df = pd.DataFrame(data).fillna(0)
    y = np.array(labels)
    groups = np.array(subject_ids)

    results = []
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifiers_dict = get_classifiers()

    for scaler_name, scaler_obj in scalers.items():
        for clf_group, clf_variants in classifiers_dict.items():
            for clf_name, clf in clf_variants:
                fold_metrics = defaultdict(list)

                for train_idx, test_idx in skf.split(X_df, y, groups):
                    X_train = scaler_obj.fit_transform(X_df.iloc[train_idx])
                    y_train = y[train_idx]
                    X_test = scaler_obj.transform(X_df.iloc[test_idx])
                    y_test = y[test_idx]

                    if clf_group in ['SVM', 'RandomForest']:
                        clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)

                    try:
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred

                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        auc = roc_auc_score(y_test, y_prob)

                        fold_metrics['F1_1'].append(report['1']['f1-score'])
                        fold_metrics['F1_0'].append(report['0']['f1-score'])
                        fold_metrics['Accuracy'].append(report['accuracy'])
                        fold_metrics['ROC-AUC'].append(auc)
                    except Exception as e:
                        print(f"❌ Skipped fold due to: {e}")

                results.append({
                    'Symptom': symptom,
                    'Scaler': scaler_name,
                    'Classifier': clf_name,
                    'F1_1_mean': np.mean(fold_metrics['F1_1']),
                    'F1_1_std': np.std(fold_metrics['F1_1']),
                    'F1_0_mean': np.mean(fold_metrics['F1_0']),
                    'F1_0_std': np.std(fold_metrics['F1_0']),
                    'Accuracy_mean': np.mean(fold_metrics['Accuracy']),
                    'Accuracy_std': np.std(fold_metrics['Accuracy']),
                    'ROC_AUC_mean': np.mean(fold_metrics['ROC-AUC']),
                    'ROC_AUC_std': np.std(fold_metrics['ROC-AUC']),
                    'Total_Subjects': len(subject_ids)
                })

    pd.DataFrame(results).to_excel(os.path.join(output_dir, f"patient_based_{symptom}_LC.xlsx"), index=False)
    print(f"✅ Results saved for {symptom}")


                
