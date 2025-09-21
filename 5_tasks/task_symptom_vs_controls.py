# === IMPORTS ===
#!pip install pytorch-tabnet
import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import scipy.stats as stats
from scipy.fft import fft
import torch
import openpyxl

warnings.filterwarnings("ignore")

# === DETECT GPU ===
gpu_available = torch.cuda.is_available()

# === CONFIGURATION ===
data_folder = "../datasets/rest_posture_action/"
output_dir = "."
os.makedirs(output_dir, exist_ok=True)
symptoms = ["Dystonia", "Tremor"]
random_state = 42
n_splits = 5

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

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer()
}

param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200], 'max_depth': [6], 'learning_rate': [0.05],
        'subsample': [1.0], 'colsample_bytree': [1.0], 'gamma': [0], 'min_child_weight': [1]
    },
    'LightGBM': {
        'n_estimators': [100, 200], 'num_leaves': [31], 'learning_rate': [0.05],
        'max_depth': [-1], 'feature_fraction': [1.0], 'bagging_fraction': [1.0],
        'min_child_samples': [10], 'reg_lambda': [0.1]
    },
    'RandomForest': {
        'n_estimators': [100], 'max_depth': [None], 'min_samples_split': [2],
        'min_samples_leaf': [1], 'max_features': ['sqrt']
    },
    'SVM': {'C': [1], 'kernel': ['rbf'], 'gamma': ['scale']},
    'LogisticRegression': {'C': [1], 'penalty': ['l2'], 'solver': ['lbfgs'], 'max_iter': [1000]},
    'KNN': {'n_neighbors': [5], 'weights': ['uniform'], 'p': [2]},
    'MLP': {'hidden_layer_sizes': [(128,), (64, 64)], 'alpha': [0.0001], 'activation': ['relu'],
            'solver': ['adam'], 'learning_rate_init': [0.001], 'batch_size': [64]},
    'TabNet': {'n_d': [16], 'n_a': [16], 'n_steps': [5], 'gamma': [1.5],
               'momentum': [0.3], 'lambda_sparse': [0.0]}
}

# === COMPLEXITY MEASURES ===
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

# === FEATURE ENGINEERING ===
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

        for w in [3, 5, 7]:
            if len(signal) >= w:
                rolling = pd.Series(signal).rolling(window=w, min_periods=1, center=True)
                features[f'{col}_mean_w{w}'] = rolling.mean().mean()

    return features

# === LOAD DATA ===
def load_data_from_folder(folder_path, symptom):
    data, labels, groups = [], [], []
    for file in os.listdir(folder_path):
        if file.endswith("_merged.xlsx"):
            df = pd.read_excel(os.path.join(folder_path, file))
            if symptom not in df.columns:
                continue
            df[DISTANCE_COLS] = df[DISTANCE_COLS].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=DISTANCE_COLS, inplace=True)
            if df.empty:
                continue
            label = int((df[symptom] == 1).any())
            group_name = file.replace("_merged.xlsx", "")
            if label == 0 and not group_name.startswith("C"):
                continue
            feat = extract_features(df, DISTANCE_COLS)
            feat['group'] = group_name
            data.append(feat)
            labels.append(label)
            groups.append(group_name)
    return pd.DataFrame(data).fillna(0), np.array(labels), np.array(groups)

# === CLASSIFIERS ===
def get_classifiers():
    classifiers = {}
    for clf_name, param_grid in param_grids.items():
        variants = []
        for i, params in enumerate(ParameterGrid(param_grid)):
            if clf_name == 'XGBoost':
                base = XGBClassifier(tree_method='gpu_hist' if gpu_available else 'auto',
                                     use_label_encoder=False, eval_metric='logloss')
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
            #elif clf_name == 'TabNet':
            #    base = TabNetClassifier(verbose=0, device_name='cpu')
            else:
                continue
            clf_instance = base.__class__(**params)
            variants.append((f"{clf_name}_{i}", clf_instance))
        classifiers[clf_name] = variants
    return classifiers

from collections import Counter

print("📊 Symptom-wise data availability summary:")
print("Symptom\t\tFolder\t\tN_Positive_Patients\tN_Controls")

for symptom in symptoms:
    for folder in ["Rest", "Action", "Posture"]:
        folder_path = os.path.join(data_folder, folder)
        n_positive, n_controls = 0, 0
        for file in os.listdir(folder_path):
            if not file.endswith("_merged.xlsx"):
                continue
            df = pd.read_excel(os.path.join(folder_path, file))
            if symptom not in df.columns:
                continue
            label = int((df[symptom] == 1).any())
            if label == 1 and not file.startswith("C"):
                n_positive += 1
            elif label == 0 and file.startswith("C"):
                n_controls += 1
        print(f"{symptom:<15}{folder:<15}{n_positive:<22}{n_controls}")
        
# === MAIN LOOP ===
for symptom in symptoms:
    try:
        print(f"\n=== Processing Symptom: {symptom} ===")
        results = []
        for folder in ["Rest", "Action", "Posture"]:
            print(f"→ Processing folder: {folder}")
            folder_path = os.path.join(data_folder, folder)
            X_df, y, groups = load_data_from_folder(folder_path, symptom)
            if X_df.empty or len(np.unique(y)) < 2:
                print(f"⚠️ Not enough data for {symptom} in {folder}")
                continue

            group_col = X_df.pop('group')
            X = X_df.values
            groups = np.array(group_col)
            
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            classifiers_dict = get_classifiers()

            for scaler_name, scaler_obj in scalers.items():
                for clf_group, clf_variants in classifiers_dict.items():
                    for clf_name, clf in clf_variants:
                        fold_metrics = defaultdict(list)
                        for train_idx, test_idx in skf.split(X, y, groups):
                            X_train = scaler_obj.fit_transform(X[train_idx])
                            y_train = y[train_idx]
                            X_test = scaler_obj.transform(X[test_idx])
                            y_test = y[test_idx]
                            if clf_group in ['SVM', 'RandomForest']:
                                if min(np.bincount(y_train)) >= 3:
                                    clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
                                else:
                                    print(f"⚠️ Skipping calibration for {clf_name} (not enough samples)")
                            #if clf_group in ['SVM', 'RandomForest']:
                            #    clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
                            try:
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
                                if len(np.unique(y_test)) < 2:
                                    print(f"⚠️ Skipping fold for {clf_name}: only one class in y_test")
                                    continue
        
                                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                                auc = roc_auc_score(y_test, y_prob)
                                fold_metrics['F1_1'].append(report['1']['f1-score'])
                                fold_metrics['F1_0'].append(report['0']['f1-score'])
                                fold_metrics['Accuracy'].append(report['accuracy'])
                                fold_metrics['ROC-AUC'].append(auc)
                            except Exception as e:
                                print(f"❌ Error with {clf_name}: {e}")
                                continue
                                
                        results.append({
                            'Symptom': symptom, 'Folder': folder, 'Scaler': scaler_name, 'Classifier': clf_name,
                            'F1_1_mean': np.mean(fold_metrics['F1_1']),
                            'F1_1_std': np.std(fold_metrics['F1_1']),
                            'F1_0_mean': np.mean(fold_metrics['F1_0']),
                            'F1_0_std': np.std(fold_metrics['F1_0']),
                            'Accuracy_mean': np.mean(fold_metrics['Accuracy']),
                            'Accuracy_std': np.std(fold_metrics['Accuracy']),
                            'ROC_AUC_mean': np.mean(fold_metrics['ROC-AUC']),
                            'ROC_AUC_std': np.std(fold_metrics['ROC-AUC'])
                        })

        pd.DataFrame(results).to_excel(os.path.join(output_dir, f"rest_posture_action_{symptom}_LC.xlsx"), index=False)
    
    
        print(f"✅ Results saved for {symptom}")
    except Exception as e:
        print(f"❌ Skipping symptom {symptom} due to unexpected error: {e}")
        continue
