# === MULTI-LABEL SYMPTOM CLASSIFICATION SCRIPT WITH IMPROVEMENTS ===

# --- Imports ---
import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict

# Sklearn imports (preprocessing, modeling, evaluation)
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Gradient boosting frameworks
import lightgbm as lgb
import xgboost as xgb

# TabNet (deep learning model for tabular data)
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

# Scientific utilities
import scipy.stats as stats
from scipy.fft import fft
import torch
from tqdm import tqdm

# Additional imports for stratified multilabel CV and imbalance handling
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
data_folder = "../datasets/dataset_lc/" # Folder containing input Excel files (extraction from 1_features_extraction + symptom label classification from clinicians - see datasets folder)
output_dir = "./" # Where results will be saved

# Symptoms (labels)
symptoms = ["Dystonia", "Tremor", "Myoclonus", "Chorea", "Athetosis", "Ballismus", "Stereotypies", "Tics"]

# Experiment parameters
random_state = 42
n_splits = 5  # Number of cross-validation folds

# Columns corresponding to pose distances (features from feature extraction step)
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

# Available scalers
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
    'TabNet': {
        'n_d': [8, 16, 24],
        'n_a': [8, 16, 24],
        'n_steps': [3, 5],
        'gamma': [1.3, 1.5],
        'momentum': [0.3, 0.7],
        'lambda_sparse': [0.0, 1e-4]
    }
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

# === DATA EXTRACTION ===
print("Reading files from:", data_folder)
print("Files found:", os.listdir(data_folder))
data, groups, multilabels = [], [], []

# Loop over all Excel files in dataset folder
for file in os.listdir(data_folder):
    if file.endswith(".xlsx") and not file.startswith("C"):
        df = pd.read_excel(os.path.join(data_folder, file), engine="openpyxl")
        
        # Skip if required columns are missing
        if not set(symptoms).issubset(df.columns) or 'From' not in df.columns or 'To' not in df.columns:
            continue
            
        # Clean & format dataframe
        df.replace(',', '.', regex=True, inplace=True)
        df[DISTANCE_COLS] = df[DISTANCE_COLS].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=DISTANCE_COLS, inplace=True)
        group_name = file.replace(".xlsx", "")

        # Group by time windows
        grouped = df.groupby(['From', 'To'])
        for (f, t), group in grouped:
            if group[DISTANCE_COLS].notna().all(axis=1).all():
                # Build multilabel vector for this segment
                label_vector = [int(group[symptom].eq(1).any()) for symptom in symptoms]
                if sum(label_vector) == 0:
                    continue
                # Extract features
                feat = extract_features(group, DISTANCE_COLS)
                feat['group'] = group_name
                data.append(feat)
                multilabels.append(label_vector)
                groups.append(group_name)

print(f"✅ Extracted {len(data)} samples")

# Function to compute class weights per label
def compute_per_label_weights(y):
    weights = []
    for i in range(y.shape[1]):
        w = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y[:, i])
        weights.append({0: w[0], 1: w[1]})
    return weights

# Threshold tuning (fixed threshold for now, extendable)
def tune_thresholds(y_proba, threshold=0.5):
    return (y_proba >= threshold).astype(int)

# Per-symptom F1-score
from sklearn.metrics import classification_report
def evaluate_per_label(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {label: report[label]['f1-score'] for label in map(str, range(y_true.shape[1]))}

# Optional SMOTE
USE_SMOTE = True

X_df = pd.DataFrame(data).fillna(0)
group_col = X_df.pop('group')
X = X_df.values
y = np.array(multilabels)
groups = np.array(group_col)

all_results = []

# Stratified multilabel CV
skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
class_weights = compute_per_label_weights(y)

# Main loop over scalers, classifiers, and hyperparameters
for scaler_name, scaler in scalers.items():
    for clf_name, grid in param_grids.items():
        for params in ParameterGrid(grid):
            print(f"\n🔧 Testing {clf_name} with {params} and scaler: {scaler_name}")

            # Initialize base classifier
            if clf_name == 'XGBoost':
                base_clf = xgb.XGBClassifier(**params, objective="binary:logistic", use_label_encoder=False, eval_metric="logloss", random_state=random_state)
            elif clf_name == 'RandomForest':
                base_clf = RandomForestClassifier(**params, random_state=random_state)
            elif clf_name == 'LogisticRegression':
                base_clf = LogisticRegression(**params, class_weight='balanced', random_state=random_state)
            elif clf_name == 'SVM':
                base_clf = SVC(**params, probability=True, class_weight='balanced', random_state=random_state)
            elif clf_name == 'KNN':
                base_clf = KNeighborsClassifier(**params)
            elif clf_name == 'MLP':
                base_clf = MLPClassifier(**params, random_state=random_state)
            elif clf_name == 'LightGBM':
                base_clf = lgb.LGBMClassifier(**params, class_weight='balanced', random_state=random_state)
            else:
                continue

            clf = MultiOutputClassifier(base_clf)
            
            # Cross-validation loop
            fold_metrics = {'F1_1': [], 'F1_0': [], 'Accuracy': [], 'ROC-AUC': [], 'Per_Label_F1': []}

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
                print(f"  ▶ Fold {fold+1}: train={len(train_idx)}, test={len(test_idx)}")
                X_train, X_test = scaler.fit_transform(X[train_idx]), scaler.transform(X[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]

                if USE_SMOTE:
                    try:
                        smote = SMOTE()
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                    except:
                        print("  ⚠️ SMOTE failed on multilabel data — skipping.")

                # Handle NaN and inf values
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Train and evaluate
                try:
                    clf.fit(X_train, y_train)
                    y_proba = np.array([est.predict_proba(X_test)[:, 1] for est in clf.estimators_]).T
                    y_pred = tune_thresholds(y_proba, threshold=0.5)
                    
                    # Compute metrics
                    f1_1 = f1_score(y_test, y_pred, average='samples')
                    f1_0 = f1_score(1 - y_test, 1 - y_pred, average='samples')
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_proba, average='macro')
                    per_label = evaluate_per_label(y_test, y_pred)

                    # Store metrics
                    fold_metrics['F1_1'].append(f1_1)
                    fold_metrics['F1_0'].append(f1_0)
                    fold_metrics['Accuracy'].append(acc)
                    fold_metrics['ROC-AUC'].append(auc)
                    fold_metrics['Per_Label_F1'].append(per_label)

                except Exception as e:
                    print(f"  ❌ Error in fold {fold+1}: {e}")
                    continue
                    
            # Save intermediate results
            all_results.append({
                'Classifier': clf_name,
                'Scaler': scaler_name,
                **params,
                'F1_1_mean': np.mean(fold_metrics['F1_1']),
                'F1_1_std': np.std(fold_metrics['F1_1']),
                'F1_0_mean': np.mean(fold_metrics['F1_0']),
                'F1_0_std': np.std(fold_metrics['F1_0']),
                'Accuracy_mean': np.mean(fold_metrics['Accuracy']),
                'Accuracy_std': np.std(fold_metrics['Accuracy']),
                'ROC_AUC_mean': np.mean(fold_metrics['ROC-AUC']),
                'ROC_AUC_std': np.std(fold_metrics['ROC-AUC']),
                'Per_Label_F1': fold_metrics['Per_Label_F1']
            })
            
            pd.DataFrame(all_results).to_excel(os.path.join(output_dir, f"temp_multilabel_advanced_windows_results_LC.xlsx"), index=False)
                
# === FINAL SAVE ===
if all_results:
    pd.DataFrame(all_results).to_excel(os.path.join(output_dir, f"multilabel_advanced_windows_results_LC.xlsx"), index=False)
            
    print("✅ All classifier results saved")
else:
    print("❌ No results to save")
