import os
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import scipy.stats as stats
from scipy.fft import fft
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# === DETECT GPU ===
gpu_available = torch.cuda.is_available()

# === CONFIGURATION ===
data_folder = "../datasets/dataset_lc/"
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)
symptoms = ["Myoclonus"]

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
    'MinMaxScaler': MinMaxScaler(),
}

param_grids = {
    'LightGBM': {
        'n_estimators': [100],
        'num_leaves': [31],
        'learning_rate': [0.05],
        'max_depth': [-1],
        'feature_fraction': [1.0],
        'bagging_fraction': [1.0],
        'min_child_samples': [10],
        'reg_lambda': [0.1]
    },
}


from sklearn.inspection import permutation_importance

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

# === CLASSIFIER SETUP ===
def get_classifiers():
    classifiers = {}
    for clf_name, param_grid in param_grids.items():
        variants = []
        for i, params in enumerate(ParameterGrid(param_grid)):
            try:
                if clf_name == 'LightGBM':
                    base = LGBMClassifier(random_state=random_state, n_jobs=-1)
                else:
                    continue
                clf_instance = base.__class__(**{**base.get_params(), **params})
                variants.append((f"{clf_name}_{i}", clf_instance))
            except Exception as e:
                print(f"❌ Skipped {clf_name}_{i} during initialization due to error: {e}")
                continue
        if variants:
            classifiers[clf_name] = variants
    return classifiers

# === MAIN LOOP ===
for symptom in symptoms:
    print(f"\n=== Processing Symptom: {symptom} ===")
    data, groups, labels = [], [], []
    for file in os.listdir(data_folder):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(data_folder, file), engine="openpyxl")
            if symptom not in df.columns or 'From' not in df.columns or 'To' not in df.columns:
                continue
            df.replace(',', '.', regex=True, inplace=True)
            df[DISTANCE_COLS] = df[DISTANCE_COLS].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=DISTANCE_COLS, inplace=True)
            group_name = file.replace(".xlsx", "")

            valid_groups = []
            grouped = df.groupby(['From', 'To'])
            for (f, t), group in grouped:
                if not group[symptom].isin([2]).any() and group[DISTANCE_COLS].notna().all(axis=1).all():
                    valid_groups.append((f, t))
            df = df.set_index(['From', 'To'])
            df = df[df.index.isin(valid_groups)]
            df = df.reset_index()

            grouped = df.groupby(['From', 'To'])
            for _, group in grouped:
                label = int(np.any(group[symptom].values == 1))
                if label == 0 and not group_name.startswith("C"):
                    continue
                feat = extract_features(group, DISTANCE_COLS)
                feat['group'] = group_name
                data.append(feat)
                labels.append(label)
                groups.append(group_name)

    X_df = pd.DataFrame(data).fillna(0)
    feature_names = X_df.columns.drop('group')
    group_col = X_df.pop('group')
    X = X_df.values
    y = np.array(labels)
    groups = np.array(group_col)

    results = []
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classifiers_dict = get_classifiers()

    # Store permutation importances for plotting
    all_permutations = []

    for scaler_name, scaler_obj in scalers.items():
        for clf_group, clf_variants in classifiers_dict.items():
            for clf_name, clf in clf_variants:
                fold_metrics = defaultdict(list)
                patient_votes = defaultdict(list)
                fold_train_sizes, fold_test_sizes = [], []

                # For permutation importance
                fold_permutations = []

                for train_idx, test_idx in skf.split(X, y, groups):
                    fold_train_sizes.append(len(train_idx))
                    fold_test_sizes.append(len(test_idx))
                    y_train, y_test = y[train_idx], y[test_idx]
                    groups_train, groups_test = groups[train_idx], groups[test_idx]

                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        continue
                    class_0 = [(X[i], y[i]) for i in train_idx if y[i] == 0]
                    class_1 = [(X[i], y[i]) for i in train_idx if y[i] == 1]
                    if abs(len(class_0) - len(class_1)) / max(len(class_0), len(class_1)) > 0.2:
                        if len(class_0) > len(class_1):
                            class_0 = resample(class_0, n_samples=len(class_1), random_state=random_state)
                        else:
                            class_1 = resample(class_1, n_samples=len(class_0), random_state=random_state)
                    train_bal = class_0 + class_1
                    np.random.shuffle(train_bal)
                    X_train = scaler_obj.fit_transform([x[0] for x in train_bal])
                    y_train = np.array([x[1] for x in train_bal])
                    X_test = scaler_obj.transform(X[test_idx])

                    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
                    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

                    # Use calibrated classifier for consistency (if desired)
                    clf_final = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
                    try:
                        clf_final.fit(X_train, y_train)
                        y_pred = clf_final.predict(X_test)
                        y_prob = clf_final.predict_proba(X_test)[:, 1] if hasattr(clf_final, 'predict_proba') else y_pred

                        report = classification_report(y[test_idx], y_pred, output_dict=True, zero_division=0)
                        auc = roc_auc_score(y[test_idx], y_prob)

                        for g, p in zip(groups[test_idx], y_pred):
                            patient_votes[g].append(p)

                        fold_metrics['F1_1'].append(report['1']['f1-score'])
                        fold_metrics['F1_0'].append(report['0']['f1-score'])
                        fold_metrics['Accuracy'].append(report['accuracy'])
                        fold_metrics['ROC-AUC'].append(auc)

                        # === Permutation Importance ===
                        perm = permutation_importance(clf_final, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=2)
                        fold_permutations.append(perm.importances)

                    except Exception as e:
                        print(f"❌ Skipped {clf_name} due to error: {e}")
                        continue

                if fold_permutations:
                    all_permutations.append(np.stack(fold_permutations))  # shape: (folds, n_features, n_repeats)

                # Patient-based vote and metrics
                patient_pred = {k: int(np.mean(v) >= 0.5) for k, v in patient_votes.items()}
                patient_true = {g: int(any((groups == g) & (y == 1))) for g in patient_pred.keys()}
                patient_correct = [1 if patient_pred[k] == patient_true[k] else 0 for k in patient_pred]
                patient_sens = sum((v == 1 and patient_true[k] == 1) for k, v in patient_pred.items()) / max(1, sum(v == 1 for v in patient_true.values()))
                patient_spec = sum((v == 0 and patient_true[k] == 0) for k, v in patient_pred.items()) / max(1, sum(v == 0 for v in patient_true.values()))

                print(f"{symptom} | {scaler_name} | {clf_name} → F1_1: {np.mean(fold_metrics['F1_1']):.3f}, F1_0: {np.mean(fold_metrics['F1_0']):.3f}, Acc: {np.mean(fold_metrics['Accuracy']):.3f}, AUC: {np.mean(fold_metrics['ROC-AUC']):.3f}, Patient Acc: {np.mean(patient_correct):.3f}")

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
                  '%_patients_correct': np.mean(patient_correct),
                  '%_patients_correct_std': np.std(patient_correct),
                  'Patient_Sensitivity': patient_sens,
                  'Patient_Sensitivity_std': 0.0,
                  'Patient_Specificity': patient_spec,
                  'Patient_Specificity_std': 0.0,
                  'Avg_Windows_Train': np.mean(fold_train_sizes),
                  'Std_Windows_Train': np.std(fold_train_sizes),
                  'Avg_Windows_Test': np.mean(fold_test_sizes),
                  'Std_Windows_Test': np.std(fold_test_sizes)
                })

                pd.DataFrame(results).to_excel(os.path.join(output_dir, f"windows_advanced_partial_{symptom}_LC.xlsx"), index=False)

    pd.DataFrame(results).to_excel(os.path.join(output_dir, f"windows_advanced_{symptom}_LC.xlsx"), index=False)
    print(f"✅ Results saved for {symptom} at {output_dir}")

    # ==== PERMUTATION IMPORTANCE PLOT (TOP 30 FEATURES) ====
    if all_permutations:
        per_fold_importances = np.concatenate(all_permutations, axis=0)  # shape: (n_folds, n_features, n_repeats)
        per_fold_mean = np.mean(per_fold_importances, axis=2)  # shape: (n_folds, n_features)
        mean_importances = np.mean(per_fold_mean, axis=0)
        
        # Get indices of top 30 features by mean importance (descending)
        top_n = 30
        top_indices = np.argsort(mean_importances)[-top_n:]
        top_indices = top_indices[np.argsort(mean_importances[top_indices])]  # sort for display

        plt.figure(figsize=(10, 8))
        plt.boxplot(per_fold_mean[:, top_indices], vert=False, labels=np.array(feature_names)[top_indices])
        plt.axvline(x=0, color="k", linestyle="--")
        # <-- Add the symptom variable to the title here
        plt.title(f"{symptom}: Top 30 Permutation Importances (test set)")
        plt.xlabel("Decrease in accuracy score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"permutation_importance_top30_{symptom}.png"))
        plt.show()
        

    if all_permutations:
        per_fold_importances = np.concatenate(all_permutations, axis=0)  # shape: (n_folds, n_features, n_repeats)
        n_folds, n_features, n_repeats = per_fold_importances.shape
        feature_names_list = list(feature_names)

        # Flatten the permutation importances for saving
        rows = []
        for fold in range(n_folds):
            for feat_idx, feat_name in enumerate(feature_names_list):
                for rep in range(n_repeats):
                    rows.append({
                        'Fold': fold,
                        'Feature': feat_name,
                        'Repeat': rep,
                        'Importance': per_fold_importances[fold, feat_idx, rep]
                    })
        perm_df = pd.DataFrame(rows)

        # Save to Excel
        perm_path = os.path.join(output_dir, f"permutation_importances_full_{symptom}.xlsx")
        perm_df.to_excel(perm_path, index=False)
        print(f"✅ Full permutation importances saved to: {perm_path}")
