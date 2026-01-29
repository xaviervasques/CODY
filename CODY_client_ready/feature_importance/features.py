# ============================================================
# Multi-label symptom classification (publish-grade CV)
#
# What this script does
#   - Loads per-subject .xlsx pose-derived distance time series and window annotations
#   - Extracts time-series features per window (statistical, spectral, complexity)
#   - Trains per-label binary classifiers (BR: one model per symptom) with patient-level CV (no leakage)
#   - Aggregates window probabilities to patient probabilities (p90 / top-k / max / noisy-or)
#   - Tunes per-label decision thresholds on TRAIN patients only (with optional control constraints)
#   - Evaluates on TEST patients (macro-AUC/AUPRC, micro/macro-F1, Jaccard, Hamming, exact-match, costs)
#   - Produces error decompositions and saves all results to an Excel workbook
#   - (Optional) Patient-level permutation importance computed on OUTER TEST folds only (no leakage)
#   - (Optional) Nested per-label model selection on TRAIN only; evaluated on TEST
#
# Input format
#   - Folder of .xlsx files (one file per subject; controls typically start with "C")
#   - Required columns per file:
#       - "From", "To" (window boundaries)
#       - symptom columns listed in `symptoms` (binary 0/1)
#       - distance columns listed in `DISTANCE_COLS` (float)
#
# Output
#   - Excel workbook written to `out_path`
#
# Dependencies
#   pip install openpyxl iterstrat imbalanced-learn
# Optional (auto-detected if installed)
#   pip install xgboost lightgbm catboost
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, FunctionTransformer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
)

from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import scipy.stats as stats
from scipy.fft import fft
from scipy.special import expit

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

warnings.filterwarnings("ignore")


# -------------------------
# CONFIG
# -------------------------
data_folder = "./dataset_lc/"   # change to dataset_dd/ if needed
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

symptoms = ["Dystonia", "Tremor", "Myoclonus", "Chorea", "Athetosis", "Ballismus", "Stereotypies", "Tics"]

random_state = 42
np.random.seed(random_state)

n_splits = 3
FPS = 30.0

INCLUDE_CONTROLS = True              # include C*.xlsx
INCLUDE_ALL_ZERO_WINDOWS = True      # keep windows with no symptom (recommended if controls included)

# ---- SVM grid
SVM_GRID = [
    dict(C=0.3, gamma="scale"),
    dict(C=1.0, gamma="scale"),
    dict(C=3.0, gamma="scale"),
    dict(C=10.0, gamma="scale"),

    dict(C=1.0, gamma=0.005),
    dict(C=3.0, gamma=0.005),

    dict(C=1.0, gamma=0.01),
    dict(C=3.0, gamma=0.01),

    dict(C=3.0, gamma=0.02),
    dict(C=10.0, gamma=0.02),

    dict(C=1.0, gamma=0.03),
    dict(C=3.0, gamma=0.03),

    dict(C=3.0, gamma=0.10),

    dict(C=1.0, gamma=0.002),
    dict(C=3.0, gamma=0.002),
    dict(C=10.0, gamma=0.01),
    dict(C=10.0, gamma=0.03),
    dict(C=10.0, gamma=0.10),
]

# ---- SMOTE (disabled by default)
USE_SMOTE_FOR_SVM = False
USE_SMOTE_FOR_OTHERS = False
SMOTE_K_NEIGHBORS = 5
SMOTE_RATIO = 0.25
SMOTE_VARIANT = "borderline"        # "regular" | "borderline"

# ---- Class weighting
USE_CLASS_WEIGHT = True

# ---- Feature selection
USE_FEATURE_SELECTION = True
FEATURE_SELECT_K = 75

# ---- Patient aggregation
PATIENT_AGG_MODE = "p90"            # "max" | "topk" | "p90" | "noisy_or"
TOPK = 2
PERCENTILE_Q = 90

# ---- Thresholding
THRESHOLD_POLICY = "clinical_cost"  # "balanced" | "f1" | "clinical_cost" | "spec_at_least" | "fixed"
SPEC_TARGET = 0.80
CONTROL_FPR_TARGET = 0.10

# Threshold grid
THRESH_GRID = np.linspace(0.01, 0.99, 199)

# ---- Label-specific feature selection K
FEATURE_SELECT_K_BY_LABEL = {
    "Myoclonus": 150,
    "Athetosis": 150,
    "Stereotypies": 50,
}

# ---- Label-weighted clinical cost (alpha = weight on FNR)
# score = -[(1-alpha)*FPR + alpha*FNR]
CLINICAL_COST_ALPHA_BY_LABEL = {
    "Athetosis": 0.70,      # reduce FN
    "Myoclonus": 0.60,      # reduce FN moderately
    "Stereotypies": 0.40,   # reduce FP
}

# ---- Control constraints (applied on TRAIN controls only)
CONTROL_FP_MAX = 1
CONTROL_FP_MAX_BY_LABEL = {
    "Dystonia": 0,
    "Tremor": 0,
    "Myoclonus": 0,
    "Stereotypies": 0,
    # others default to CONTROL_FP_MAX
}
CONTROL_FPR_TARGET_BY_LABEL = {
    "Dystonia": 0.05,
    "Tremor": 0.02,
    "Myoclonus": 0.02,
    # others fallback to CONTROL_FPR_TARGET
}
SPEC_TARGET_BY_LABEL = {
    "Ballismus": 0.70,
    "Tics": 0.70,
    "Dystonia": 0.90,
    "Tremor": 0.85,
}

# ---- Columns
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

# ---- Scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer()
}

# ---- Panel toggles
ENABLE_SVM = True
ENABLE_OTHER_MODELS = True

# ---- Calibration method for models without predict_proba
CALIB_METHOD = "sigmoid"

# ---- Label-specific tuning toggles (applied to SVM and/or other models)
APPLY_V71B_TO_SVM = True
APPLY_V71B_TO_OTHERS = True

# ---- Nested CV per-label best-model selection (adds extra outputs)
ENABLE_NESTED_LABEL_MODEL_SELECTION = True
NESTED_INNER_SPLITS = 3  # inner CV folds; will auto-reduce if too few subjects

# ---- Nested selection objective
NESTED_SELECTION_OBJECTIVE = "constrained_error"  # "constrained_error" | "auprc"

# ---- Constraints for nested threshold tuning (TRAIN-only, inner CV)
NESTED_ENFORCE_CONTROLS_FP = True
NESTED_ENFORCE_CONTROLS_FPR = True
NESTED_ENFORCE_SPEC_MIN = True
NESTED_ENFORCE_RECALL_MIN = True

# Patient-level recall floors (used in nested constrained thresholding)
RECALL_MIN_BY_LABEL = {
    "Dystonia": 0.70,
    "Tremor": 0.70,
    "Myoclonus": 0.50,
    "Chorea": 0.50,
    "Athetosis": 0.50,
    "Stereotypies": 0.50,
    "Ballismus": 0.33,
    "Tics": 0.33,
}

# Relaxation order if no feasible threshold exists
NESTED_CONSTRAINT_RELAX_LEVELS = [0, 1, 2, 3, 4]


# -------------------------
# Complexity measures
# -------------------------
def higuchi_fd(signal, kmax=5):
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if N < (kmax + 1):
        return 0.0
    Lmk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            n_max = int(np.floor((N - m - 1) / k))
            if n_max <= 1:
                continue
            L = 0.0
            for i in range(1, n_max):
                L += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            L *= (N - 1) / (k * n_max * k)
            Lm.append(L)
        Lmk.append(np.mean(Lm) if len(Lm) else 0.0)
    Lmk = np.asarray(Lmk)
    Lmk[Lmk <= 1e-12] = 1e-12
    return float(-np.polyfit(np.log(np.arange(1, kmax + 1)), np.log(Lmk), 1)[0])


def permutation_entropy(signal, order=3, delay=1):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < (order - 1) * delay + 1:
        return 0.0
    patterns = {}
    for i in range(n - (order - 1) * delay):
        idx = tuple(np.argsort(signal[i:i + order * delay:delay]))
        patterns[idx] = patterns.get(idx, 0) + 1
    probs = np.array(list(patterns.values()), dtype=np.float64)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))


# -------------------------
# Feature engineering
# -------------------------
def extract_features(group, cols, fps=30.0):
    features = {}
    for col in cols:
        signal = group[col].values.astype(float)
        if len(signal) < 2:
            continue

        delta = np.diff(signal, prepend=signal[0])
        accel = np.abs(np.diff(delta, prepend=delta[0]))

        fft_vals = np.abs(fft(signal))
        fft_freqs = np.fft.fftfreq(len(signal), d=1.0 / fps)
        peak_idx = np.argmax(fft_vals[1:]) + 1  # exclude DC

        hist = np.histogram(signal, bins=10, density=True)[0] + 1e-6

        features.update({
            f'{col}_mean': float(np.mean(signal)),
            f'{col}_std': float(np.std(signal)),
            f'{col}_min': float(np.min(signal)),
            f'{col}_max': float(np.max(signal)),
            f'{col}_median': float(np.median(signal)),
            f'{col}_range': float(np.ptp(signal)),
            f'{col}_skew': float(stats.skew(signal)),
            f'{col}_kurtosis': float(stats.kurtosis(signal)),
            f'{col}_energy': float(np.sum(signal**2)),
            f'{col}_slope': float(np.polyfit(np.arange(len(signal)), signal, 1)[0]),
            f'{col}_iqr': float(np.percentile(signal, 75) - np.percentile(signal, 25)),
            f'{col}_entropy': float(stats.entropy(hist)),
            f'{col}_var': float(np.var(signal)),
            f'{col}_fft_peak_freq': float(fft_freqs[peak_idx]),
            f'{col}_fft_peak_amp': float(fft_vals[peak_idx]),
            f'{col}_zero_crossings_delta': int(((delta[:-1] * delta[1:]) < 0).sum()),
            f'{col}_abs_accel_mean': float(np.mean(accel)),
            f'{col}_higuchi_fd': float(higuchi_fd(signal)),
            f'{col}_perm_entropy': float(permutation_entropy(signal)),
        })

        for w in [3, 5, 7]:
            rolling = pd.Series(signal).rolling(window=w, min_periods=1, center=True).mean()
            features[f'{col}_mean_w{w}'] = float(rolling.mean())

    return features


# -------------------------
# Helpers
# -------------------------
def aggregate_patient_probs(groups, window_probs, agg_mode="max", topk=5, q=90):
    unique_ids = np.unique(groups)
    n_labels = window_probs.shape[1]
    patient_probs = np.zeros((len(unique_ids), n_labels), dtype=float)

    for i, p in enumerate(unique_ids):
        idx = np.where(groups == p)[0]
        wp = window_probs[idx]
        if wp.shape[0] == 0:
            continue

        if agg_mode == "max":
            patient_probs[i] = np.max(wp, axis=0)

        elif agg_mode == "topk":
            k = min(topk, wp.shape[0])
            part = np.partition(wp, -k, axis=0)[-k:, :]
            patient_probs[i] = np.mean(part, axis=0)

        elif agg_mode == "p90":
            patient_probs[i] = np.percentile(wp, q, axis=0)

        elif agg_mode == "noisy_or":
            wp_clip = np.clip(wp, 1e-6, 1 - 1e-6)
            patient_probs[i] = 1.0 - np.prod(1.0 - wp_clip, axis=0)

        else:
            raise ValueError(f"Unknown agg_mode={agg_mode}")

    return unique_ids, patient_probs


def safe_macro_auc(y_true, y_score):
    aucs = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, j], y_score[:, j]))
        except Exception:
            continue
    return float(np.mean(aucs)) if len(aucs) else np.nan


def safe_macro_auprc(y_true, y_score):
    aps = []
    for j in range(y_true.shape[1]):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        try:
            aps.append(average_precision_score(y_true[:, j], y_score[:, j]))
        except Exception:
            continue
    return float(np.mean(aps)) if len(aps) else np.nan


def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    denom = tn + fp
    return float(tn / denom) if denom > 0 else np.nan


def balanced_accuracy(y_true, y_pred):
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = specificity_score(y_true, y_pred)
    if spec != spec:
        return np.nan
    return float(0.5 * (rec + spec))


def tune_thresholds_policy(
    y_true_pat,
    y_score_pat,
    labels,
    policy="balanced",
    spec_target=0.80,
    grid=None,
    is_control_pat=None,
    control_fpr_target=None,
    control_fpr_target_by_label=None,
    control_fp_max=None,
    control_fp_max_by_label=None,
    spec_target_by_label=None,
    clinical_cost_alpha_by_label=None,
):
    """
    Returns per-label thresholds tuned on TRAIN patients only.
    Control constraints apply on TRAIN controls only (if present).

    Supported policies:
      - "balanced"      : maximize balanced accuracy
      - "f1"            : maximize F1
      - "clinical_cost" : maximize -[(1-a)*FPR + a*FNR] (a=0.5 -> -0.5*(FPR+FNR))
      - "spec_at_least" : enforce specificity >= target then maximize recall
      - "fixed"         : 0.5
    """
    if policy == "fixed":
        return np.full(y_true_pat.shape[1], 0.5, dtype=float)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    L = y_true_pat.shape[1]
    thr = np.full(L, 0.5, dtype=float)

    has_controls = (is_control_pat is not None) and np.any(is_control_pat)

    control_fpr_target_by_label = control_fpr_target_by_label or {}
    control_fp_max_by_label = control_fp_max_by_label or {}
    spec_target_by_label = spec_target_by_label or {}
    clinical_cost_alpha_by_label = clinical_cost_alpha_by_label or {}

    def confusion_rates(yt, yp):
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        P = tp + fn
        N = tn + fp
        fpr = (fp / N) if N > 0 else np.nan
        fnr = (fn / P) if P > 0 else np.nan
        return (tn, fp, fn, tp, float(fpr) if fpr == fpr else np.nan, float(fnr) if fnr == fnr else np.nan)

    for j in range(L):
        lab = labels[j]
        yt = y_true_pat[:, j].astype(int)
        ys = y_score_pat[:, j].astype(float)

        if len(np.unique(yt)) < 2:
            thr[j] = 0.5
            continue

        this_ctrl_fpr_target = None
        if control_fpr_target is not None:
            this_ctrl_fpr_target = float(control_fpr_target_by_label.get(lab, control_fpr_target))

        this_ctrl_fp_max = None
        if control_fp_max is not None:
            this_ctrl_fp_max = int(control_fp_max_by_label.get(lab, control_fp_max))

        this_spec_target = float(spec_target_by_label.get(lab, spec_target))

        def controls_ok(yp):
            if (not has_controls):
                return True

            yt_c = yt[is_control_pat]
            yp_c = yp[is_control_pat]
            if yt_c.size == 0:
                return True

            tn, fp, fn, tp, fpr, fnr = confusion_rates(yt_c, yp_c)

            if this_ctrl_fp_max is not None:
                if int(fp) > int(this_ctrl_fp_max):
                    return False

            if this_ctrl_fpr_target is not None and fpr == fpr:
                if float(fpr) > float(this_ctrl_fpr_target):
                    return False

            return True

        if policy == "spec_at_least":
            feasible = False
            best_rec = -1.0
            best_t = 0.5
            for t in grid:
                yp = (ys >= t).astype(int)
                if not controls_ok(yp):
                    continue
                specv = specificity_score(yt, yp)
                rec = recall_score(yt, yp, zero_division=0)
                if (specv == specv) and (specv >= this_spec_target):
                    feasible = True
                    if rec > best_rec:
                        best_rec = rec
                        best_t = float(t)
            if feasible:
                thr[j] = best_t
                continue
            effective_policy = "balanced"
        else:
            effective_policy = policy

        best = -1e18
        best_t = 0.5

        for t in grid:
            yp = (ys >= t).astype(int)
            if not controls_ok(yp):
                continue

            if effective_policy == "balanced":
                score = balanced_accuracy(yt, yp)
                if score != score:
                    continue

            elif effective_policy == "f1":
                score = f1_score(yt, yp, zero_division=0)

            elif effective_policy == "clinical_cost":
                tn, fp, fn, tp, fpr, fnr = confusion_rates(yt, yp)
                if (fpr != fpr) or (fnr != fnr):
                    continue
                alpha = float(clinical_cost_alpha_by_label.get(lab, 0.50))
                score = -((1.0 - alpha) * float(fpr) + alpha * float(fnr))

            else:
                raise ValueError(f"Unknown policy={effective_policy}")

            if score > best:
                best = float(score)
                best_t = float(t)

        thr[j] = best_t

    return thr

# -------------------------
# Nested constrained threshold selection (single label)
# -------------------------
def _confusion_counts_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return int(tn), int(fp), int(fn), int(tp)

def _rates_from_counts(tn, fp, fn, tp):
    P = tp + fn
    N = tn + fp
    fpr = (fp / N) if N > 0 else np.nan
    fnr = (fn / P) if P > 0 else np.nan
    rec = (tp / P) if P > 0 else np.nan
    spec = (tn / N) if N > 0 else np.nan
    return float(fpr) if fpr == fpr else np.nan, float(fnr) if fnr == fnr else np.nan, float(rec) if rec == rec else np.nan, float(spec) if spec == spec else np.nan

def aggregate_patient_single_label(groups_windows, y_bin_windows, proba_windows, agg_mode="p90", topk=2, q=90):
    """
    Aggregate window-level proba to patient-level proba for a single label, and compute patient-level y (OR).
    Returns: patients, y_pat, proba_pat, is_control_pat
    """
    proba_windows = np.asarray(proba_windows, dtype=float).reshape(-1, 1)
    patients, proba_pat_2d = aggregate_patient_probs(groups_windows, proba_windows, agg_mode=agg_mode, topk=topk, q=q)
    proba_pat = proba_pat_2d[:, 0].astype(float)

    y_bin_windows = np.asarray(y_bin_windows, dtype=int)
    y_pat = np.array([int(y_bin_windows[groups_windows == p].max()) for p in patients], dtype=int)
    is_control_pat = np.array([str(p).startswith("C") for p in patients], dtype=bool)
    return patients, y_pat, proba_pat, is_control_pat

def select_threshold_min_error_under_constraints(
    y_pat,
    proba_pat,
    label,
    grid,
    is_control_pat=None,
    relax_level=0,
):
    """
    Choose threshold t that minimizes patient-level error rate (FP+FN)/N on the provided patient set,
    under constraints (depending on relax_level). Constraints are evaluated on this set only.

    relax_level:
      0: enforce controls FP + controls FPR + spec_min + recall_min
      1: drop spec_min
      2: drop controls FPR
      3: drop recall_min
      4: drop all constraints
    """
    y_pat = np.asarray(y_pat, dtype=int)
    proba_pat = np.asarray(proba_pat, dtype=float)

    if grid is None:
        grid = np.linspace(0.01, 0.99, 199)

    if len(np.unique(y_pat)) < 2:
        return 0.5, {"feasible": True, "relax_level": relax_level, "n_feasible": 0}

    this_fp_max = int(CONTROL_FP_MAX_BY_LABEL.get(label, CONTROL_FP_MAX))
    this_fpr_target = float(CONTROL_FPR_TARGET_BY_LABEL.get(label, CONTROL_FPR_TARGET))
    this_spec_min = float(SPEC_TARGET_BY_LABEL.get(label, SPEC_TARGET))
    this_recall_min = float(RECALL_MIN_BY_LABEL.get(label, 0.0))

    has_controls = (is_control_pat is not None) and np.any(is_control_pat)

    def _constraints_ok(y_true, y_pred):
        tn, fp, fn, tp = _confusion_counts_binary(y_true, y_pred)
        fpr, fnr, rec, spec = _rates_from_counts(tn, fp, fn, tp)

        if has_controls:
            yc = y_true[is_control_pat]
            pc = y_pred[is_control_pat]
            tn_c, fp_c, fn_c, tp_c = _confusion_counts_binary(yc, pc)
            fpr_c, _, _, _ = _rates_from_counts(tn_c, fp_c, fn_c, tp_c)

            if relax_level <= 3:
                if NESTED_ENFORCE_CONTROLS_FP and relax_level < 4:
                    if int(fp_c) > int(this_fp_max):
                        return False

            if relax_level <= 1:
                if NESTED_ENFORCE_CONTROLS_FPR and relax_level < 2:
                    if (fpr_c == fpr_c) and (float(fpr_c) > float(this_fpr_target)):
                        return False

        if NESTED_ENFORCE_SPEC_MIN and relax_level < 1 and relax_level < 4:
            if (spec == spec) and (float(spec) < float(this_spec_min)):
                return False

        if NESTED_ENFORCE_RECALL_MIN and relax_level < 3 and relax_level < 4:
            if int((y_true == 1).sum()) > 0:
                if (rec == rec) and (float(rec) < float(this_recall_min)):
                    return False

        return True

    best_t = 0.5
    best_err = +1e18
    n_feasible = 0

    for t in grid:
        y_pred = (proba_pat >= float(t)).astype(int)

        if relax_level < 4:
            if not _constraints_ok(y_pat, y_pred):
                continue

        tn, fp, fn, tp = _confusion_counts_binary(y_pat, y_pred)
        err_rate = (fp + fn) / max(1, (tp + tn + fp + fn))

        n_feasible += 1
        if err_rate < best_err:
            best_err = float(err_rate)
            best_t = float(t)

    feasible = (n_feasible > 0)
    return best_t, {"feasible": feasible, "relax_level": relax_level, "n_feasible": int(n_feasible), "best_err": float(best_err) if feasible else np.nan}

def adjusted_smote_k(y_train_bin, requested_k=5):
    pos = int(np.sum(y_train_bin == 1))
    neg = int(np.sum(y_train_bin == 0))
    if pos == 0 or neg == 0:
        return None
    m = min(pos, neg)
    if m < 2:
        return None
    k = min(requested_k, m - 1)
    return int(k) if k >= 1 else None


def smote_sampling_strategy_safe_positives_only(y_train_bin, target_ratio=0.25):
    yb = np.asarray(y_train_bin).astype(int)
    n_pos = int((yb == 1).sum())
    n_neg = int((yb == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    if n_pos >= n_neg:
        return None
    current_ratio = n_pos / max(n_neg, 1)
    if current_ratio >= target_ratio:
        return None
    return float(target_ratio)


def configure_estimator_for_label_general(estimator, y_train_bin):
    if not USE_CLASS_WEIGHT:
        return estimator
    n_pos = int((y_train_bin == 1).sum())
    n_neg = int((y_train_bin == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return estimator
    try:
        if "class_weight" in estimator.get_params():
            estimator.set_params(class_weight="balanced")
    except Exception:
        pass
    return estimator


def make_calibrated(base_estimator, method="sigmoid", cv=3):
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)


# -------------------------
# Safe calibration helpers
# -------------------------
def safe_calibrate_or_raw_linear_svc(base_linear_svc, y_train_bin, method="sigmoid", max_cv=3):
    """
    Return either:
      - CalibratedClassifierCV(LinearSVC, cv=cv_used) when feasible, or
      - the raw LinearSVC when calibration is not feasible (too few samples in a class).
    """
    yb = np.asarray(y_train_bin).astype(int)
    n_pos = int((yb == 1).sum())
    n_neg = int((yb == 0).sum())
    m = int(min(n_pos, n_neg))

    if m < 2:
        return base_linear_svc

    cv_used = int(min(max_cv, m))
    cv_used = int(max(cv_used, 2))

    try:
        return CalibratedClassifierCV(estimator=base_linear_svc, method=method, cv=cv_used)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_linear_svc, method=method, cv=cv_used)


def decision_function_to_proba(estimator_or_pipeline, X_eval):
    """
    Converts decision_function output into [0,1] via sigmoid.
    Fallback when predict_proba is not available or calibration is not feasible.
    """
    df = estimator_or_pipeline.decision_function(X_eval)
    df = np.asarray(df).reshape(-1)
    return expit(df).astype(float)


def build_pipeline_generic(scaler_or_none, estimator, y_train_bin, label=None, n_features=None, enable_smote=False, apply_v71b=False):
    """
    Unified pipeline builder:
      - supports Identity scaler
      - supports label-specific K (feature selection) when apply_v71b=True
    """
    steps = []
    if scaler_or_none is None:
        steps.append(("scaler", FunctionTransformer(lambda x: x, validate=False)))
    else:
        steps.append(("scaler", scaler_or_none))

    if USE_FEATURE_SELECTION and (FEATURE_SELECT_K is not None) and (FEATURE_SELECT_K > 0):
        k_default = int(FEATURE_SELECT_K)
        if apply_v71b and (label is not None):
            k_use = int(FEATURE_SELECT_K_BY_LABEL.get(label, k_default))
        else:
            k_use = k_default
        if n_features is not None:
            k_use = int(min(int(k_use), int(n_features)))
        steps.append(("select", SelectKBest(mutual_info_classif, k=int(k_use))))

    if not enable_smote:
        steps.append(("clf", estimator))
        return ImbPipeline(steps=steps), False, None, None

    k = adjusted_smote_k(y_train_bin, requested_k=SMOTE_K_NEIGHBORS)
    strategy = smote_sampling_strategy_safe_positives_only(y_train_bin, target_ratio=SMOTE_RATIO)
    smote_cls = SMOTE if SMOTE_VARIANT == "regular" else BorderlineSMOTE

    if k is None or strategy is None:
        steps.append(("clf", estimator))
        return ImbPipeline(steps=steps), False, None, None

    steps.append(("smote", smote_cls(
        random_state=random_state,
        k_neighbors=k,
        sampling_strategy=strategy,
    )))
    steps.append(("clf", estimator))
    return ImbPipeline(steps=steps), True, k, strategy


def build_error_decomposition(model_name, patient_store_dict, labels):
    rows = []
    yt_list = patient_store_dict.get(model_name, {}).get("y_true", [])
    yp_list = patient_store_dict.get(model_name, {}).get("y_pred", [])
    if not (len(yt_list) and len(yp_list)):
        return pd.DataFrame()

    YT = np.vstack(yt_list)
    YP = np.vstack(yp_list)

    for j, lab in enumerate(labels):
        yt = YT[:, j].astype(int)
        yp = YP[:, j].astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        P = int((yt == 1).sum())
        N = int((yt == 0).sum())
        fpr = float(fp / N) if N > 0 else np.nan
        fnr = float(fn / P) if P > 0 else np.nan

        rows.append(dict(
            model=model_name,
            label=lab,
            P=P,
            N=N,
            prevalence=float(P / (P + N)) if (P + N) > 0 else np.nan,
            TP=int(tp),
            FN=int(fn),
            FP=int(fp),
            TN=int(tn),
            FNR=float(fnr) if fnr == fnr else np.nan,
            FPR=float(fpr) if fpr == fpr else np.nan,
            errors=int(fp + fn),
            errors_per_patient=float((fp + fn) / (P + N)) if (P + N) > 0 else np.nan,
        ))
    return pd.DataFrame(rows)


def clinical_cost_from_confusions(y_true, y_pred):
    """
    Macro-label clinical cost: mean over labels of 0.5*(FPR + FNR)
    """
    costs = []
    L = y_true.shape[1]
    for j in range(L):
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        P = (tp + fn)
        N = (tn + fp)
        fpr = (fp / N) if N > 0 else np.nan
        fnr = (fn / P) if P > 0 else np.nan
        if (fpr == fpr) and (fnr == fnr):
            costs.append(0.5 * (float(fpr) + float(fnr)))
    return float(np.mean(costs)) if len(costs) else np.nan


def clinical_cost_weighted_from_confusions(y_true, y_pred, labels, alpha_by_label):
    """
    Macro-label weighted clinical cost: mean over labels of [(1-a)*FPR + a*FNR]
    """
    costs = []
    L = y_true.shape[1]
    for j in range(L):
        lab = labels[j]
        a = float(alpha_by_label.get(lab, 0.50))
        yt = y_true[:, j].astype(int)
        yp = y_pred[:, j].astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        P = (tp + fn)
        N = (tn + fp)
        fpr = (fp / N) if N > 0 else np.nan
        fnr = (fn / P) if P > 0 else np.nan
        if (fpr == fpr) and (fnr == fnr):
            costs.append((1.0 - a) * float(fpr) + a * float(fnr))
    return float(np.mean(costs)) if len(costs) else np.nan


# -------------------------
# Data extraction
# -------------------------
data, groups, multilabels = [], [], []
print("Reading files from:", data_folder)

files = sorted([f for f in os.listdir(data_folder) if f.endswith(".xlsx")])
assert len(files) > 0, f"No .xlsx files found in {data_folder}"

n_files_used = 0
for file in files:
    if (not INCLUDE_CONTROLS) and file.startswith("C"):
        continue

    path = os.path.join(data_folder, file)
    df = pd.read_excel(path, engine="openpyxl")

    required = set(symptoms) | {"From", "To"} | set(DISTANCE_COLS)
    if not required.issubset(df.columns):
        print(f"  ⚠️ Skipping {file}: missing required columns")
        continue

    df = df.copy()
    df.replace(',', '.', regex=True, inplace=True)
    df[DISTANCE_COLS] = df[DISTANCE_COLS].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=DISTANCE_COLS, inplace=True)

    group_name = file.replace(".xlsx", "")
    n_files_used += 1

    for (f, t), grp in df.groupby(["From", "To"]):
        if not grp[DISTANCE_COLS].notna().all(axis=1).all():
            continue

        label_vector = [int(grp[s].eq(1).any()) for s in symptoms]
        if (not INCLUDE_ALL_ZERO_WINDOWS) and (sum(label_vector) == 0):
            continue

        feat = extract_features(grp, DISTANCE_COLS, fps=FPS)
        feat["group"] = group_name
        data.append(feat)
        multilabels.append(label_vector)
        groups.append(group_name)

print(f"✅ Used {n_files_used} files | Extracted {len(data)} windows | Unique subjects={len(set(groups))}")
assert len(data) > 0, "No windows extracted. Check parsing / From-To grouping / column names."

X_df = pd.DataFrame(data).fillna(0)
group_col = X_df.pop("group")
X = X_df.values.astype(float)
y = np.asarray(multilabels, dtype=int)
groups = np.asarray(group_col)

assert X.shape[0] == y.shape[0] == groups.shape[0]
assert y.shape[1] == len(symptoms)

# -------------------------
# Patient-level stratification labels (OR across windows)
# -------------------------
unique_subjects = np.unique(groups)

subject_Y = []
subject_to_window_idx = {}
for p in unique_subjects:
    idx = np.where(groups == p)[0]
    subject_to_window_idx[p] = idx
    subject_Y.append((y[idx].max(axis=0)).astype(int))
subject_Y = np.asarray(subject_Y, dtype=int)

print("\n[DIAG] Patient-level label prevalence:")
for j, s in enumerate(symptoms):
    prev = float(subject_Y[:, j].mean())
    pos = int(subject_Y[:, j].sum())
    print(f"  - {s:12s}: {pos:2d}/{len(unique_subjects)} ({prev*100:5.1f}%)")

# Sanity checks
assert len(unique_subjects) == subject_Y.shape[0]
assert X.shape[1] > 0, "No features found."
if USE_FEATURE_SELECTION and (FEATURE_SELECT_K is not None) and (FEATURE_SELECT_K > 0):
    assert int(FEATURE_SELECT_K) <= X.shape[1], f"FEATURE_SELECT_K={FEATURE_SELECT_K} > n_features={X.shape[1]}"
for k in FEATURE_SELECT_K_BY_LABEL.values():
    assert int(k) >= 1, "FEATURE_SELECT_K_BY_LABEL must be >= 1."
for a in CLINICAL_COST_ALPHA_BY_LABEL.values():
    assert 0.0 <= float(a) <= 1.0, "Clinical cost alpha must be within [0,1]."

mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# -------------------------
# Model specs (non-SVM panel)
# -------------------------
def other_model_specs():
    """
    Exactly 1 configuration per model family (scaler included).
    """
    specs = []

    specs.append(dict(
        family="LogReg",
        model="LogReg(p=l2,C=0.3)",
        requires_scaler=True,
        scaler_name="MinMaxScaler",
        build=lambda: LogisticRegression(
            C=0.3, penalty="l2", solver="liblinear", max_iter=8000,
            class_weight="balanced", random_state=random_state,
        ),
    ))

    specs.append(dict(
        family="SGDLog",
        model="SGDLog(p=l2,alpha=1e-4)",
        requires_scaler=True,
        scaler_name="StandardScaler",
        build=lambda: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4,
            max_iter=8000, tol=1e-4, class_weight="balanced",
            random_state=random_state,
        ),
    ))

    specs.append(dict(
        family="LinearSVC",
        model="LinearSVC+SafeCalib(C=3.0,cw=bal)",
        requires_scaler=True,
        scaler_name="PowerTransformer",
        needs_safe_calibration=True,
        build=lambda: LinearSVC(C=3.0, class_weight="balanced", random_state=random_state),
    ))

    specs.append(dict(
        family="KNN",
        model="KNN(k=9,w=uniform,p=2)",
        requires_scaler=True,
        scaler_name="MinMaxScaler",
        build=lambda: KNeighborsClassifier(
            n_neighbors=9, weights="uniform", metric="minkowski", p=2,
        ),
    ))

    specs.append(dict(
        family="MLP",
        model="MLP(h=(64,),a=0.001,lr0=0.0003,early_stop=0)",
        requires_scaler=True,
        scaler_name="StandardScaler",
        build=lambda: MLPClassifier(
            hidden_layer_sizes=(64,), alpha=1e-3, learning_rate_init=3e-4,
            max_iter=4000,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    ))


    specs.append(dict(
        family="RF",
        model="RF(n=600,depth=10,leaf=2,mf=sqrt)",
        requires_scaler=False,
        build=lambda: RandomForestClassifier(
            n_estimators=600, max_depth=10, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=random_state, n_jobs=1,
        ),
    ))

    specs.append(dict(
        family="ET",
        model="ET(n=800,depth=10,leaf=2,mf=sqrt)",
        requires_scaler=False,
        build=lambda: ExtraTreesClassifier(
            n_estimators=800, max_depth=10, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=random_state, n_jobs=1,
        ),
    ))

    specs.append(dict(
        family="Ada",
        model="AdaBoost(n=300,lr=0.5)",
        requires_scaler=False,
        build=lambda: AdaBoostClassifier(
            n_estimators=300, learning_rate=0.5, random_state=random_state,
        ),
    ))

    specs.append(dict(
        family="GB",
        model="GB(n=400,lr=0.05,sub=0.9)",
        requires_scaler=False,
        build=lambda: GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, subsample=0.9, random_state=random_state,
        ),
    ))

    specs.append(dict(
        family="HGB",
        model="HGB(it=300,lr=0.05,d=3,ln=31,ml=20,l2=0.1)",
        requires_scaler=False,
        build=lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1,
            random_state=random_state,
        ),
    ))

    try:
        from xgboost import XGBClassifier  # type: ignore
        specs.append(dict(
            family="XGB",
            model="XGB(n=400,d=3,lr=0.05,sub=0.9,col=0.9,mcw=5)",
            requires_scaler=False,
            build=lambda: XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, min_child_weight=5,
                reg_lambda=5.0, random_state=random_state, n_jobs=1, eval_metric="logloss",
            ),
        ))
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore
        specs.append(dict(
            family="LGBM",
            model="LGBM(n=600,leaves=31,lr=0.05,mcs=20)",
            requires_scaler=False,
            build=lambda: LGBMClassifier(
                n_estimators=600, num_leaves=31, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
                reg_lambda=1.0, random_state=random_state, n_jobs=1, verbosity=-1,
            ),
        ))
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier  # type: ignore
        specs.append(dict(
            family="CAT",
            model="CAT(it=800,d=6,lr=0.05,l2=3.0)",
            requires_scaler=False,
            build=lambda: CatBoostClassifier(
                iterations=800, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                loss_function="Logloss", random_seed=random_state, verbose=False,
            ),
        ))
    except Exception:
        pass

    return specs


# -------------------------
# Run meta (saved to Excel)
# -------------------------
run_meta = {
    "version": "V7.3 (V7.1b fixed+applied, p90 agg, dense threshold grid, nested label selection added)",
    "data_folder": data_folder,
    "INCLUDE_CONTROLS": INCLUDE_CONTROLS,
    "INCLUDE_ALL_ZERO_WINDOWS": INCLUDE_ALL_ZERO_WINDOWS,
    "PATIENT_AGG_MODE": PATIENT_AGG_MODE,
    "TOPK": TOPK,
    "PERCENTILE_Q": PERCENTILE_Q,
    "THRESHOLD_POLICY": THRESHOLD_POLICY,
    "SPEC_TARGET": SPEC_TARGET,
    "CONTROL_FPR_TARGET": CONTROL_FPR_TARGET,
    "CONTROL_FPR_TARGET_BY_LABEL": json.dumps(CONTROL_FPR_TARGET_BY_LABEL),
    "CONTROL_FP_MAX": CONTROL_FP_MAX,
    "CONTROL_FP_MAX_BY_LABEL": json.dumps(CONTROL_FP_MAX_BY_LABEL),
    "SPEC_TARGET_BY_LABEL": json.dumps(SPEC_TARGET_BY_LABEL),
    "SVM_GRID": json.dumps(SVM_GRID),
    "USE_SMOTE_FOR_SVM": USE_SMOTE_FOR_SVM,
    "USE_SMOTE_FOR_OTHERS": USE_SMOTE_FOR_OTHERS,
    "SMOTE_K_NEIGHBORS": SMOTE_K_NEIGHBORS,
    "SMOTE_RATIO": SMOTE_RATIO,
    "SMOTE_VARIANT": SMOTE_VARIANT,
    "USE_CLASS_WEIGHT": USE_CLASS_WEIGHT,
    "USE_FEATURE_SELECTION": USE_FEATURE_SELECTION,
    "FEATURE_SELECT_K": FEATURE_SELECT_K,
    "FEATURE_SELECT_K_BY_LABEL": json.dumps(FEATURE_SELECT_K_BY_LABEL),
    "CLINICAL_COST_ALPHA_BY_LABEL": json.dumps(CLINICAL_COST_ALPHA_BY_LABEL),
    "THRESH_GRID": json.dumps(list(map(float, THRESH_GRID))),
    "n_splits": n_splits,
    "FPS": FPS,
    "n_windows": int(X.shape[0]),
    "n_subjects": int(len(unique_subjects)),
    "n_features": int(X.shape[1]),
    "symptoms": symptoms,
    "ENABLE_SVM": ENABLE_SVM,
    "ENABLE_OTHER_MODELS": ENABLE_OTHER_MODELS,
    "CALIB_METHOD": CALIB_METHOD,
    "APPLY_V71B_TO_SVM": APPLY_V71B_TO_SVM,
    "APPLY_V71B_TO_OTHERS": APPLY_V71B_TO_OTHERS,
    "ENABLE_NESTED_LABEL_MODEL_SELECTION": ENABLE_NESTED_LABEL_MODEL_SELECTION,
    "NESTED_INNER_SPLITS": NESTED_INNER_SPLITS,
}

# ============================================================
# FEATURE IMPORTANCE (patient-level permutation importance)
#   - computed on OUTER TEST folds only (no leakage)
#   - delta in patient-level error rate (FP+FN)/N for each label
# ============================================================

ENABLE_FEATURE_IMPORTANCE = True
feature_importance_long_rows = []

FEATURE_IMPORTANCE_MAX_FEATURES = 50
FEATURE_IMPORTANCE_N_REPEATS = 5
FEATURE_IMPORTANCE_RANDOM_STATE = 42

def patient_level_or_labels(y_win_bin, g_win):
    """Compute patient-level label vector (OR across windows) for a single binary label."""
    pats = np.unique(g_win)
    y_pat = np.array([int(y_win_bin[g_win == p].max()) for p in pats], dtype=int)
    return pats, y_pat

def patient_level_predict_from_window_proba(g_win, proba_win, agg_mode="p90", topk=2, q=90):
    """Aggregate window probabilities to patient probabilities for one label."""
    proba_win = np.asarray(proba_win, dtype=float).reshape(-1, 1)
    pats, proba_pat_2d = aggregate_patient_probs(g_win, proba_win, agg_mode=agg_mode, topk=topk, q=q)
    return pats, proba_pat_2d[:, 0].astype(float)

def patient_error_rate(y_pat, y_pred_pat):
    """Error rate = (FP + FN)/N"""
    tn, fp, fn, tp = _confusion_counts_binary(np.asarray(y_pat, int), np.asarray(y_pred_pat, int))
    denom = max(1, tn + fp + fn + tp)
    return float((fp + fn) / denom)

def _get_selected_feature_indices(pipe):
    """
    Return indices (w.r.t original X columns) of selected features if SelectKBest exists,
    else return all features.
    """
    if hasattr(pipe, "named_steps") and ("select" in pipe.named_steps):
        sel = pipe.named_steps["select"]
        if hasattr(sel, "get_support"):
            mask = sel.get_support()
            return np.where(mask)[0].astype(int).tolist()
    return None

def permutation_importance_patient_level_error(
    pipe,
    X_test,
    y_test_bin,
    g_test,
    threshold,
    feature_names,
    agg_mode="p90",
    topk=2,
    q=90,
    n_repeats=5,
    max_features=50,
    rng_seed=42,
):
    """
    Permutation importance as delta(error_rate) at patient-level for one label.

    Baseline:
      proba_win = pipe.predict_proba(X_test)[:,1] (or fallback)
      proba_pat = aggregate(p90)
      pred_pat  = proba_pat >= threshold
      base_err  = (FP+FN)/N

    Importance(feature f) = mean over repeats of (err_perm - base_err).
    """
    rng = np.random.RandomState(rng_seed)

    if hasattr(pipe, "predict_proba"):
        p0_win = np.asarray(pipe.predict_proba(X_test))[:, 1]
    elif hasattr(pipe, "decision_function"):
        p0_win = decision_function_to_proba(pipe, X_test)
    else:
        p0_win = pipe.predict(X_test).astype(float)

    pats, y_pat = patient_level_or_labels(y_test_bin, g_test)
    pats2, p0_pat = patient_level_predict_from_window_proba(g_test, p0_win, agg_mode=agg_mode, topk=topk, q=q)
    assert np.all(pats == pats2), "Patient order mismatch in aggregation."
    y0_pred = (p0_pat >= float(threshold)).astype(int)
    base_err = patient_error_rate(y_pat, y0_pred)

    selected_idx = _get_selected_feature_indices(pipe)
    if selected_idx is None:
        perm_idx = list(range(X_test.shape[1]))
    else:
        perm_idx = selected_idx[:]

    if (max_features is not None) and (len(perm_idx) > int(max_features)):
        perm_idx = perm_idx[: int(max_features)]

    rows = []

    for fi in perm_idx:
        deltas = []
        for r in range(int(n_repeats)):
            X_tmp = X_test.copy()
            perm = rng.permutation(X_tmp.shape[0])
            X_tmp[:, fi] = X_tmp[perm, fi]
            if hasattr(pipe, "predict_proba"):
                p_win = np.asarray(pipe.predict_proba(X_tmp))[:, 1]
            elif hasattr(pipe, "decision_function"):
                p_win = decision_function_to_proba(pipe, X_tmp)
            else:
                p_win = pipe.predict(X_tmp).astype(float)

            _, p_pat = patient_level_predict_from_window_proba(g_test, p_win, agg_mode=agg_mode, topk=topk, q=q)
            y_pred = (p_pat >= float(threshold)).astype(int)
            err = patient_error_rate(y_pat, y_pred)
            deltas.append(err - base_err)

        rows.append(dict(
            feature=str(feature_names[fi]),
            feature_index=int(fi),
            importance_delta_error_mean=float(np.mean(deltas)),
            importance_delta_error_std=float(np.std(deltas)),
            baseline_error=float(base_err),
            n_repeats=int(n_repeats),
        ))

    return pd.DataFrame(rows).sort_values("importance_delta_error_mean", ascending=False).reset_index(drop=True)

# -------------------------
# Training + evaluation loop
# -------------------------
all_rows = []
per_label_rows = []
patient_store = {}


# -------------------------
# SVM branch
# -------------------------
if ENABLE_SVM:
    for scaler_name, scaler in scalers.items():
        for svm_hp in SVM_GRID:
            model_name = f"SVM(C={svm_hp['C']},gamma={svm_hp['gamma']})"
            store_key = f"{scaler_name}__{model_name}"
            patient_store.setdefault(store_key, {"y_true": [], "y_pred": []})

            print(f"\n==============================")
            print(f"Scaler={scaler_name} | Model={model_name}")
            print(f"==============================")

            for fold, (p_tr, p_te) in enumerate(mskf.split(unique_subjects, subject_Y), start=1):
                train_subjects = unique_subjects[p_tr]
                test_subjects = unique_subjects[p_te]
                assert len(set(train_subjects).intersection(set(test_subjects))) == 0

                train_idx = np.concatenate([subject_to_window_idx[p] for p in train_subjects])
                test_idx = np.concatenate([subject_to_window_idx[p] for p in test_subjects])

                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                g_train = groups[train_idx]
                g_test = groups[test_idx]

                n_labels = y.shape[1]
                window_proba_test = np.zeros((len(test_idx), n_labels), dtype=float)
                window_proba_train = np.zeros((len(train_idx), n_labels), dtype=float)

                smote_used = {}
                smote_k_used = {}
                smote_strategy_used = {}

                fitted_pipes_by_label = {}

                for j, lab in enumerate(symptoms):
                    y_train_bin = y_train[:, j].astype(int)

                    if len(np.unique(y_train_bin)) < 2:
                        const_p = float(np.mean(y_train_bin))
                        window_proba_test[:, j] = const_p
                        window_proba_train[:, j] = const_p
                        smote_used[lab] = False
                        smote_k_used[lab] = None
                        smote_strategy_used[lab] = None
                        fitted_pipes_by_label[lab] = None
                        continue

                    est = SVC(
                        C=float(svm_hp["C"]),
                        kernel="rbf",
                        gamma=svm_hp["gamma"],
                        probability=True,
                        random_state=random_state,
                    )
                    est = configure_estimator_for_label_general(est, y_train_bin)

                    pipe, used, k_used, strat_used = build_pipeline_generic(
                        scaler, est, y_train_bin,
                        label=lab, n_features=X_train.shape[1],
                        enable_smote=bool(USE_SMOTE_FOR_SVM),
                        apply_v71b=bool(APPLY_V71B_TO_SVM),
                    )

                    smote_used[lab] = bool(used)
                    smote_k_used[lab] = k_used
                    smote_strategy_used[lab] = strat_used

                    pipe.fit(X_train, y_train_bin)
                    window_proba_test[:, j] = pipe.predict_proba(X_test)[:, 1]
                    window_proba_train[:, j] = pipe.predict_proba(X_train)[:, 1]
                    fitted_pipes_by_label[lab] = pipe

                subj_train, subj_proba_train = aggregate_patient_probs(
                    g_train, window_proba_train, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
                )
                subj_test, subj_proba_test = aggregate_patient_probs(
                    g_test, window_proba_test, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
                )

                subj_y_train = np.vstack([(y_train[g_train == p].max(axis=0)).astype(int) for p in subj_train])
                subj_y_test = np.vstack([(y_test[g_test == p].max(axis=0)).astype(int) for p in subj_test])

                is_control_train = np.array([str(p).startswith("C") for p in subj_train], dtype=bool)

                alpha_dict = CLINICAL_COST_ALPHA_BY_LABEL if APPLY_V71B_TO_SVM else {}
                thresholds = tune_thresholds_policy(
                    subj_y_train,
                    subj_proba_train,
                    labels=symptoms,
                    policy=THRESHOLD_POLICY,
                    spec_target=SPEC_TARGET,
                    grid=THRESH_GRID,
                    is_control_pat=is_control_train,
                    control_fpr_target=CONTROL_FPR_TARGET,
                    control_fpr_target_by_label=CONTROL_FPR_TARGET_BY_LABEL,
                    control_fp_max=CONTROL_FP_MAX,
                    control_fp_max_by_label=CONTROL_FP_MAX_BY_LABEL,
                    spec_target_by_label=SPEC_TARGET_BY_LABEL,
                    clinical_cost_alpha_by_label=alpha_dict,
                )

                subj_pred_test = (subj_proba_test >= thresholds[None, :]).astype(int)

                if ENABLE_FEATURE_IMPORTANCE:
                    feature_names = list(X_df.columns)

                    for j, lab in enumerate(symptoms):
                        pipe_fit = fitted_pipes_by_label.get(lab, None)
                        if pipe_fit is None:
                            continue

                        thr_lab = float(thresholds[j])
                        y_te_bin = y_test[:, j].astype(int)

                        imp_df = permutation_importance_patient_level_error(
                            pipe=pipe_fit,
                            X_test=X_test,
                            y_test_bin=y_te_bin,
                            g_test=g_test,
                            threshold=thr_lab,
                            feature_names=feature_names,
                            agg_mode=PATIENT_AGG_MODE,
                            topk=TOPK,
                            q=PERCENTILE_Q,
                            n_repeats=FEATURE_IMPORTANCE_N_REPEATS,
                            max_features=FEATURE_IMPORTANCE_MAX_FEATURES,
                            rng_seed=FEATURE_IMPORTANCE_RANDOM_STATE + 1000*fold + 13*j,
                        )

                        imp_df["fold"] = int(fold)
                        imp_df["label"] = str(lab)
                        imp_df["scaler"] = str(scaler_name)
                        imp_df["model"] = str(model_name)
                        imp_df["model_key"] = str(store_key)
                        feature_importance_long_rows.append(imp_df)

                patient_store[store_key]["y_true"].append(subj_y_test)
                patient_store[store_key]["y_pred"].append(subj_pred_test)

                is_control_test = np.array([str(p).startswith("C") for p in subj_test], dtype=bool)
                controls_fpr = {}
                controls_fp = {}
                if np.any(is_control_test):
                    for j, lab in enumerate(symptoms):
                        yt_c = subj_y_test[is_control_test, j]
                        yp_c = subj_pred_test[is_control_test, j]
                        cm = confusion_matrix(yt_c, yp_c, labels=[0, 1])
                        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                        denom = fp + tn
                        controls_fpr[lab] = float(fp / denom) if denom > 0 else np.nan
                        controls_fp[lab] = int(fp)
                else:
                    controls_fpr = {lab: np.nan for lab in symptoms}
                    controls_fp = {lab: np.nan for lab in symptoms}

                macro_auc = safe_macro_auc(subj_y_test, subj_proba_test)
                macro_auprc = safe_macro_auprc(subj_y_test, subj_proba_test)
                micro_f1 = f1_score(subj_y_test, subj_pred_test, average="micro", zero_division=0)
                macro_f1 = f1_score(subj_y_test, subj_pred_test, average="macro", zero_division=0)
                jac = jaccard_score(subj_y_test, subj_pred_test, average="samples", zero_division=0)
                ham_loss = hamming_loss(subj_y_test, subj_pred_test)
                ham_acc = 1.0 - float(ham_loss)
                exact_match = accuracy_score(subj_y_test, subj_pred_test)

                clinical_cost = clinical_cost_from_confusions(subj_y_test, subj_pred_test)
                clinical_cost_w = clinical_cost_weighted_from_confusions(subj_y_test, subj_pred_test, symptoms, CLINICAL_COST_ALPHA_BY_LABEL)

                for j, lab in enumerate(symptoms):
                    yt = subj_y_test[:, j]
                    yp = subj_pred_test[:, j]
                    ys = subj_proba_test[:, j]

                    prec = precision_score(yt, yp, zero_division=0)
                    rec = recall_score(yt, yp, zero_division=0)
                    f1v = f1_score(yt, yp, zero_division=0)
                    specv = specificity_score(yt, yp)

                    auc = np.nan
                    auprc = np.nan
                    if len(np.unique(yt)) >= 2:
                        try:
                            auc = roc_auc_score(yt, ys)
                        except Exception:
                            auc = np.nan
                        try:
                            auprc = average_precision_score(yt, ys)
                        except Exception:
                            auprc = np.nan

                    per_label_rows.append(dict(
                        scaler=scaler_name,
                        model=model_name,
                        fold=int(fold),
                        label=lab,
                        precision=float(prec),
                        recall=float(rec),
                        specificity=float(specv) if specv == specv else np.nan,
                        f1=float(f1v),
                        auc=float(auc) if auc == auc else np.nan,
                        auprc=float(auprc) if auprc == auprc else np.nan,
                        threshold=float(thresholds[j]),
                        smote_used=bool(smote_used.get(lab, False)),
                        smote_k=smote_k_used.get(lab, None),
                        smote_strategy=smote_strategy_used.get(lab, None),
                        support_pos=int(np.sum(yt == 1)),
                        support_neg=int(np.sum(yt == 0)),
                    ))

                print(
                    f"  ▶ Fold {fold}/{n_splits}: "
                    f"macroAUC={macro_auc:.3f} | macroAUPRC={macro_auprc:.3f} | "
                    f"microF1={micro_f1:.3f} | macroF1={macro_f1:.3f} | "
                    f"Jaccard={jac:.3f} | HammingAcc={ham_acc:.3f} | ExactMatch={exact_match:.3f} | "
                    f"ClinicalCost={clinical_cost:.3f} | ClinicalCostW={clinical_cost_w:.3f}"
                )

                all_rows.append(dict(
                    scaler=scaler_name,
                    model=model_name,
                    fold=int(fold),
                    n_train_subjects=int(len(train_subjects)),
                    n_test_subjects=int(len(test_subjects)),
                    n_train_windows=int(len(train_idx)),
                    n_test_windows=int(len(test_idx)),
                    patient_macro_auc=float(macro_auc) if macro_auc == macro_auc else np.nan,
                    patient_macro_auprc=float(macro_auprc) if macro_auprc == macro_auprc else np.nan,
                    patient_micro_f1=float(micro_f1),
                    patient_macro_f1=float(macro_f1),
                    patient_jaccard_samples=float(jac),
                    patient_hamming_loss=float(ham_loss),
                    patient_hamming_acc=float(ham_acc),
                    patient_exact_match=float(exact_match),
                    patient_clinical_cost=float(clinical_cost) if clinical_cost == clinical_cost else np.nan,
                    patient_clinical_cost_weighted=float(clinical_cost_w) if clinical_cost_w == clinical_cost_w else np.nan,
                    thresholds=json.dumps({symptoms[j]: float(thresholds[j]) for j in range(len(symptoms))}),
                    smote_used=json.dumps(smote_used),
                    smote_k_used=json.dumps(smote_k_used),
                    smote_strategy_used=json.dumps(smote_strategy_used),
                    controls_only_fpr=json.dumps(controls_fpr),
                    controls_only_fp=json.dumps(controls_fp),
                    n_controls_test=int(is_control_test.sum()),
                ))


# -------------------------
# Other models (panel)
# -------------------------
if ENABLE_OTHER_MODELS:
    specs = other_model_specs()
    print(f"\n[INFO] Other-model panel size: {len(specs)} configs")

    for spec in specs:
        model_name = spec["model"]

        if spec.get("scaler_name", None) is not None:
            sn = spec["scaler_name"]
            scaler_iter = [(sn, scalers[sn])]
        elif spec["requires_scaler"]:
            scaler_iter = list(scalers.items())
        else:
            scaler_iter = [("Identity", None)]

        for scaler_name, scaler_obj in scaler_iter:
            store_key = f"{scaler_name}__{model_name}"
            patient_store.setdefault(store_key, {"y_true": [], "y_pred": []})

            print(f"\n==============================")
            print(f"Scaler={scaler_name} | Model={model_name}")
            print(f"==============================")

            for fold, (p_tr, p_te) in enumerate(mskf.split(unique_subjects, subject_Y), start=1):
                train_subjects = unique_subjects[p_tr]
                test_subjects = unique_subjects[p_te]
                assert len(set(train_subjects).intersection(set(test_subjects))) == 0

                train_idx = np.concatenate([subject_to_window_idx[p] for p in train_subjects])
                test_idx = np.concatenate([subject_to_window_idx[p] for p in test_subjects])

                X_train = X[train_idx]
                X_test = X[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                g_train = groups[train_idx]
                g_test = groups[test_idx]

                n_labels = y.shape[1]
                window_proba_test = np.zeros((len(test_idx), n_labels), dtype=float)
                window_proba_train = np.zeros((len(train_idx), n_labels), dtype=float)

                smote_used = {}
                smote_k_used = {}
                smote_strategy_used = {}
                fitted_pipes_by_label = {}

                for j, lab in enumerate(symptoms):
                    y_train_bin = y_train[:, j].astype(int)

                    if len(np.unique(y_train_bin)) < 2:
                        const_p = float(np.mean(y_train_bin))
                        window_proba_test[:, j] = const_p
                        window_proba_train[:, j] = const_p
                        smote_used[lab] = False
                        smote_k_used[lab] = None
                        smote_strategy_used[lab] = None
                        fitted_pipes_by_label[lab] = None
                        continue

                    est = spec["build"]()
                    est = configure_estimator_for_label_general(est, y_train_bin)

                    if spec.get("needs_safe_calibration", False):
                        est = safe_calibrate_or_raw_linear_svc(est, y_train_bin, method=CALIB_METHOD, max_cv=3)

                    pipe, used, k_used, strat_used = build_pipeline_generic(
                        scaler_obj, est, y_train_bin,
                        label=lab, n_features=X_train.shape[1],
                        enable_smote=bool(USE_SMOTE_FOR_OTHERS),
                        apply_v71b=bool(APPLY_V71B_TO_OTHERS),
                    )

                    smote_used[lab] = bool(used)
                    smote_k_used[lab] = k_used
                    smote_strategy_used[lab] = strat_used

                    pipe.fit(X_train, y_train_bin)
                    fitted_pipes_by_label[lab] = pipe

                    if hasattr(pipe, "predict_proba"):
                        proba_test = np.asarray(pipe.predict_proba(X_test))
                        proba_train = np.asarray(pipe.predict_proba(X_train))
                        window_proba_test[:, j] = proba_test[:, 1]
                        window_proba_train[:, j] = proba_train[:, 1]
                    else:
                        if hasattr(pipe, "decision_function"):
                            window_proba_test[:, j] = decision_function_to_proba(pipe, X_test)
                            window_proba_train[:, j] = decision_function_to_proba(pipe, X_train)
                        else:
                            yp_test = pipe.predict(X_test).astype(int)
                            yp_train = pipe.predict(X_train).astype(int)
                            window_proba_test[:, j] = yp_test.astype(float)
                            window_proba_train[:, j] = yp_train.astype(float)

                subj_train, subj_proba_train = aggregate_patient_probs(
                    g_train, window_proba_train, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
                )
                subj_test, subj_proba_test = aggregate_patient_probs(
                    g_test, window_proba_test, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
                )

                subj_y_train = np.vstack([(y_train[g_train == p].max(axis=0)).astype(int) for p in subj_train])
                subj_y_test = np.vstack([(y_test[g_test == p].max(axis=0)).astype(int) for p in subj_test])

                is_control_train = np.array([str(p).startswith("C") for p in subj_train], dtype=bool)

                alpha_dict = CLINICAL_COST_ALPHA_BY_LABEL if APPLY_V71B_TO_OTHERS else {}
                thresholds = tune_thresholds_policy(
                    subj_y_train,
                    subj_proba_train,
                    labels=symptoms,
                    policy=THRESHOLD_POLICY,
                    spec_target=SPEC_TARGET,
                    grid=THRESH_GRID,
                    is_control_pat=is_control_train,
                    control_fpr_target=CONTROL_FPR_TARGET,
                    control_fpr_target_by_label=CONTROL_FPR_TARGET_BY_LABEL,
                    control_fp_max=CONTROL_FP_MAX,
                    control_fp_max_by_label=CONTROL_FP_MAX_BY_LABEL,
                    spec_target_by_label=SPEC_TARGET_BY_LABEL,
                    clinical_cost_alpha_by_label=alpha_dict,
                )

                subj_pred_test = (subj_proba_test >= thresholds[None, :]).astype(int)

                if ENABLE_FEATURE_IMPORTANCE:
                    feature_names = list(X_df.columns)

                    for j, lab in enumerate(symptoms):
                        pipe_fit = fitted_pipes_by_label.get(lab, None)
                        if pipe_fit is None:
                            continue

                        thr_lab = float(thresholds[j])
                        y_te_bin = y_test[:, j].astype(int)

                        imp_df = permutation_importance_patient_level_error(
                            pipe=pipe_fit,
                            X_test=X_test,
                            y_test_bin=y_te_bin,
                            g_test=g_test,
                            threshold=thr_lab,
                            feature_names=feature_names,
                            agg_mode=PATIENT_AGG_MODE,
                            topk=TOPK,
                            q=PERCENTILE_Q,
                            n_repeats=FEATURE_IMPORTANCE_N_REPEATS,
                            max_features=FEATURE_IMPORTANCE_MAX_FEATURES,
                            rng_seed=FEATURE_IMPORTANCE_RANDOM_STATE + 1000*fold + 13*j,
                        )

                        imp_df["fold"] = int(fold)
                        imp_df["label"] = str(lab)
                        imp_df["scaler"] = str(scaler_name)
                        imp_df["model"] = str(spec["model"])
                        imp_df["model_key"] = str(store_key)

                        feature_importance_long_rows.append(imp_df)

                patient_store[store_key]["y_true"].append(subj_y_test)
                patient_store[store_key]["y_pred"].append(subj_pred_test)

                is_control_test = np.array([str(p).startswith("C") for p in subj_test], dtype=bool)
                controls_fpr = {}
                controls_fp = {}
                if np.any(is_control_test):
                    for j, lab in enumerate(symptoms):
                        yt_c = subj_y_test[is_control_test, j]
                        yp_c = subj_pred_test[is_control_test, j]
                        cm = confusion_matrix(yt_c, yp_c, labels=[0, 1])
                        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                        denom = fp + tn
                        controls_fpr[lab] = float(fp / denom) if denom > 0 else np.nan
                        controls_fp[lab] = int(fp)
                else:
                    controls_fpr = {lab: np.nan for lab in symptoms}
                    controls_fp = {lab: np.nan for lab in symptoms}

                macro_auc = safe_macro_auc(subj_y_test, subj_proba_test)
                macro_auprc = safe_macro_auprc(subj_y_test, subj_proba_test)
                micro_f1 = f1_score(subj_y_test, subj_pred_test, average="micro", zero_division=0)
                macro_f1 = f1_score(subj_y_test, subj_pred_test, average="macro", zero_division=0)
                jac = jaccard_score(subj_y_test, subj_pred_test, average="samples", zero_division=0)
                ham_loss = hamming_loss(subj_y_test, subj_pred_test)
                ham_acc = 1.0 - float(ham_loss)
                exact_match = accuracy_score(subj_y_test, subj_pred_test)

                clinical_cost = clinical_cost_from_confusions(subj_y_test, subj_pred_test)
                clinical_cost_w = clinical_cost_weighted_from_confusions(subj_y_test, subj_pred_test, symptoms, CLINICAL_COST_ALPHA_BY_LABEL)

                for j, lab in enumerate(symptoms):
                    yt = subj_y_test[:, j]
                    yp = subj_pred_test[:, j]
                    ys = subj_proba_test[:, j]

                    prec = precision_score(yt, yp, zero_division=0)
                    rec = recall_score(yt, yp, zero_division=0)
                    f1v = f1_score(yt, yp, zero_division=0)
                    specv = specificity_score(yt, yp)

                    auc = np.nan
                    auprc = np.nan
                    if len(np.unique(yt)) >= 2:
                        try:
                            auc = roc_auc_score(yt, ys)
                        except Exception:
                            auc = np.nan
                        try:
                            auprc = average_precision_score(yt, ys)
                        except Exception:
                            auprc = np.nan

                    per_label_rows.append(dict(
                        scaler=scaler_name,
                        model=model_name,
                        fold=int(fold),
                        label=lab,
                        precision=float(prec),
                        recall=float(rec),
                        specificity=float(specv) if specv == specv else np.nan,
                        f1=float(f1v),
                        auc=float(auc) if auc == auc else np.nan,
                        auprc=float(auprc) if auprc == auprc else np.nan,
                        threshold=float(thresholds[j]),
                        smote_used=bool(smote_used.get(lab, False)),
                        smote_k=smote_k_used.get(lab, None),
                        smote_strategy=smote_strategy_used.get(lab, None),
                        support_pos=int(np.sum(yt == 1)),
                        support_neg=int(np.sum(yt == 0)),
                    ))

                print(
                    f"  ▶ Fold {fold}/{n_splits}: "
                    f"macroAUC={macro_auc:.3f} | macroAUPRC={macro_auprc:.3f} | "
                    f"microF1={micro_f1:.3f} | macroF1={macro_f1:.3f} | "
                    f"Jaccard={jac:.3f} | HammingAcc={ham_acc:.3f} | ExactMatch={exact_match:.3f} | "
                    f"ClinicalCost={clinical_cost:.3f} | ClinicalCostW={clinical_cost_w:.3f}"
                )

                all_rows.append(dict(
                    scaler=scaler_name,
                    model=model_name,
                    fold=int(fold),
                    n_train_subjects=int(len(train_subjects)),
                    n_test_subjects=int(len(test_subjects)),
                    n_train_windows=int(len(train_idx)),
                    n_test_windows=int(len(test_idx)),
                    patient_macro_auc=float(macro_auc) if macro_auc == macro_auc else np.nan,
                    patient_macro_auprc=float(macro_auprc) if macro_auprc == macro_auprc else np.nan,
                    patient_micro_f1=float(micro_f1),
                    patient_macro_f1=float(macro_f1),
                    patient_jaccard_samples=float(jac),
                    patient_hamming_loss=float(ham_loss),
                    patient_hamming_acc=float(ham_acc),
                    patient_exact_match=float(exact_match),
                    patient_clinical_cost=float(clinical_cost) if clinical_cost == clinical_cost else np.nan,
                    patient_clinical_cost_weighted=float(clinical_cost_w) if clinical_cost_w == clinical_cost_w else np.nan,
                    thresholds=json.dumps({symptoms[j]: float(thresholds[j]) for j in range(len(symptoms))}),
                    smote_used=json.dumps(smote_used),
                    smote_k_used=json.dumps(smote_k_used),
                    smote_strategy_used=json.dumps(smote_strategy_used),
                    controls_only_fpr=json.dumps(controls_fpr),
                    controls_only_fp=json.dumps(controls_fp),
                    n_controls_test=int(is_control_test.sum()),
                ))

# -------------------------
# Nested CV: best model per label selected on TRAIN only; evaluated on TEST
# -------------------------
nested_rows = []
nested_label_rows = []
nested_patient_store = {}

def enumerate_all_candidate_configs():
    """
    Candidate list for nested label selection:
      - all SVM configs (scaler x grid)
      - all other-model specs (their scaler(s) or Identity)
    """
    candidates = []

    for scaler_name, scaler in scalers.items():
        for svm_hp in SVM_GRID:
            model_name = f"SVM(C={svm_hp['C']},gamma={svm_hp['gamma']})"
            key = f"{scaler_name}__{model_name}"
            candidates.append(dict(
                key=key,
                scaler_name=scaler_name,
                scaler_obj=scaler,
                family="SVM",
                model_name=model_name,
                kind="svm",
                svm_hp=svm_hp,
            ))

    if ENABLE_OTHER_MODELS:
        for spec in other_model_specs():
            model_name = spec["model"]
            if spec.get("scaler_name", None) is not None:
                sn = spec["scaler_name"]
                scaler_iter = [(sn, scalers[sn])]
            elif spec["requires_scaler"]:
                scaler_iter = list(scalers.items())
            else:
                scaler_iter = [("Identity", None)]

            for scaler_name, scaler_obj in scaler_iter:
                key = f"{scaler_name}__{model_name}"
                candidates.append(dict(
                    key=key,
                    scaler_name=scaler_name,
                    scaler_obj=scaler_obj,
                    family=spec.get("family", "OTHER"),
                    model_name=model_name,
                    kind="other",
                    spec=spec,
                ))

    return candidates


def fit_predict_label_proba(candidate, X_tr, y_tr_bin, X_eval, label, apply_v71b, enable_smote):
    """
    Fit a single-label binary model and output proba for eval.

    Robust:
      - constant/degenerate y -> constant predictor
      - any fit/predict failure -> returns None
    """
    y_tr_bin = np.asarray(y_tr_bin).astype(int)

    if len(np.unique(y_tr_bin)) < 2:
        const_p = float(np.mean(y_tr_bin))
        return np.full(X_eval.shape[0], const_p, dtype=float)

    if candidate["kind"] == "svm":
        hp = candidate["svm_hp"]
        est = SVC(
            C=float(hp["C"]),
            kernel="rbf",
            gamma=hp["gamma"],
            probability=True,
            random_state=random_state,
        )
    else:
        est = candidate["spec"]["build"]()

    est = configure_estimator_for_label_general(est, y_tr_bin)

    if candidate["kind"] == "other":
        spec = candidate.get("spec", {})
        if spec.get("needs_safe_calibration", False):
            est = safe_calibrate_or_raw_linear_svc(est, y_tr_bin, method=CALIB_METHOD, max_cv=3)

    pipe, _, _, _ = build_pipeline_generic(
        candidate["scaler_obj"], est, y_tr_bin,
        label=label, n_features=X_tr.shape[1],
        enable_smote=bool(enable_smote),
        apply_v71b=bool(apply_v71b),
    )

    try:
        pipe.fit(X_tr, y_tr_bin)
    except Exception:
        return None

    try:
        if hasattr(pipe, "predict_proba"):
            proba = np.asarray(pipe.predict_proba(X_eval))[:, 1]
            return proba.astype(float)
    except Exception:
        return None

    try:
        if hasattr(pipe, "decision_function"):
            return decision_function_to_proba(pipe, X_eval)
    except Exception:
        return None

    try:
        return pipe.predict(X_eval).astype(float)
    except Exception:
        return None


def inner_score_candidate_constrained_error(
    cand,
    X_train_outer, y_train_outer_label, g_train_outer,
    inner_tr_subjects, inner_va_subjects,
    tr_subject_to_window_idx,
    label_name,
):
    """
    One inner split:
      1) fit model on inner-tr windows
      2) get proba on inner-tr windows and inner-val windows
      3) aggregate to patient-level
      4) choose threshold on inner-tr patients under constraints (with relaxation if needed)
      5) compute error rate on inner-val patients
    Returns: (err_rate, diag_dict) or (None, diag_dict)
    """
    inner_tr_idx = np.concatenate([tr_subject_to_window_idx[p] for p in inner_tr_subjects])
    inner_va_idx = np.concatenate([tr_subject_to_window_idx[p] for p in inner_va_subjects])

    X_tr_in = X_train_outer[inner_tr_idx]
    X_va_in = X_train_outer[inner_va_idx]
    y_tr_in = y_train_outer_label[inner_tr_idx].astype(int)
    y_va_in = y_train_outer_label[inner_va_idx].astype(int)

    g_tr_in = g_train_outer[inner_tr_idx]
    g_va_in = g_train_outer[inner_va_idx]

    apply_v71b = (APPLY_V71B_TO_SVM if cand["kind"] == "svm" else APPLY_V71B_TO_OTHERS)
    enable_smote = (USE_SMOTE_FOR_SVM if cand["kind"] == "svm" else USE_SMOTE_FOR_OTHERS)

    proba_tr = fit_predict_label_proba(
        cand, X_tr_in, y_tr_in, X_tr_in,
        label=label_name, apply_v71b=apply_v71b, enable_smote=enable_smote
    )
    proba_va = fit_predict_label_proba(
        cand, X_tr_in, y_tr_in, X_va_in,
        label=label_name, apply_v71b=apply_v71b, enable_smote=enable_smote
    )
    if (proba_tr is None) or (proba_va is None):
        return None, {"status": "fit_or_predict_failed"}

    _, y_pat_tr, p_pat_tr, is_ctrl_tr = aggregate_patient_single_label(
        g_tr_in, y_tr_in, proba_tr,
        agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
    )
    _, y_pat_va, p_pat_va, _ = aggregate_patient_single_label(
        g_va_in, y_va_in, proba_va,
        agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
    )

    chosen_t = None
    chosen_diag = None
    for rl in NESTED_CONSTRAINT_RELAX_LEVELS:
        t, diag = select_threshold_min_error_under_constraints(
            y_pat_tr, p_pat_tr, label=label_name, grid=THRESH_GRID,
            is_control_pat=is_ctrl_tr, relax_level=int(rl)
        )
        if diag.get("feasible", False):
            chosen_t = float(t)
            chosen_diag = diag
            break

    if chosen_t is None:
        chosen_t = 0.5
        chosen_diag = {"feasible": False, "relax_level": 999, "n_feasible": 0}

    y_pred_va = (p_pat_va >= chosen_t).astype(int)
    tn, fp, fn, tp = _confusion_counts_binary(y_pat_va, y_pred_va)
    err_rate = (fp + fn) / max(1, (tp + tn + fp + fn))

    return float(err_rate), {
        "status": "ok",
        "threshold": float(chosen_t),
        "relax_level": int(chosen_diag.get("relax_level", -1)),
        "n_feasible": int(chosen_diag.get("n_feasible", 0)),
        "val_fp": int(fp),
        "val_fn": int(fn),
        "val_tp": int(tp),
        "val_tn": int(tn),
    }


if ENABLE_NESTED_LABEL_MODEL_SELECTION:
    candidates = enumerate_all_candidate_configs()
    print(f"\n[INFO] Nested label selection candidates: {len(candidates)} configs")
    print(f"[INFO] Nested objective: {NESTED_SELECTION_OBJECTIVE}")

    nested_store_key = "NestedLabelSelection"
    nested_patient_store[nested_store_key] = {"y_true": [], "y_pred": []}
    nested_patient_store[nested_store_key + "__choices"] = []
    nested_patient_store[nested_store_key + "__choices_meta"] = []

    for fold, (p_tr, p_te) in enumerate(mskf.split(unique_subjects, subject_Y), start=1):
        train_subjects = unique_subjects[p_tr]
        test_subjects = unique_subjects[p_te]
        assert len(set(train_subjects).intersection(set(test_subjects))) == 0

        train_idx = np.concatenate([subject_to_window_idx[p] for p in train_subjects])
        test_idx = np.concatenate([subject_to_window_idx[p] for p in test_subjects])

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]
        g_test = groups[test_idx]

        inner_splits = int(min(NESTED_INNER_SPLITS, max(2, len(train_subjects))))
        inner_splits = int(min(inner_splits, len(train_subjects)))
        inner_mskf = MultilabelStratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

        tr_subjects = np.unique(g_train)
        tr_subject_Y = np.array([(y_train[g_train == p].max(axis=0)).astype(int) for p in tr_subjects], dtype=int)
        tr_subject_to_window_idx = {p: np.where(g_train == p)[0] for p in tr_subjects}

        chosen_key_by_label = {}
        chosen_diag_by_label = {}

        print(f"\n[NESTED] Outer fold {fold}/{n_splits}: selecting best model per label (inner_splits={inner_splits})")

        for j, lab in enumerate(symptoms):
            pat_y_train_label = tr_subject_Y[:, j]
            if len(np.unique(pat_y_train_label)) < 2:
                chosen_key_by_label[lab] = None
                chosen_diag_by_label[lab] = {"reason": "constant_label_on_outer_train"}
                continue

            best_err = +1e18
            best_candidate = None
            best_meta = None

            for cand in candidates:
                errs = []
                metas = []

                for i_tr, i_va in inner_mskf.split(tr_subjects, tr_subject_Y):
                    inner_tr_subjects = tr_subjects[i_tr]
                    inner_va_subjects = tr_subjects[i_va]

                    err, meta = inner_score_candidate_constrained_error(
                        cand,
                        X_train, y_train[:, j], g_train,
                        inner_tr_subjects, inner_va_subjects,
                        tr_subject_to_window_idx,
                        label_name=lab,
                    )
                    if err is None:
                        continue
                    errs.append(float(err))
                    metas.append(meta)

                if len(errs) == 0:
                    continue

                mean_err = float(np.mean(errs))
                if mean_err < best_err:
                    best_err = mean_err
                    best_candidate = cand
                    best_meta = {
                        "mean_err": float(mean_err),
                        "mean_hamming_acc": float(1.0 - mean_err),
                        "n_inner_used": int(len(errs)),
                        "relax_level_mean": float(np.mean([m.get("relax_level", np.nan) for m in metas])),
                    }

            chosen_key_by_label[lab] = best_candidate["key"] if best_candidate else None
            chosen_diag_by_label[lab] = best_meta if best_meta else {"reason": "no_candidate_scored"}

            if best_candidate:
                print(f"  [NESTED] label={lab:12s} best_inner_err={best_err:.3f} (HammAcc={1.0-best_err:.3f}) -> {best_candidate['key']}")
            else:
                print(f"  [NESTED] label={lab:12s} best_inner_err=NA -> None")

        n_labels = len(symptoms)
        window_proba_train = np.zeros((len(train_idx), n_labels), dtype=float)
        window_proba_test = np.zeros((len(test_idx), n_labels), dtype=float)

        for j, lab in enumerate(symptoms):
            key = chosen_key_by_label[lab]
            if key is None:
                const_p = float(np.mean(y_train[:, j].astype(int)))
                window_proba_train[:, j] = const_p
                window_proba_test[:, j] = const_p
                continue

            cand = next((c for c in candidates if c["key"] == key), None)
            assert cand is not None, f"Chosen key not found among candidates: {key}"

            y_tr_bin = y_train[:, j].astype(int)

            apply_v71b = (APPLY_V71B_TO_SVM if cand["kind"] == "svm" else APPLY_V71B_TO_OTHERS)
            enable_smote = (USE_SMOTE_FOR_SVM if cand["kind"] == "svm" else USE_SMOTE_FOR_OTHERS)

            proba_tr = fit_predict_label_proba(
                cand, X_train, y_tr_bin, X_train,
                label=lab, apply_v71b=apply_v71b, enable_smote=enable_smote
            )
            proba_te = fit_predict_label_proba(
                cand, X_train, y_tr_bin, X_test,
                label=lab, apply_v71b=apply_v71b, enable_smote=enable_smote
            )

            if (proba_tr is None) or (proba_te is None):
                const_p = float(np.mean(y_tr_bin))
                window_proba_train[:, j] = const_p
                window_proba_test[:, j] = const_p
            else:
                window_proba_train[:, j] = proba_tr
                window_proba_test[:, j] = proba_te

        subj_train, subj_proba_train = aggregate_patient_probs(
            g_train, window_proba_train, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
        )
        subj_test, subj_proba_test = aggregate_patient_probs(
            g_test, window_proba_test, agg_mode=PATIENT_AGG_MODE, topk=TOPK, q=PERCENTILE_Q
        )

        subj_y_train = np.vstack([(y_train[g_train == p].max(axis=0)).astype(int) for p in subj_train])
        subj_y_test = np.vstack([(y_test[g_test == p].max(axis=0)).astype(int) for p in subj_test])

        is_control_train = np.array([str(p).startswith("C") for p in subj_train], dtype=bool)

        thresholds = tune_thresholds_policy(
            subj_y_train,
            subj_proba_train,
            labels=symptoms,
            policy=THRESHOLD_POLICY,
            spec_target=SPEC_TARGET,
            grid=THRESH_GRID,
            is_control_pat=is_control_train,
            control_fpr_target=CONTROL_FPR_TARGET,
            control_fpr_target_by_label=CONTROL_FPR_TARGET_BY_LABEL,
            control_fp_max=CONTROL_FP_MAX,
            control_fp_max_by_label=CONTROL_FP_MAX_BY_LABEL,
            spec_target_by_label=SPEC_TARGET_BY_LABEL,
            clinical_cost_alpha_by_label=CLINICAL_COST_ALPHA_BY_LABEL,
        )

        subj_pred_test = (subj_proba_test >= thresholds[None, :]).astype(int)

        nested_patient_store[nested_store_key]["y_true"].append(subj_y_test)
        nested_patient_store[nested_store_key]["y_pred"].append(subj_pred_test)
        nested_patient_store[nested_store_key + "__choices"].append(dict(fold=int(fold), **chosen_key_by_label))
        nested_patient_store[nested_store_key + "__choices_meta"].append(dict(fold=int(fold), **chosen_diag_by_label))

        macro_auc = safe_macro_auc(subj_y_test, subj_proba_test)
        macro_auprc = safe_macro_auprc(subj_y_test, subj_proba_test)
        micro_f1 = f1_score(subj_y_test, subj_pred_test, average="micro", zero_division=0)
        macro_f1 = f1_score(subj_y_test, subj_pred_test, average="macro", zero_division=0)
        jac = jaccard_score(subj_y_test, subj_pred_test, average="samples", zero_division=0)
        ham_loss = hamming_loss(subj_y_test, subj_pred_test)
        ham_acc = 1.0 - float(ham_loss)
        exact_match = accuracy_score(subj_y_test, subj_pred_test)
        clinical_cost = clinical_cost_from_confusions(subj_y_test, subj_pred_test)
        clinical_cost_w = clinical_cost_weighted_from_confusions(subj_y_test, subj_pred_test, symptoms, CLINICAL_COST_ALPHA_BY_LABEL)

        nested_rows.append(dict(
            method="NestedBestPerLabel_ConstrainedError",
            fold=int(fold),
            n_train_subjects=int(len(train_subjects)),
            n_test_subjects=int(len(test_subjects)),
            patient_macro_auc=float(macro_auc) if macro_auc == macro_auc else np.nan,
            patient_macro_auprc=float(macro_auprc) if macro_auprc == macro_auprc else np.nan,
            patient_micro_f1=float(micro_f1),
            patient_macro_f1=float(macro_f1),
            patient_jaccard_samples=float(jac),
            patient_hamming_acc=float(ham_acc),
            patient_exact_match=float(exact_match),
            patient_clinical_cost=float(clinical_cost) if clinical_cost == clinical_cost else np.nan,
            patient_clinical_cost_weighted=float(clinical_cost_w) if clinical_cost_w == clinical_cost_w else np.nan,
            thresholds=json.dumps({symptoms[j]: float(thresholds[j]) for j in range(len(symptoms))}),
            chosen_models=json.dumps(chosen_key_by_label),
            chosen_meta=json.dumps(chosen_diag_by_label),
        ))

        for j, lab in enumerate(symptoms):
            yt = subj_y_test[:, j]
            yp = subj_pred_test[:, j]
            ys = subj_proba_test[:, j]

            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1v = f1_score(yt, yp, zero_division=0)
            specv = specificity_score(yt, yp)

            auc = np.nan
            auprc = np.nan
            if len(np.unique(yt)) >= 2:
                try:
                    auc = roc_auc_score(yt, ys)
                except Exception:
                    auc = np.nan
                try:
                    auprc = average_precision_score(yt, ys)
                except Exception:
                    auprc = np.nan

            nested_label_rows.append(dict(
                method="NestedBestPerLabel_ConstrainedError",
                fold=int(fold),
                label=lab,
                chosen_key=chosen_key_by_label.get(lab, None),
                chosen_meta=json.dumps(chosen_diag_by_label.get(lab, {})),
                precision=float(prec),
                recall=float(rec),
                specificity=float(specv) if specv == specv else np.nan,
                f1=float(f1v),
                auc=float(auc) if auc == auc else np.nan,
                auprc=float(auprc) if auprc == auprc else np.nan,
                threshold=float(thresholds[j]),
                support_pos=int(np.sum(yt == 1)),
                support_neg=int(np.sum(yt == 0)),
            ))

        print(
            f"[NESTED] Fold {fold}/{n_splits}: "
            f"macroAUC={macro_auc:.3f} | macroAUPRC={macro_auprc:.3f} | "
            f"microF1={micro_f1:.3f} | HammingAcc={ham_acc:.3f} | ExactMatch={exact_match:.3f} | "
            f"ClinicalCostW={clinical_cost_w:.3f}"
        )


# -------------------------
# Summaries + decompositions + save
# -------------------------
results_df = pd.DataFrame(all_rows)
per_label_df = pd.DataFrame(per_label_rows)

controls_rows = []
for _, r in results_df.iterrows():
    d_fpr = json.loads(r["controls_only_fpr"]) if isinstance(r["controls_only_fpr"], str) else {}
    d_fp = json.loads(r["controls_only_fp"]) if isinstance(r["controls_only_fp"], str) else {}
    for lab in symptoms:
        controls_rows.append(dict(
            scaler=r["scaler"],
            model=r["model"],
            fold=r["fold"],
            label=lab,
            controls_fpr=d_fpr.get(lab, np.nan),
            controls_fp=d_fp.get(lab, np.nan),
            n_controls_test=r.get("n_controls_test", np.nan),
        ))
controls_df = pd.DataFrame(controls_rows)

controls_summary = (
    controls_df
    .groupby(["scaler", "model", "label"], as_index=False)
    .agg(
        controls_fpr_mean=("controls_fpr", "mean"),
        controls_fpr_std=("controls_fpr", "std"),
        controls_fp_mean=("controls_fp", "mean"),
        controls_fp_std=("controls_fp", "std"),
        n_folds=("fold", "count"),
    )
    .sort_values(["controls_fp_mean", "controls_fpr_mean"], ascending=True)
)

summary = (
    results_df
    .groupby(["scaler", "model"], as_index=False)
    .agg(
        patient_macro_auc_mean=("patient_macro_auc", "mean"),
        patient_macro_auc_std=("patient_macro_auc", "std"),
        patient_macro_auprc_mean=("patient_macro_auprc", "mean"),
        patient_macro_auprc_std=("patient_macro_auprc", "std"),
        patient_micro_f1_mean=("patient_micro_f1", "mean"),
        patient_micro_f1_std=("patient_micro_f1", "std"),
        patient_macro_f1_mean=("patient_macro_f1", "mean"),
        patient_macro_f1_std=("patient_macro_f1", "std"),
        patient_jaccard_samples_mean=("patient_jaccard_samples", "mean"),
        patient_jaccard_samples_std=("patient_jaccard_samples", "std"),
        patient_hamming_acc_mean=("patient_hamming_acc", "mean"),
        patient_hamming_acc_std=("patient_hamming_acc", "std"),
        patient_exact_match_mean=("patient_exact_match", "mean"),
        patient_exact_match_std=("patient_exact_match", "std"),
        patient_clinical_cost_mean=("patient_clinical_cost", "mean"),
        patient_clinical_cost_std=("patient_clinical_cost", "std"),
        patient_clinical_cost_weighted_mean=("patient_clinical_cost_weighted", "mean"),
        patient_clinical_cost_weighted_std=("patient_clinical_cost_weighted", "std"),
        n_folds=("fold", "count"),
    )
    .sort_values(
        ["patient_macro_auprc_mean", "patient_macro_auc_mean", "patient_micro_f1_mean"],
        ascending=False
    )
    .reset_index(drop=True)
)

label_summary = (
    per_label_df
    .groupby(["scaler", "model", "label"], as_index=False)
    .agg(
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        f1_mean=("f1", "mean"),
        auc_mean=("auc", "mean"),
        auprc_mean=("auprc", "mean"),
        threshold_mean=("threshold", "mean"),
        support_pos_mean=("support_pos", "mean"),
        support_neg_mean=("support_neg", "mean"),
        smote_used_rate=("smote_used", "mean"),
    )
    .sort_values(["auprc_mean", "auc_mean", "f1_mean"], ascending=False)
)

def key_from_row(row):
    return f"{row['scaler']}__{row['model']}"

best_by_auprc = key_from_row(summary.iloc[0]) if len(summary) else None

summary_by_hamming = summary.sort_values(
    ["patient_hamming_acc_mean", "patient_jaccard_samples_mean", "patient_micro_f1_mean"],
    ascending=False
).reset_index(drop=True)

summary_by_jaccard = summary.sort_values(
    ["patient_jaccard_samples_mean", "patient_hamming_acc_mean", "patient_micro_f1_mean"],
    ascending=False
).reset_index(drop=True)

summary_by_auc = summary.sort_values(
    ["patient_macro_auc_mean", "patient_macro_auprc_mean", "patient_micro_f1_mean"],
    ascending=False
).reset_index(drop=True)

summary_by_cost = summary.sort_values(
    ["patient_clinical_cost_mean", "patient_hamming_acc_mean", "patient_jaccard_samples_mean"],
    ascending=[True, False, False]
).reset_index(drop=True)

best_by_hamming = key_from_row(summary_by_hamming.iloc[0]) if len(summary_by_hamming) else None
best_by_jaccard = key_from_row(summary_by_jaccard.iloc[0]) if len(summary_by_jaccard) else None
best_by_auc = key_from_row(summary_by_auc.iloc[0]) if len(summary_by_auc) else None
best_by_cost = key_from_row(summary_by_cost.iloc[0]) if len(summary_by_cost) else None

print("\n[Best keys]")
print(" best_by_auprc   =", best_by_auprc)
print(" best_by_hamming =", best_by_hamming)
print(" best_by_jaccard =", best_by_jaccard)
print(" best_by_auc     =", best_by_auc)
print(" best_by_cost    =", best_by_cost)

for k in [best_by_auprc, best_by_hamming, best_by_jaccard, best_by_auc, best_by_cost]:
    if k is None:
        continue
    assert k in patient_store, f"{k} not in patient_store"
    assert len(patient_store[k]["y_true"]) == n_splits, f"{k} missing folds in y_true"
    assert len(patient_store[k]["y_pred"]) == n_splits, f"{k} missing folds in y_pred"

error_decomp_best_auprc_df = build_error_decomposition(best_by_auprc, patient_store, symptoms) if best_by_auprc else pd.DataFrame()
error_decomp_best_hamming_df = build_error_decomposition(best_by_hamming, patient_store, symptoms) if best_by_hamming else pd.DataFrame()
error_decomp_best_jaccard_df = build_error_decomposition(best_by_jaccard, patient_store, symptoms) if best_by_jaccard else pd.DataFrame()
error_decomp_best_auc_df = build_error_decomposition(best_by_auc, patient_store, symptoms) if best_by_auc else pd.DataFrame()
error_decomp_best_cost_df = build_error_decomposition(best_by_cost, patient_store, symptoms) if best_by_cost else pd.DataFrame()

nested_df = pd.DataFrame(nested_rows) if len(nested_rows) else pd.DataFrame()
nested_label_df = pd.DataFrame(nested_label_rows) if len(nested_label_rows) else pd.DataFrame()

nested_summary = pd.DataFrame()
nested_label_summary = pd.DataFrame()
nested_error_decomp = pd.DataFrame()
nested_choices_df = pd.DataFrame()

if len(nested_df):
    nested_summary = (
        nested_df
        .groupby(["method"], as_index=False)
        .agg(
            patient_macro_auc_mean=("patient_macro_auc", "mean"),
            patient_macro_auc_std=("patient_macro_auc", "std"),
            patient_macro_auprc_mean=("patient_macro_auprc", "mean"),
            patient_macro_auprc_std=("patient_macro_auprc", "std"),
            patient_micro_f1_mean=("patient_micro_f1", "mean"),
            patient_micro_f1_std=("patient_micro_f1", "std"),
            patient_hamming_acc_mean=("patient_hamming_acc", "mean"),
            patient_hamming_acc_std=("patient_hamming_acc", "std"),
            patient_exact_match_mean=("patient_exact_match", "mean"),
            patient_exact_match_std=("patient_exact_match", "std"),
            patient_clinical_cost_weighted_mean=("patient_clinical_cost_weighted", "mean"),
            patient_clinical_cost_weighted_std=("patient_clinical_cost_weighted", "std"),
            n_folds=("fold", "count"),
        )
    )

if len(nested_label_df):
    nested_label_summary = (
        nested_label_df
        .groupby(["method", "label", "chosen_key"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            specificity_mean=("specificity", "mean"),
            f1_mean=("f1", "mean"),
            auc_mean=("auc", "mean"),
            auprc_mean=("auprc", "mean"),
            threshold_mean=("threshold", "mean"),
            support_pos_mean=("support_pos", "mean"),
            support_neg_mean=("support_neg", "mean"),
            n_folds=("fold", "count"),
        )
        .sort_values(["label", "auprc_mean"], ascending=[True, False])
    )

if ENABLE_NESTED_LABEL_MODEL_SELECTION and len(nested_patient_store.get("NestedLabelSelection", {}).get("y_true", [])):
    nested_error_decomp = build_error_decomposition("NestedLabelSelection", nested_patient_store, symptoms)

if ENABLE_NESTED_LABEL_MODEL_SELECTION:
    nested_choices_df = pd.DataFrame(nested_patient_store.get("NestedLabelSelection__choices", []))

out_path = os.path.join(output_dir, "BR_PANEL_patientlevel_results_V7_3_HAMMING_FINAL.xlsx")


# -------------------------
# Feature importance summaries
# -------------------------
feature_importance_long = pd.DataFrame()
feature_importance_summary = pd.DataFrame()

if ENABLE_FEATURE_IMPORTANCE and ("feature_importance_long_rows" in globals()) and len(feature_importance_long_rows):
    feature_importance_long = pd.concat(feature_importance_long_rows, ignore_index=True)

    feature_importance_summary = (
        feature_importance_long
        .groupby(["label", "feature"], as_index=False)
        .agg(
            importance_delta_error_mean=("importance_delta_error_mean", "mean"),
            importance_delta_error_std=("importance_delta_error_mean", "std"),
            baseline_error_mean=("baseline_error", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["label", "importance_delta_error_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    feature_importance_summary_by_model = (
        feature_importance_long
        .groupby(["model_key", "scaler", "model", "label", "feature"], as_index=False)
        .agg(
            importance_delta_error_mean=("importance_delta_error_mean", "mean"),
            importance_delta_error_std=("importance_delta_error_mean", "std"),
            baseline_error_mean=("baseline_error", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["model_key", "label", "importance_delta_error_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    pd.DataFrame([run_meta]).to_excel(writer, sheet_name="run_meta", index=False)

    summary.to_excel(writer, sheet_name="summary", index=False)
    results_df.to_excel(writer, sheet_name="by_fold", index=False)
    label_summary.to_excel(writer, sheet_name="label_summary", index=False)
    per_label_df.to_excel(writer, sheet_name="label_by_fold", index=False)
    controls_summary.to_excel(writer, sheet_name="controls_summary_fp_fpr", index=False)
    controls_df.to_excel(writer, sheet_name="controls_by_fold_fp_fpr", index=False)

    error_decomp_best_auprc_df.to_excel(writer, sheet_name="error_decomp_best_by_AUPRC", index=False)
    error_decomp_best_hamming_df.to_excel(writer, sheet_name="error_decomp_best_by_Hamming", index=False)
    error_decomp_best_jaccard_df.to_excel(writer, sheet_name="error_decomp_best_by_Jaccard", index=False)
    error_decomp_best_auc_df.to_excel(writer, sheet_name="error_decomp_best_by_AUC", index=False)
    error_decomp_best_cost_df.to_excel(writer, sheet_name="error_decomp_best_by_ClinicalCost", index=False)

    if len(nested_df):
        nested_df.to_excel(writer, sheet_name="nested_by_fold", index=False)
    if len(nested_summary):
        nested_summary.to_excel(writer, sheet_name="nested_summary", index=False)
    if len(nested_label_df):
        nested_label_df.to_excel(writer, sheet_name="nested_label_by_fold", index=False)
    if len(nested_label_summary):
        nested_label_summary.to_excel(writer, sheet_name="nested_label_summary", index=False)
    if len(nested_choices_df):
        nested_choices_df.to_excel(writer, sheet_name="nested_choices_by_fold", index=False)
    if len(nested_error_decomp):
        nested_error_decomp.to_excel(writer, sheet_name="nested_error_decomp", index=False)
    if len(feature_importance_long):
        feature_importance_long.to_excel(writer, sheet_name="feature_importance_long", index=False)
    if len(feature_importance_summary):
        feature_importance_summary.to_excel(writer, sheet_name="feature_importance_summary", index=False)
    if len(feature_importance_summary_by_model):
        feature_importance_summary_by_model.to_excel(writer, sheet_name="feature_importance_summary_by_model", index=False)

print("\n✅ Saved:", out_path)
print("\nTop-10 (by patient macro-AUPRC):")
print(summary.head(10).to_string(index=False))

if best_by_hamming:
    print(f"\n[Best by HammingAcc] {best_by_hamming}")
if best_by_jaccard:
    print(f"[Best by Jaccard] {best_by_jaccard}")
if best_by_cost:
    print(f"[Best by ClinicalCost] {best_by_cost}")

if len(error_decomp_best_hamming_df):
    print("\nError decomposition (best by HammingAcc):")
    print(error_decomp_best_hamming_df.sort_values("errors", ascending=False).to_string(index=False))

if ENABLE_NESTED_LABEL_MODEL_SELECTION and len(nested_error_decomp):
    print("\n[NESTED] Error decomposition:")
    print(nested_error_decomp.sort_values("errors", ascending=False).to_string(index=False))
