"""
evaluate.py  —  Combined Ensemble Accuracy Report
===================================================
Loads truth_features_full.xlsx + lie_features_full.xlsx,
runs XGBoost predictions, CNN predictions, and the weighted
ensemble (55% XGBoost + 45% CNN), then prints a full report.

Run:  python evaluate.py
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

XGB_WEIGHT = 0.55
CNN_WEIGHT = 0.45

# ── Load XGBoost model ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  LIE DETECTOR — ENSEMBLE ACCURACY REPORT")
print("="*55)

model           = joblib.load("rf_lie_model.pkl")
label_encoder   = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ── Load data ─────────────────────────────────────────────────────────────────
truth_df = pd.read_excel("truth_features_full.xlsx")
lie_df   = pd.read_excel("lie_features_full.xlsx")
truth_df["label"] = 0
lie_df["label"]   = 1
df = pd.concat([truth_df, lie_df], ignore_index=True)
df = df.drop(columns=["Serial_No"], errors="ignore")

y_true = df["label"].values
X      = df.drop(columns=["label"])

# ── Feature engineering (mirrors model.py) ───────────────────────────────────
def engineer(X):
    X = X.copy()
    X['Pitch_Var_Ratio']    = X['PitchSD']           / (X['MeanPitch']        + 1e-9)
    X['Jitter_Shimmer_Int'] = X['Jitter']            *  X['Shimmer']
    X['HNR_Jitter_Ratio']   = X['HNR']               / (X['Jitter']           + 1e-9)
    X['Spectral_Spread']    = X['SpectralBandwidth'] / (X['SpectralCentroid']  + 1e-9)
    X['MFCC1_Energy_Ratio'] = X['MFCC_1']            / (X['RMS']              + 1e-9)
    X['Formant_Distance']   = X['F2']                -  X['F1']
    X['Pitch_Range_Norm']   = X['Pitch_Range']       / (X['MeanPitch']        + 1e-9)
    return X

X = engineer(X)[feature_columns]

# ── XGBoost predictions ───────────────────────────────────────────────────────
xgb_probs  = model.predict_proba(X)          # shape (N, 2)
xgb_truth  = xgb_probs[:, 0]                 # probability of Truth
xgb_lie    = xgb_probs[:, 1]
xgb_pred   = (xgb_lie >= 0.5).astype(int)
xgb_acc    = np.mean(xgb_pred == y_true) * 100

print(f"\n🌲 XGBoost Ensemble")
print(f"   Accuracy : {xgb_acc:.1f}%")
print(f"   Correct  : {np.sum(xgb_pred == y_true)} / {len(y_true)}")

# ── CNN predictions ───────────────────────────────────────────────────────────
cnn_available = False
try:
    import tensorflow as tf
    if os.path.exists("cnn_lie_model.h5"):
        cnn_model = tf.keras.models.load_model("cnn_lie_model.h5")

        # Build feature images same way as cnn_model.py --from_features
        from scipy.ndimage import uniform_filter

        def features_to_image(row, img_size=128):
            vec = row.astype(np.float32)
            vmin, vmax = vec.min(), vec.max()
            vec = (vec - vmin) / (vmax - vmin + 1e-9)
            n   = img_size * img_size
            vec = np.tile(vec, int(np.ceil(n / len(vec))))[:n].reshape(img_size, img_size)
            ch1 = np.cumsum(vec, axis=1) / np.arange(1, img_size+1)[None, :]
            ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-9)
            lm  = uniform_filter(vec, size=5)
            ch2 = np.sqrt(np.maximum(uniform_filter(vec**2, size=5) - lm**2, 0))
            ch2 = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-9)
            return np.stack([vec, ch1, ch2], axis=-1).astype(np.float32)

        # Use raw features (all columns before top-20 selection) for CNN images
        truth_raw = pd.read_excel("truth_features_full.xlsx").drop(columns=["Serial_No"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        lie_raw   = pd.read_excel("lie_features_full.xlsx").drop(columns=["Serial_No"],   errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        common    = sorted(set(truth_raw.columns) & set(lie_raw.columns))
        raw_vals  = pd.concat([truth_raw[common], lie_raw[common]], ignore_index=True).values

        X_img = np.array([features_to_image(r) for r in raw_vals], dtype=np.float32)
        cnn_probs = cnn_model.predict(X_img, verbose=0)
        cnn_truth = cnn_probs[:, 0]
        cnn_lie   = cnn_probs[:, 1]
        cnn_pred  = (cnn_lie >= 0.5).astype(int)
        cnn_acc   = np.mean(cnn_pred == y_true) * 100
        cnn_available = True

        print(f"\n🧠 CNN Model")
        print(f"   Accuracy : {cnn_acc:.1f}%")
        print(f"   Correct  : {np.sum(cnn_pred == y_true)} / {len(y_true)}")
    else:
        print("\n🧠 CNN Model  →  cnn_lie_model.h5 not found, skipping")
except Exception as e:
    print(f"\n🧠 CNN Model  →  skipped ({e})")

# ── Weighted ensemble ─────────────────────────────────────────────────────────
if cnn_available:
    ens_truth = XGB_WEIGHT * xgb_truth + CNN_WEIGHT * cnn_truth
    ens_lie   = XGB_WEIGHT * xgb_lie   + CNN_WEIGHT * cnn_lie
    ens_pred  = (ens_lie >= 0.5).astype(int)
    ens_acc   = np.mean(ens_pred == y_true) * 100

    print(f"\n⚡ Weighted Ensemble  (XGB {int(XGB_WEIGHT*100)}% + CNN {int(CNN_WEIGHT*100)}%)")
    print(f"   Accuracy : {ens_acc:.1f}%")
    print(f"   Correct  : {np.sum(ens_pred == y_true)} / {len(y_true)}")

# ── Per-class breakdown ───────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "-"*55)
if cnn_available:
    print("📊 Ensemble Classification Report:")
    print(classification_report(y_true, ens_pred, target_names=["Truth", "Lie"]))
    cm = confusion_matrix(y_true, ens_pred)
else:
    print("📊 XGBoost Classification Report:")
    print(classification_report(y_true, xgb_pred, target_names=["Truth", "Lie"]))
    cm = confusion_matrix(y_true, xgb_pred)

print("Confusion Matrix:")
print(f"              Predicted Truth  Predicted Lie")
print(f"Actual Truth       {cm[0][0]:<15}  {cm[0][1]}")
print(f"Actual Lie         {cm[1][0]:<15}  {cm[1][1]}")
print("="*55)