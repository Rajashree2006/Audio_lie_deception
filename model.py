# -*- coding: utf-8 -*-
"""
model.py  —  LIE DETECTOR TRAINING PIPELINE
============================================
Saves THREE files to outputs/ that Flask needs:
  outputs/rf_lie_model.pkl       ← trained model
  outputs/label_encoder.pkl      ← LabelEncoder (Truth=0, Lie=1)
  outputs/feature_columns.pkl    ← exact top-20 feature list  ← NEW

Copy all three next to app.py before starting Flask.
"""

import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble        import HistGradientBoostingClassifier, VotingClassifier
from xgboost                 import XGBClassifier

os.makedirs("outputs", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
truth = pd.read_excel('truth_features_full.xlsx')
lie   = pd.read_excel('lie_features_full.xlsx')
truth['label'] = 0
lie['label']   = 1

df = pd.concat([truth, lie], ignore_index=True)
df = df.drop(columns=['Serial_No'], errors='ignore')

X = df.drop(columns=['label'])
y = df['label']
print("Dataset shape:", df.shape)

# ── Save LabelEncoder ─────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(["Truth", "Lie"])          # index 0 = Truth, index 1 = Lie
joblib.dump(le, "outputs/label_encoder.pkl")
print("label_encoder.pkl saved")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(data):
    X_new = data.copy()
    X_new['Pitch_Var_Ratio']    = X_new['PitchSD']           / (X_new['MeanPitch']         + 1e-9)
    X_new['Jitter_Shimmer_Int'] = X_new['Jitter']            *  X_new['Shimmer']
    X_new['HNR_Jitter_Ratio']   = X_new['HNR']               / (X_new['Jitter']            + 1e-9)
    X_new['Spectral_Spread']    = X_new['SpectralBandwidth'] / (X_new['SpectralCentroid']   + 1e-9)
    X_new['MFCC1_Energy_Ratio'] = X_new['MFCC_1']            / (X_new['RMS']               + 1e-9)
    X_new['Formant_Distance']   = X_new['F2']                -  X_new['F1']
    X_new['Pitch_Range_Norm']   = X_new['Pitch_Range']       / (X_new['MeanPitch']         + 1e-9)
    return X_new

X_train = engineer_features(X_train)
X_test  = engineer_features(X_test)
X       = engineer_features(X)
print("Features after engineering:", X_train.shape[1])

# ── Feature selection via XGBoost ─────────────────────────────────────────────
temp_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          eval_metric='logloss', random_state=42)
temp_xgb.fit(X_train, y_train)

importances  = pd.Series(temp_xgb.feature_importances_, index=X_train.columns)
TOP_K        = 20
top_features = list(importances.nlargest(TOP_K).index)

X_train = X_train[top_features]
X_test  = X_test[top_features]
print("Top Features:", top_features)

# ── Save feature_columns.pkl  ◄── CRITICAL for predict_audio.py ──────────────
joblib.dump(top_features, "outputs/feature_columns.pkl")
print("feature_columns.pkl saved")

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# ── Models ────────────────────────────────────────────────────────────────────
hgb_model = Pipeline([('s', StandardScaler()),
                       ('clf', HistGradientBoostingClassifier(random_state=42))])

xgb_model = Pipeline([('s', StandardScaler()),
                       ('clf', XGBClassifier(n_estimators=300, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8,
                                             eval_metric='logloss', random_state=42))])

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(name, model):
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {"name": name, "model": model,
            "cv_auc": cv_auc,
            "acc": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, prob),
            "pred": pred}

results = []
for name, model in [("HistGradientBoost", hgb_model), ("XGBoost", xgb_model)]:
    r = evaluate(name, model)
    results.append(r)
    print(f"{name} | CV AUC:{r['cv_auc']:.3f} | Test Acc:{r['acc']:.3f} | Test AUC:{r['auc']:.3f}")

# ── Voting ensemble ───────────────────────────────────────────────────────────
ensemble = VotingClassifier(
    estimators=[('hgb', HistGradientBoostingClassifier(random_state=42)),
                ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))],
    voting='soft'
)
r = evaluate("Voting Ensemble", ensemble)
results.append(r)
print(f"Voting Ensemble | CV AUC:{r['cv_auc']:.3f} | Test Acc:{r['acc']:.3f} | Test AUC:{r['auc']:.3f}")

# ── Best model ────────────────────────────────────────────────────────────────
best = max(results, key=lambda x: x['auc'])
print("\nBEST MODEL:", best['name'])
print(classification_report(y_test, best['pred'], target_names=['Truth', 'Lie']))

joblib.dump(best['model'], "outputs/rf_lie_model.pkl")
print("\n✅ Saved to outputs/:")
print("   rf_lie_model.pkl")
print("   label_encoder.pkl")
print("   feature_columns.pkl")
print("\nCopy these three files next to app.py, then run:  python app.py")