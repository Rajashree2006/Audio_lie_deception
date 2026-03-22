"""
predict_audio.py
================
LieDetector class — loads trained model + encoder,
extracts features, applies feature engineering (mirrors model.py),
generates spectrograms, and returns structured predictions.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from extract_features import extract_features, generate_spectrograms, FEATURE_NAMES

warnings.filterwarnings("ignore")

_DROP_COLS = {"Serial_No", "Renamed_File"}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the engineer_features() function in model.py exactly."""
    X = df.copy()
    X['Pitch_Var_Ratio']    = X['PitchSD']           / (X['MeanPitch']         + 1e-9)
    X['Jitter_Shimmer_Int'] = X['Jitter']            *  X['Shimmer']
    X['HNR_Jitter_Ratio']   = X['HNR']               / (X['Jitter']            + 1e-9)
    X['Spectral_Spread']    = X['SpectralBandwidth'] / (X['SpectralCentroid']   + 1e-9)
    X['MFCC1_Energy_Ratio'] = X['MFCC_1']            / (X['RMS']               + 1e-9)
    X['Formant_Distance']   = X['F2']                -  X['F1']
    X['Pitch_Range_Norm']   = X['Pitch_Range']       / (X['MeanPitch']         + 1e-9)
    return X


class LieDetector:
    def __init__(
        self,
        model_path="rf_lie_model.pkl",
        encoder_path="label_encoder.pkl",
        feature_columns_path="feature_columns.pkl",
        upload_folder="uploads",
    ):
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

        # Load label encoder
        self.label_encoder = None
        if encoder_path and os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)

        # Feature list — priority:
        # 1. feature_columns.pkl  2. model.feature_names_in_  3. FEATURE_NAMES fallback
        self.trained_features = None

        if os.path.exists(feature_columns_path):
            self.trained_features = joblib.load(feature_columns_path)
            print(f"[LieDetector] Loaded {len(self.trained_features)} features from feature_columns.pkl")

        elif hasattr(self.model, "feature_names_in_"):
            self.trained_features = list(self.model.feature_names_in_)
            print(f"[LieDetector] Loaded {len(self.trained_features)} features from model.feature_names_in_")

        else:
            self.trained_features = [c for c in FEATURE_NAMES if c not in _DROP_COLS]
            n_model = getattr(self.model, "n_features_in_", "?")
            print(f"[LieDetector] WARNING: using {len(self.trained_features)} fallback features "
                  f"(model expects {n_model})")

        self.upload_folder = upload_folder

    def predict_audio(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}

        try:
            # 1. Extract features
            features_df = extract_features(audio_path)
            if features_df is None or features_df.empty:
                return {"error": "Feature extraction returned empty DataFrame."}

            # 2. Drop metadata columns
            features_df = features_df.drop(columns=list(_DROP_COLS), errors="ignore")

            # 3. Apply feature engineering (mirrors model.py)
            features_df = _engineer_features(features_df)

            # 4. Align to trained feature set
            if self.trained_features:
                for col in self.trained_features:
                    if col not in features_df.columns:
                        print(f"[LieDetector] Missing feature '{col}' — filling with 0.0")
                        features_df[col] = 0.0
                features_df = features_df[self.trained_features]

            # 5. Shape check
            n_expected = getattr(self.model, "n_features_in_", len(self.trained_features or []))
            if features_df.shape[1] != n_expected:
                return {
                    "error": (
                        f"Feature mismatch: model expects {n_expected} features "
                        f"but got {features_df.shape[1]}. "
                        f"Re-run model.py to retrain and regenerate feature_columns.pkl."
                    )
                }

            # 6. Predict
            pred_encoded  = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]

            # 7. Decode label
            if self.label_encoder:
                class_labels = list(self.label_encoder.classes_)
                label = str(self.label_encoder.inverse_transform([pred_encoded])[0])
            else:
                class_labels = [str(c) for c in self.model.classes_]
                label = str(pred_encoded)

            # 8. Map probabilities
            # LabelEncoder.fit() sorts alphabetically regardless of fit order:
            # 'Lie' < 'Truth' alphabetically → index 0 = Lie, index 1 = Truth
            response = {"result": label, "truth_probability": 0.0, "lie_probability": 0.0}
            for i, cls in enumerate(class_labels):
                cls_lower = str(cls).lower()
                if cls_lower in ("truth", "true"):
                    response["truth_probability"] = round(float(probabilities[i]), 4)
                elif cls_lower in ("lie", "deception", "false"):
                    response["lie_probability"] = round(float(probabilities[i]), 4)

            # Fallback if no class matched by name
            if response["truth_probability"] == 0.0 and response["lie_probability"] == 0.0:
                response["lie_probability"]   = round(float(probabilities[0]), 4)
                response["truth_probability"] = round(float(probabilities[min(1, len(probabilities)-1)]), 4)

            # Sanity check — final result must match the higher probability
            if response["truth_probability"] >= response["lie_probability"]:
                response["result"] = "Truth"
            else:
                response["result"] = "Lie"

            # 9. Generate spectrograms
            try:
                spec_urls = generate_spectrograms(audio_path, self.upload_folder)
                response["spectrograms"] = {} if "error" in spec_urls else spec_urls
            except Exception as se:
                print(f"[LieDetector] Spectrogram error: {se}")
                response["spectrograms"] = {}

            return response

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"error": str(exc)}