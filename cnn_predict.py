"""
cnn_predict.py
==============
CNNLieDetector — loads the trained CNN (cnn_lie_model.h5) and predicts
Truth / Lie from a raw audio file by converting it to a Mel spectrogram image.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid as _uuid

warnings.filterwarnings("ignore")

# MUST match training
IMG_SIZE   = 224
N_MELS     = 128
HOP_LENGTH = 512
SR         = 22050
DURATION   = 15.0   # ✅ FIXED


def _audio_to_image(audio_path: str) -> np.ndarray | None:
    try:
        import librosa
        import librosa.display
        import tensorflow as tf

        # ✅ FIXED duration
        y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION)

        # ✅ NORMALIZATION (VERY IMPORTANT)
        y = y / (np.max(np.abs(y)) + 1e-9)

        if len(y) < sr * 0.3:
            return None

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(2, 2), dpi=IMG_SIZE // 2)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        librosa.display.specshow(
            mel_db, sr=sr, hop_length=HOP_LENGTH,
            x_axis=None, y_axis=None,
            ax=ax, cmap="magma"
        )

        ax.set_axis_off()
        plt.tight_layout(pad=0)
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        buf  = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = buf.reshape(h, w, 3).astype(np.float32)
        plt.close(fig)

        # Resize to model input
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]).numpy().astype(np.float32)

        # ✅ SCALE IMAGE (IMPORTANT FOR CNN)
        img = img / 255.0

        return img

    except Exception as e:
        print(f"[CNNLieDetector] Image conversion failed: {e}")
        return None


def _save_cnn_spectrogram(audio_path: str, out_dir: str) -> str | None:
    try:
        import librosa
        import librosa.display

        # ✅ MATCH duration here too
        y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=DURATION)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0d1826")
        ax.set_facecolor("#050d1f")

        img = librosa.display.specshow(
            mel_db, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="mel",
            ax=ax, cmap="magma"
        )

        cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cb.ax.tick_params(colors="#6a88c0", labelsize=7)

        ax.set_title("CNN Input — Mel Spectrogram", color="#c5d8ff",
                     fontsize=11, fontweight="bold", pad=8)

        ax.set_xlabel("Time (s)", color="#6a88c0", fontsize=8)
        ax.set_ylabel("Mel Frequency", color="#6a88c0", fontsize=8)

        ax.tick_params(colors="#6a88c0", labelsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3460")

        uid      = _uuid.uuid4().hex[:10]
        filename = f"{uid}_cnn_mel.png"
        path     = os.path.join(out_dir, filename)

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1826")
        plt.close(fig)

        return f"/uploads/{filename}"

    except Exception as e:
        print(f"[CNNLieDetector] Spectrogram save failed: {e}")
        return None


class CNNLieDetector:

    def __init__(
        self,
        model_path: str    = "cnn_lie_model.h5",
        upload_folder: str = "uploads",
    ):
        self.model         = None
        self.upload_folder = upload_folder

        try:
            import tensorflow as tf
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"[CNNLieDetector] Model loaded from {model_path}")
            else:
                print(f"[CNNLieDetector] Model not found at {model_path} — CNN disabled")
        except Exception as e:
            print(f"[CNNLieDetector] Could not load model: {e}")

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, audio_path: str) -> dict:
        if not self.available:
            return {"cnn_error": "CNN model not loaded"}

        img = _audio_to_image(audio_path)
        if img is None:
            return {"cnn_error": "Could not convert audio"}

        try:
            # Add batch dimension
            batch = np.expand_dims(img, axis=0)

            probs = self.model.predict(batch, verbose=0)[0]

            truth_prob = float(probs[0])
            lie_prob   = float(probs[1])

            # ✅ DEBUG (VERY IMPORTANT)
            print(f"[CNN DEBUG] Truth={truth_prob:.4f}, Lie={lie_prob:.4f}")

            # ✅ IMPROVED DECISION LOGIC
            if lie_prob > 0.40:
                label = "Lie"
            else:
                label = "Truth"

            result = {
                "cnn_result": label,
                "cnn_truth_probability": round(truth_prob, 4),
                "cnn_lie_probability":   round(lie_prob,   4),
            }

            spec_url = _save_cnn_spectrogram(audio_path, self.upload_folder)
            if spec_url:
                result["cnn_spectrogram"] = spec_url

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"cnn_error": str(e)}