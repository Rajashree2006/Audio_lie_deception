"""
extract_features.py
====================
Two modes:
  1. Run directly  → batch-extracts features from a folder into an Excel file
                     (original behaviour, unchanged)
  2. Imported      → extract_features(path)      returns a 1-row DataFrame
                     generate_spectrograms(path)  returns dict of image URLs
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import parselmouth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from parselmouth.praat import call
from scipy.stats import skew
import uuid as _uuid
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAMES  (used by predict_audio.py as fallback)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = (
    ["Serial_No", "MeanPitch", "MinPitch", "MaxPitch", "PitchSD",
     "Pitch_Range", "Pitch_Skewness", "MeanIntensity", "IntensitySD",
     "Energy", "Jitter", "Jitter_RAP", "Jitter_PPQ5",
     "Shimmer", "Shimmer_APQ3", "Shimmer_APQ5",
     "HNR", "HNR_std", "HNR_min", "HNR_max",
     "F1", "F2", "F3", "F2_F1", "F3_F2",
     "SpectralCentroid", "SpectralBandwidth", "SpectralRolloff",
     "RMS", "VoicedRatio"]
    + [f"MFCC_{i}"       for i in range(1, 14)]
    + [f"MFCC_delta_{i}" for i in range(1, 14)]
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION  (one file → dict)  — same logic as original
# ─────────────────────────────────────────────────────────────────────────────

def _extract_row(path: str) -> dict:
    y, sr = librosa.load(path, sr=None)
    sound = parselmouth.Sound(path)

    # Pitch
    pitch        = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    if len(pitch_values) == 0:
        pitch_values = np.array([0.0])

    mean_pitch  = float(np.mean(pitch_values))
    min_pitch   = float(np.min(pitch_values))
    max_pitch   = float(np.max(pitch_values))
    pitch_sd    = float(np.std(pitch_values))
    pitch_range = max_pitch - min_pitch
    pitch_skew  = float(skew(pitch_values))

    # Intensity
    intensity_values = sound.to_intensity().values[0]
    mean_intensity   = float(np.mean(intensity_values))
    intensity_sd     = float(np.std(intensity_values))

    # Energy
    energy = float(np.sum(y ** 2))

    # Jitter / Shimmer
    pp           = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter       = call(pp,          "Get jitter (local)",  0, 0, 0.0001, 0.02, 1.3)
    jitter_rap   = call(pp,          "Get jitter (rap)",    0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5  = call(pp,          "Get jitter (ppq5)",   0, 0, 0.0001, 0.02, 1.3)
    shimmer      = call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, pp], "Get shimmer (apq3)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, pp], "Get shimmer (apq5)",  0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_vals    = harmonicity.values[harmonicity.values != -200]
    if len(hnr_vals) == 0:
        hnr_vals = np.array([0.0])
    hnr     = float(np.mean(hnr_vals))
    hnr_std = float(np.std(hnr_vals))
    hnr_min = float(np.min(hnr_vals))
    hnr_max = float(np.max(hnr_vals))

    # Formants
    formant = call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
    f1      = float(call(formant, "Get mean", 1, 0, 0, "Hertz"))
    f2      = float(call(formant, "Get mean", 2, 0, 0, "Hertz"))
    f3      = float(call(formant, "Get mean", 3, 0, 0, "Hertz"))
    f2_f1   = f2 - f1
    f3_f2   = f3 - f2

    # Spectral
    spectral_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y,  sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y,   sr=sr)))
    rms                = float(np.mean(librosa.feature.rms(y=y)))

    # Voiced ratio
    total_frames = len(pitch.selected_array['frequency'])
    voiced_ratio = len(pitch_values) / total_frames if total_frames > 0 else 0.0

    # MFCC
    mfcc            = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean       = np.mean(mfcc, axis=1)
    mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)

    row = {
        "Serial_No":         os.path.basename(path),
        "MeanPitch":         mean_pitch,   "MinPitch":      min_pitch,
        "MaxPitch":          max_pitch,    "PitchSD":       pitch_sd,
        "Pitch_Range":       pitch_range,  "Pitch_Skewness":pitch_skew,
        "MeanIntensity":     mean_intensity, "IntensitySD": intensity_sd,
        "Energy":            energy,
        "Jitter":            jitter,       "Jitter_RAP":   jitter_rap,
        "Jitter_PPQ5":       jitter_ppq5,
        "Shimmer":           shimmer,      "Shimmer_APQ3": shimmer_apq3,
        "Shimmer_APQ5":      shimmer_apq5,
        "HNR":               hnr,          "HNR_std":      hnr_std,
        "HNR_min":           hnr_min,      "HNR_max":      hnr_max,
        "F1":                f1,           "F2":           f2,
        "F3":                f3,           "F2_F1":        f2_f1,
        "F3_F2":             f3_f2,
        "SpectralCentroid":  spectral_centroid,
        "SpectralBandwidth": spectral_bandwidth,
        "SpectralRolloff":   spectral_rolloff,
        "RMS":               rms,
        "VoicedRatio":       voiced_ratio,
    }
    for i in range(13):
        row[f"MFCC_{i+1}"]       = float(mfcc_mean[i])
        row[f"MFCC_delta_{i+1}"] = float(mfcc_delta_mean[i])

    return row


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (imported by predict_audio.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(audio_path: str) -> pd.DataFrame:
    """Single-file feature extraction → 1-row DataFrame."""
    try:
        return pd.DataFrame([_extract_row(audio_path)])
    except Exception:
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SPECTROGRAM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_CMAP             = "magma"
_DPI              = 150
_FIGSIZE_SINGLE   = (8, 3)
_FIGSIZE_COMBINED = (12, 8)


def _style_ax(ax, title):
    ax.set_title(title, color="#c5d8ff", fontsize=10, pad=6, fontweight="bold")
    ax.tick_params(colors="#6a88c0", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#1e3460")
    ax.set_facecolor("#050d1f")


def _save(fig, path):
    fig.savefig(path, dpi=_DPI, bbox_inches="tight",
                facecolor="#0d1826", edgecolor="none")
    plt.close(fig)


def generate_spectrograms(audio_path: str, out_dir: str) -> dict:
    """
    Generate 5 dark-themed spectrogram PNGs for *audio_path*.
    Returns dict: { "combined", "waveform", "stft", "mel", "mfcc" } → URL strings.
    """
    os.makedirs(out_dir, exist_ok=True)
    uid = _uuid.uuid4().hex[:10]

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return {"error": str(e)}

    times  = np.linspace(0, len(y) / sr, num=len(y))
    D_db   = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    mel_db = librosa.power_to_db(
                 librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    urls   = {}

    # Waveform
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE, facecolor="#0d1826")
    ax.plot(times, y, color="#2d6cdf", linewidth=0.7, alpha=0.9)
    _style_ax(ax, "Waveform  (amplitude vs time)")
    ax.set_xlabel("Time (s)", color="#6a88c0", fontsize=8)
    ax.set_ylabel("Amplitude", color="#6a88c0", fontsize=8)
    p = os.path.join(out_dir, f"{uid}_waveform.png"); _save(fig, p)
    urls["waveform"] = f"/uploads/{os.path.basename(p)}"

    # STFT
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE, facecolor="#0d1826")
    im = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="hz",
                                  ax=ax, cmap=_CMAP)
    fig.colorbar(im, ax=ax, format="%+2.0f dB").ax.tick_params(colors="#6a88c0", labelsize=7)
    _style_ax(ax, "STFT Spectrogram  (frequency vs time, dB)")
    ax.set_xlabel("Time (s)", color="#6a88c0", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", color="#6a88c0", fontsize=8)
    p = os.path.join(out_dir, f"{uid}_stft.png"); _save(fig, p)
    urls["stft"] = f"/uploads/{os.path.basename(p)}"

    # Mel
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE, facecolor="#0d1826")
    im = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                  ax=ax, cmap=_CMAP)
    fig.colorbar(im, ax=ax, format="%+2.0f dB").ax.tick_params(colors="#6a88c0", labelsize=7)
    _style_ax(ax, "Mel Spectrogram  (perceptual scale, dB)")
    ax.set_xlabel("Time (s)", color="#6a88c0", fontsize=8)
    ax.set_ylabel("Mel Frequency", color="#6a88c0", fontsize=8)
    p = os.path.join(out_dir, f"{uid}_mel.png"); _save(fig, p)
    urls["mel"] = f"/uploads/{os.path.basename(p)}"

    # MFCC
    fig, ax = plt.subplots(figsize=_FIGSIZE_SINGLE, facecolor="#0d1826")
    im = librosa.display.specshow(mfcc, sr=sr, x_axis="time",
                                  ax=ax, cmap="coolwarm")
    fig.colorbar(im, ax=ax).ax.tick_params(colors="#6a88c0", labelsize=7)
    _style_ax(ax, "MFCC  (first 20 coefficients)")
    ax.set_xlabel("Time (s)", color="#6a88c0", fontsize=8)
    ax.set_ylabel("MFCC coefficient", color="#6a88c0", fontsize=8)
    p = os.path.join(out_dir, f"{uid}_mfcc.png"); _save(fig, p)
    urls["mfcc"] = f"/uploads/{os.path.basename(p)}"

    # Combined 2×2
    fig = plt.figure(figsize=_FIGSIZE_COMBINED, facecolor="#0d1826")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.set_facecolor("#050d1f")
    ax0.plot(times, y, color="#2d6cdf", linewidth=0.6)
    _style_ax(ax0, "Waveform"); ax0.set_xlabel("Time (s)", color="#6a88c0", fontsize=7)

    ax1 = fig.add_subplot(gs[0, 1]); ax1.set_facecolor("#050d1f")
    i1 = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="hz",
                                   ax=ax1, cmap=_CMAP)
    fig.colorbar(i1, ax=ax1, format="%+2.0f dB", pad=0.02).ax.tick_params(colors="#6a88c0", labelsize=6)
    _style_ax(ax1, "STFT")

    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor("#050d1f")
    i2 = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                   ax=ax2, cmap=_CMAP)
    fig.colorbar(i2, ax=ax2, format="%+2.0f dB", pad=0.02).ax.tick_params(colors="#6a88c0", labelsize=6)
    _style_ax(ax2, "Mel Spectrogram")

    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor("#050d1f")
    i3 = librosa.display.specshow(mfcc, sr=sr, x_axis="time",
                                   ax=ax3, cmap="coolwarm")
    fig.colorbar(i3, ax=ax3, pad=0.02).ax.tick_params(colors="#6a88c0", labelsize=6)
    _style_ax(ax3, "MFCC")

    fig.suptitle("Audio Analysis — CNN Feature Maps", color="#7aa2ff",
                 fontsize=13, fontweight="bold", y=1.01)
    p = os.path.join(out_dir, f"{uid}_combined.png"); _save(fig, p)
    urls["combined"] = f"/uploads/{os.path.basename(p)}"

    return urls


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE  (python extract_features.py)  — original behaviour unchanged
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    audio_folder = r"D:\j2ee\test\truth_audio"
    output_file  = r"D:\j2ee\test\truth_features_full.xlsx"

    rows = []
    for file in os.listdir(audio_folder):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(audio_folder, file)
        try:
            rows.append(_extract_row(path))
            print("Processed:", file)
        except Exception as e:
            print("Error:", file, e)

    pd.DataFrame(rows).to_excel(output_file, index=False)
    print("Feature extraction completed. Saved to:", output_file)