"""
app.py  —  Flask backend for Audio Lie Detection
=================================================
Runs XGBoost Ensemble + CNN in parallel.
Final verdict = weighted average (55% XGBoost + 45% CNN).
If cnn_lie_model.h5 is missing, falls back to XGBoost only.

LIVE AUDIO FIX:
  Browsers record in audio/webm (Chrome) or audio/ogg (Firefox) — never real WAV.
  predict_live() now converts whatever the browser sends → proper 22050 Hz mono WAV
  using ffmpeg before passing it to the detectors, preventing the empty DataFrame error.
"""

import os, uuid, warnings, subprocess, tempfile, logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from predict_audio import LieDetector
from cnn_predict   import CNNLieDetector
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app           = Flask(__name__)
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "webm"}

xgb_detector = LieDetector(
    model_path           = os.path.join(BASE_DIR, "rf_lie_model.pkl"),
    encoder_path         = os.path.join(BASE_DIR, "label_encoder.pkl"),
    feature_columns_path = os.path.join(BASE_DIR, "feature_columns.pkl"),
    upload_folder        = UPLOAD_FOLDER,
)
cnn_detector = CNNLieDetector(
    model_path    = os.path.join(BASE_DIR, "cnn_lie_model.h5"),
    upload_folder = UPLOAD_FOLDER,
)

XGB_WEIGHT = 0.55
CNN_WEIGHT = 0.45

# ── Target audio spec (must match your CNN / feature extractor settings) ──────
TARGET_SR       = 22050   # sample rate  (matches SR = 22050 in cnn_model.py)
TARGET_CHANNELS = 1       # mono


def convert_to_wav(input_path: str) -> str:
    """
    Convert ANY audio format (webm, ogg, mp3, m4a …) to a clean
    22050 Hz mono WAV using ffmpeg.

    Returns the path to the new .wav file (saved alongside the original).
    Raises RuntimeError if ffmpeg fails.

    Why this is needed:
      - Browsers record with MediaRecorder which outputs audio/webm (Chrome)
        or audio/ogg (Firefox).  Neither is a real WAV file even if you name
        the blob "live_recording.wav".
      - librosa can sometimes decode webm, but only if the system ffmpeg is
        present AND the codec is supported.  It fails silently on many setups,
        returning an empty array → "Feature extraction returned empty DataFrame".
      - By pre-converting here we guarantee the detectors always receive a
        proper PCM WAV regardless of what the browser sent.
    """
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite without asking
        "-i", input_path,            # input (any format)
        "-ar", str(TARGET_SR),       # resample to 22050 Hz
        "-ac", str(TARGET_CHANNELS), # mix down to mono
        "-sample_fmt", "s16",        # 16-bit PCM  (librosa default expectation)
        "-vn",                       # strip any video stream (some webm have it)
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-300:]}")

    log.info("Converted %s → %s", os.path.basename(input_path),
             os.path.basename(output_path))
    return output_path


def run_prediction(file_storage, convert_audio: bool = False):
    """
    Core prediction pipeline shared by /predict and /predict-live.

    Parameters
    ----------
    file_storage  : werkzeug FileStorage object from request.files["audio"]
    convert_audio : if True, run ffmpeg conversion before prediction.
                    Always True for live recordings (browser sends webm/ogg).
                    False for uploaded files (user already chose a proper format).
    """
    if file_storage.filename == "":
        return {"error": "No file selected."}, 400

    ext = (
        file_storage.filename.rsplit(".", 1)[-1].lower()
        if "." in file_storage.filename
        else "wav"
    )
    raw_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.{ext}")
    file_storage.save(raw_path)
    log.info("Saved upload → %s  (convert=%s)", os.path.basename(raw_path), convert_audio)

    # ── Convert live browser audio to proper WAV ───────────────────────────
    if convert_audio:
        try:
            filepath = convert_to_wav(raw_path)
        except RuntimeError as e:
            # ffmpeg unavailable or corrupt audio — tell the user clearly
            return {"error": f"Audio conversion failed: {e}"}, 500
        finally:
            # Remove the raw webm/ogg blob — we only need the converted WAV
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        filepath = raw_path

    # ── XGBoost prediction (always runs) ──────────────────────────────────
    xgb = xgb_detector.predict_audio(filepath)
    if "error" in xgb:
        return xgb, 500

    # ── CNN prediction (runs only if model is loaded) ─────────────────────
    cnn = cnn_detector.predict(filepath) if cnn_detector.available else {}

    xgb_truth = xgb.get("truth_probability", 0.5)
    xgb_lie   = xgb.get("lie_probability",   0.5)

    if cnn and "cnn_error" not in cnn:
        final_truth = XGB_WEIGHT * xgb_truth + CNN_WEIGHT * cnn.get("cnn_truth_probability", 0.5)
        final_lie   = XGB_WEIGHT * xgb_lie   + CNN_WEIGHT * cnn.get("cnn_lie_probability",   0.5)
        model_used  = "XGBoost + CNN Ensemble"
    else:
        final_truth = xgb_truth
        final_lie   = xgb_lie
        model_used  = "XGBoost Ensemble"

    resp = {
        "result":            "Lie" if final_lie > final_truth else "Truth",
        "truth_probability": round(final_truth, 4),
        "lie_probability":   round(final_lie,   4),
        "truth_percent":     round(final_truth * 100, 1),
        "lie_percent":       round(final_lie   * 100, 1),
        "model_used":        model_used,
        "xgb_truth":         round(xgb_truth * 100, 1),
        "xgb_lie":           round(xgb_lie   * 100, 1),
        "spectrograms":      xgb.get("spectrograms", {}),
    }
    if cnn and "cnn_error" not in cnn:
        resp["cnn_truth"] = round(cnn.get("cnn_truth_probability", 0) * 100, 1)
        resp["cnn_lie"]   = round(cnn.get("cnn_lie_probability",   0) * 100, 1)
    if "cnn_spectrogram" in cnn:
        resp["spectrograms"]["cnn_mel"] = cnn["cnn_spectrogram"]

    return resp, 200


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():       return render_template("index.html")

@app.route("/second")
def second():     return render_template("second.html")

@app.route("/live")
def live_audio(): return render_template("liveAudio.html")

@app.route("/result")
def result():     return render_template("result.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename): return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles uploaded audio files from the dashboard.
    User-uploaded files are already in a proper format (wav/mp3/m4a),
    so no conversion is needed.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio field in request."}), 400
    d, s = run_prediction(request.files["audio"], convert_audio=False)
    return jsonify(d), s


@app.route("/predict-live", methods=["POST"])
def predict_live():
    """
    Handles live browser recordings from liveAudio.html.

    THE FIX: convert_audio=True
    ----------------------------
    The browser's MediaRecorder API records in audio/webm (Chrome) or
    audio/ogg (Firefox), even when you name the blob "live_recording.wav".
    This caused librosa to receive corrupt/unreadable bytes, silently return
    an empty array, and trigger:
        ⚠ Feature extraction returned empty DataFrame.

    Setting convert_audio=True runs ffmpeg to rewrite the file as a genuine
    22050 Hz mono PCM WAV before any feature extraction happens.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio received."}), 400
    d, s = run_prediction(request.files["audio"], convert_audio=True)
    return jsonify(d), s


@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "xgb_loaded": xgb_detector.model is not None,
        "cnn_loaded": cnn_detector.available,
        "ffmpeg":     subprocess.run(
                          ["ffmpeg", "-version"], capture_output=True
                      ).returncode == 0,   # tells you if ffmpeg is installed
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)