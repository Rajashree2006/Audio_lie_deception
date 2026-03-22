"""
cnn_model.py  —  IMPROVED CNN Lie Detector
============================================
Improvements over original:
  1. Multi-channel input: Mel + MFCC + Delta-MFCC (3 real channels)
  2. SpecAugment: random time/freq masking during training
  3. Residual blocks (ResNet-style skip connections)
  4. Channel Attention (squeeze-and-excitation)
  5. Cosine LR decay schedule (replaces ReduceLROnPlateau)
  6. Label smoothing (reduces overconfidence)
  7. Class-weight balancing (handles imbalanced datasets)
  8. Larger image size (224x224) for better frequency resolution
  9. Longer audio clip (15s) to capture more speech patterns
 10. Cleaner augmentation (no rotation — spectrograms aren't rotation-invariant)

Usage:
    # Train from audio files (RECOMMENDED):
    python cnn_model.py --truth_dir path/to/truth_wavs --lie_dir path/to/lie_wavs

    # Fallback: train from feature spreadsheets:
    python cnn_model.py --from_features

    # Custom paths:
    python cnn_model.py --truth_dir truth_audio --lie_dir lie_audio --epochs 80 --img_size 224
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMG_SIZE    = 224        # ↑ from 128 — more frequency resolution
N_MELS      = 128
N_MFCC      = 40        # NEW: more MFCC coefficients
HOP_LENGTH  = 512
SR          = 22050
DURATION    = 15.0      # ↑ from 10s — capture more speech
BATCH_SIZE  = 16        # ↑ from 8 — more stable gradients
EPOCHS      = 80        # ↑ from 60
WARMUP_EPOCHS = 5       # NEW: cosine schedule warmup
TEST_SPLIT  = 0.20
RANDOM_SEED = 42
OUTPUT_DIR  = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ── Dependencies ───────────────────────────────────────────────────────────────
try:
    import librosa, librosa.display
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[WARN] librosa not found — falling back to feature-based mode")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    print(f"[INFO] TensorFlow {tf.__version__} detected")
    print(f"[INFO] GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
except ImportError:
    print("[ERROR] TensorFlow not found.  Run:  pip install tensorflow")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1: Multi-channel spectrogram (Mel + MFCC + Delta)
# ══════════════════════════════════════════════════════════════════════════════

def audio_to_multichannel(path: str, img_size: int = IMG_SIZE) -> np.ndarray | None:
    """
    Converts one audio file → (img_size, img_size, 3) float32 array.

    Channel 0: Mel spectrogram (perceptual frequency content)
    Channel 1: MFCC map      (vocal tract shape / timbre)
    Channel 2: Delta-MFCC    (rate of change — catches deception-related micro-tremors)

    Each channel is independently normalised to [0, 255] so the CNN sees
    genuinely different information in every channel, unlike a plain RGB render.
    """
    try:
        y, sr = librosa.load(path, sr=SR, mono=True, duration=DURATION)
        if len(y) < sr * 0.5:
            print(f"  [SKIP] {os.path.basename(path)} — too short")
            return None

        # ── Channel 0: Mel spectrogram ────────────────────────────────────
        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # ── Channel 1: MFCC (mean across coefficients → 2-D time map) ────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     hop_length=HOP_LENGTH)

        # ── Channel 2: Delta-MFCC (first derivative) ──────────────────────
        delta_mfcc = librosa.feature.delta(mfcc)

        def _to_uint8_image(arr2d: np.ndarray) -> np.ndarray:
            """Normalise a 2-D array to uint8 and resize to img_size × img_size."""
            arr = arr2d.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255.0
            # Resize via tensorflow
            arr_3d = arr[..., np.newaxis]          # (H, W, 1)
            arr_3d = tf.image.resize(arr_3d, [img_size, img_size]).numpy()
            return arr_3d[..., 0].astype(np.float32)  # back to (H, W)

        ch0 = _to_uint8_image(mel_db)
        ch1 = _to_uint8_image(mfcc)
        ch2 = _to_uint8_image(delta_mfcc)

        return np.stack([ch0, ch1, ch2], axis=-1)   # (H, W, 3)

    except Exception as e:
        print(f"  [WARN] {os.path.basename(path)}: {e}")
        return None


def load_audio_dataset(truth_dir: str, lie_dir: str,
                       img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for label, folder in [(0, truth_dir), (1, lie_dir)]:
        tag = "Truth" if label == 0 else "Lie"
        if not os.path.isdir(folder):
            print(f"[ERROR] Folder not found: {folder}")
            continue
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac")))
        print(f"  {tag}: {len(files)} files in {folder}")
        ok = 0
        for fn in files:
            arr = audio_to_multichannel(os.path.join(folder, fn), img_size)
            if arr is not None:
                X.append(arr)
                y.append(label)
                ok += 1
        print(f"  {tag}: {ok}/{len(files)} loaded successfully")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Feature-spreadsheet fallback (unchanged logic, cleaner multi-channel build)
# ══════════════════════════════════════════════════════════════════════════════

def features_to_image(row: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    from scipy.ndimage import uniform_filter
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


def load_feature_dataset(truth_xlsx: str, lie_xlsx: str,
                         img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    truth_df = pd.read_excel(truth_xlsx)
    lie_df   = pd.read_excel(lie_xlsx)
    drop = {"Serial_No", "Renamed_File", "label"}
    for df in [truth_df, lie_df]:
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)
    truth_df = truth_df.select_dtypes(include=[np.number]).fillna(0)
    lie_df   = lie_df.select_dtypes(include=[np.number]).fillna(0)
    common   = sorted(set(truth_df.columns) & set(lie_df.columns))
    print(f"  Feature columns: {len(common)}, Truth: {len(truth_df)}, Lie: {len(lie_df)}")
    X = [features_to_image(r, img_size) for r in truth_df[common].values] + \
        [features_to_image(r, img_size) for r in lie_df[common].values]
    y = [0]*len(truth_df) + [1]*len(lie_df)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2: SpecAugment — mask random time & frequency bands
# ══════════════════════════════════════════════════════════════════════════════

def spec_augment(img: tf.Tensor,
                 freq_mask_param: int = 20,
                 time_mask_param: int = 30,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2) -> tf.Tensor:
    """
    Applies SpecAugment to a single (H, W, C) spectrogram tensor.
    Randomly zeroes out horizontal (frequency) and vertical (time) strips.
    This prevents the CNN from over-relying on specific frequency or time regions.
    """
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]

    for _ in range(n_freq_masks):
        f  = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.maximum(1, h - f), dtype=tf.int32)
        mask = tf.concat([
            tf.ones([f0, tf.shape(img)[1], tf.shape(img)[2]]),
            tf.zeros([f,  tf.shape(img)[1], tf.shape(img)[2]]),
            tf.ones([tf.maximum(0, h - f0 - f), tf.shape(img)[1], tf.shape(img)[2]])
        ], axis=0)
        img = img * tf.cast(mask[:h, :, :], img.dtype)

    for _ in range(n_time_masks):
        t  = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.maximum(1, w - t), dtype=tf.int32)
        mask = tf.concat([
            tf.ones([tf.shape(img)[0], t0, tf.shape(img)[2]]),
            tf.zeros([tf.shape(img)[0], t,  tf.shape(img)[2]]),
            tf.ones([tf.shape(img)[0], tf.maximum(0, w - t0 - t), tf.shape(img)[2]])
        ], axis=1)
        img = img * tf.cast(mask[:, :w, :], img.dtype)

    return img


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3 & 4: Residual blocks + Channel Attention (SE block)
# ══════════════════════════════════════════════════════════════════════════════

def residual_block(x: tf.Tensor, filters: int,
                   reg: regularizers.Regularizer) -> tf.Tensor:
    """
    ResNet-style residual block.
    The skip connection lets gradients flow directly from deep to shallow layers,
    fixing vanishing gradients and allowing the model to learn residuals only.

        input ──→ Conv → BN → ReLU → Conv → BN ──→ + → ReLU
              └────────────────── (1×1 proj if needed) ──┘
    """
    shortcut = x

    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)

    # Project shortcut if channel count changed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same",
                                 kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def se_block(x: tf.Tensor, ratio: int = 8) -> tf.Tensor:
    """
    Squeeze-and-Excitation (channel attention) block.
    Learns WHICH feature maps (channels) matter most and rescales them.
    Particularly useful for spectrograms where some frequency bands are far
    more diagnostic of deception than others.

        x → GlobalAvgPool → Dense(C/r) → ReLU → Dense(C) → Sigmoid → scale x
    """
    c = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(1, c // ratio), activation="relu")(se)
    se = layers.Dense(c, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, c))(se)
    return layers.Multiply()([x, se])


# ══════════════════════════════════════════════════════════════════════════════
# Full improved CNN architecture
# ══════════════════════════════════════════════════════════════════════════════

def build_improved_cnn(input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3)) -> keras.Model:
    """
    Improved architecture summary:
      - Input augmentation: flip + brightness + contrast (no rotation!)
      - SpecAugment layer
      - 4 residual blocks with SE attention, increasing filters: 32→64→128→256
      - GlobalAveragePooling + dense head with dropout
      - Output: softmax over 2 classes (Truth, Lie)

    Why no rotation? Rotating a spectrogram is meaningless — time is the x-axis
    and frequency is the y-axis. Rotating mixes them up and destroys the signal.
    """
    reg = regularizers.l2(1e-4)
    inp = keras.Input(shape=input_shape, name="spectrogram_input")

    # ── Normalise to [-1, 1] ───────────────────────────────────────────────
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)

    # ── Standard image augmentation (training only) ────────────────────────
    # NOTE: No RandomRotation — rotation is not meaningful for spectrograms
    x = layers.RandomFlip("horizontal")(x)           # time-mirror is valid
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # ── IMPROVEMENT 2: SpecAugment ─────────────────────────────────────────
    x = layers.Lambda(
        lambda t: tf.map_fn(spec_augment, t),
        name="spec_augment"
    )(x)

    # ── Block 1: 32 filters ────────────────────────────────────────────────
    x = residual_block(x, 32, reg)
    x = se_block(x)                                  # channel attention
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.20)(x)

    # ── Block 2: 64 filters ────────────────────────────────────────────────
    x = residual_block(x, 64, reg)
    x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # ── Block 3: 128 filters ───────────────────────────────────────────────
    x = residual_block(x, 128, reg)
    x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # ── Block 4: 256 filters ───────────────────────────────────────────────
    x = residual_block(x, 256, reg)
    x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # ── Classification head ────────────────────────────────────────────────
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.50)(x)
    x   = layers.Dense(64,  activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.30)(x)
    out = layers.Dense(2, activation="softmax", name="output")(x)

    return keras.Model(inp, out, name="LieDetector_CNN_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(X: np.ndarray, y: np.ndarray, epochs: int = EPOCHS) -> tuple:
    n_truth = int(np.sum(y == 0))
    n_lie   = int(np.sum(y == 1))
    print(f"\n[Train] Dataset: X={X.shape}  Truth={n_truth}  Lie={n_lie}")

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    y_train_oh = keras.utils.to_categorical(y_train, 2)
    y_test_oh  = keras.utils.to_categorical(y_test,  2)

    # ── IMPROVEMENT 7: Class weights (handles imbalanced data) ────────────
    cw_values = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(cw_values[0]), 1: float(cw_values[1])}
    print(f"[Train] Class weights: Truth={class_weight[0]:.2f}  Lie={class_weight[1]:.2f}")

    # ── Build model ────────────────────────────────────────────────────────
    model = build_improved_cnn(input_shape=X.shape[1:])
    model.summary()

    # ── IMPROVEMENT 5: Cosine LR decay schedule ───────────────────────────
    # Smoothly anneals the learning rate from LR → 0 over training.
    # Much more stable than ReduceLROnPlateau which drops LR too aggressively.
    total_steps   = (len(X_train) // BATCH_SIZE) * epochs
    warmup_steps  = (len(X_train) // BATCH_SIZE) * WARMUP_EPOCHS
    lr_schedule   = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = 1e-3,
        decay_steps           = total_steps,
        alpha                 = 1e-6,       # minimum LR at end of training
        warmup_target         = 1e-3,
        warmup_steps          = warmup_steps,
    )

    # ── IMPROVEMENT 6: Label smoothing (reduces overconfidence) ───────────
    # Instead of hard targets [1,0] or [0,1], uses [0.9, 0.1] etc.
    # Prevents the model from becoming overconfident on small datasets.
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer = keras.optimizers.Adam(lr_schedule),
        loss      = loss_fn,
        metrics   = ["accuracy"]
    )

    # ── Callbacks ──────────────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20,   # ↑ patience — cosine schedule needs more epochs
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "cnn_best_v2.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]
    # NOTE: ReduceLROnPlateau removed — cosine schedule handles it better

    # ── Fit ────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train_oh,
        validation_data = (X_test, y_test_oh),
        epochs          = epochs,
        batch_size      = BATCH_SIZE,
        class_weight    = class_weight,
        callbacks       = cb_list,
        verbose         = 1
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[Eval] Test set results:")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=["Truth", "Lie"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(f"              Predicted Truth  Predicted Lie")
    print(f"Actual Truth       {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Lie         {cm[1][0]:<16} {cm[1][1]}")

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "cnn_lie_model.h5")
    model.save(final_path)
    print(f"\n✅ Improved CNN saved → {final_path}")

    # ── Plot training curves ───────────────────────────────────────────────
    _plot_history(history)

    return model, history


def _plot_history(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1826")
    for ax in axes:
        ax.set_facecolor("#050d1f")
        ax.tick_params(colors="#6a88c0")
        for s in ax.spines.values():
            s.set_edgecolor("#1e3460")
    axes[0].plot(history.history["accuracy"],    color="#00ffcc", label="train")
    axes[0].plot(history.history["val_accuracy"], color="#ff6b6b", label="val")
    axes[0].set_title("Accuracy", color="#c5d8ff")
    axes[0].legend(facecolor="#0d1826", labelcolor="white")
    axes[1].plot(history.history["loss"],    color="#00ffcc", label="train")
    axes[1].plot(history.history["val_loss"], color="#ff6b6b", label="val")
    axes[1].set_title("Loss", color="#c5d8ff")
    axes[1].legend(facecolor="#0d1826", labelcolor="white")
    fig.suptitle("CNN v2 Training History", color="#7aa2ff", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "cnn_training_history_v2.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="#0d1826")
    plt.close(fig)
    print(f"Training plot saved → {out}")


# ── Sample spectrogram preview ─────────────────────────────────────────────────

def save_sample_spectrograms(X: np.ndarray, y: np.ndarray, n: int = 6) -> None:
    """Save side-by-side 3-channel preview so you can verify the input looks right."""
    sample_dir = os.path.join(OUTPUT_DIR, "cnn_spectrogram_samples_v2")
    os.makedirs(sample_dir, exist_ok=True)
    indices = np.random.choice(len(X), min(n, len(X)), replace=False)
    for i, idx in enumerate(indices):
        img   = X[idx]
        label = "Truth" if y[idx] == 0 else "Lie"
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor="#0d1826")
        titles = ["Mel spectrogram", "MFCC", "Delta-MFCC"]
        for j, ax in enumerate(axes):
            ch = img[..., j]
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-9)
            ax.imshow(ch, aspect="auto", cmap="magma", origin="lower")
            ax.set_title(titles[j], color="#c5d8ff", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"Sample {i+1} — {label}", color="#7aa2ff", fontsize=10)
        path = os.path.join(sample_dir, f"sample_{i+1}_{label.lower()}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="#0d1826")
        plt.close(fig)
    print(f"Sample spectrograms saved → {sample_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Improved CNN Lie Detector v2")
    parser.add_argument("--truth_dir",     default="truth_audio",
                        help="Folder of truth .wav/.mp3 files")
    parser.add_argument("--lie_dir",       default="lie_audio",
                        help="Folder of lie .wav/.mp3 files")
    parser.add_argument("--truth_xlsx",    default="truth_features_full.xlsx")
    parser.add_argument("--lie_xlsx",      default="lie_features_full.xlsx")
    parser.add_argument("--from_features", action="store_true",
                        help="Force feature-spreadsheet mode (fallback)")
    parser.add_argument("--epochs",        type=int, default=EPOCHS)
    parser.add_argument("--img_size",      type=int, default=IMG_SIZE,
                        help="Spectrogram image size (default 224)")
    args = parser.parse_args()

    # Override global if user passes --img_size
    img_size = args.img_size

    use_audio = (
        not args.from_features
        and LIBROSA_OK
        and os.path.isdir(args.truth_dir)
        and os.path.isdir(args.lie_dir)
    )

    if use_audio:
        print(f"\n[Mode] AUDIO — truth: {args.truth_dir}  lie: {args.lie_dir}")
        print(f"[Mode] Image size: {img_size}×{img_size}, Duration: {DURATION}s")
        X, y = load_audio_dataset(args.truth_dir, args.lie_dir, img_size)
    else:
        if not args.from_features:
            print(f"\n[WARN] Audio folders not found. Falling back to feature spreadsheets.")
            print(f"       To use audio: place .wav files in '{args.truth_dir}/' and '{args.lie_dir}/'")
        print("\n[Mode] FEATURE SPREADSHEET (fallback)")
        X, y = load_feature_dataset(args.truth_xlsx, args.lie_xlsx, img_size)

    if len(X) == 0:
        print("[ERROR] No samples loaded. Check folder paths.")
        sys.exit(1)

    print(f"\n[Data] Loaded {len(X)} samples | Truth: {np.sum(y==0)} | Lie: {np.sum(y==1)}")

    save_sample_spectrograms(X, y)
    train(X, y, epochs=args.epochs)


if __name__ == "__main__":
    main()