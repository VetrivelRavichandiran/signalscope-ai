"""Lightweight waveform classifier trained on synthetic signal features."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.dsp_engine import generate_signal, spectral_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "saved_model.pkl"
FEATURE_NAMES = [
    "dominant_frequency_hz", "spectral_centroid_hz", "spectral_bandwidth_hz",
    "occupied_bandwidth_hz", "spectral_entropy", "noise_floor", "snr_db",
    "zero_crossing_rate", "crest_factor",
]


def feature_vector(samples: np.ndarray, sample_rate: float) -> dict[str, float]:
    result = spectral_features(samples, sample_rate)
    x = np.asarray(samples, dtype=float)
    result["zero_crossing_rate"] = float(np.mean(np.diff(np.signbit(x)) != 0)) if len(x) > 1 else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-12
    result["crest_factor"] = float(np.max(np.abs(x)) / rms)
    result["snr_db"] = result["snr_db"] if np.isfinite(result["snr_db"]) else -10.0
    return result


def train_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Train a compact model from deliberately varied synthetic examples."""
    rows, labels = [], []
    rng = np.random.default_rng(2026)
    classes = ["SINE", "SQUARE", "AM", "FM", "FSK", "CHIRP"]
    for label in classes:
        for _ in range(55):
            rate = int(rng.choice([8000, 12000, 16000, 22050]))
            carrier = float(rng.uniform(250, min(3200, rate / 4)))
            duration = float(rng.uniform(0.7, 1.25))
            noisy, _ = generate_signal(label, rate, duration, carrier, float(rng.uniform(5, 34)), rng)
            values = feature_vector(noisy, rate)
            rows.append([values[name] for name in FEATURE_NAMES])
            labels.append(label)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    model.fit(np.asarray(rows), labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def load_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


def predict_signal(samples: np.ndarray, sample_rate: float) -> tuple[str, float, dict[str, float]]:
    model = load_model()
    features = feature_vector(samples, sample_rate)
    row = np.asarray([[features[name] for name in FEATURE_NAMES]])
    probabilities = model.predict_proba(row)[0]
    index = int(np.argmax(probabilities))
    return str(model.classes_[index]), float(probabilities[index]), features
