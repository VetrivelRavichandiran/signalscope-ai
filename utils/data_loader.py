"""Load WAV, CSV, and text signal samples into a single channel float array."""
from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from scipy.io import wavfile


def _to_mono_float(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(float) / max(abs(info.min), info.max)
    else:
        arr = arr.astype(float)
    arr = np.nan_to_num(arr)
    return arr


def load_uploaded_signal(uploaded_file, fallback_sample_rate: float = 16000.0) -> tuple[np.ndarray, float, str]:
    """Return samples, sampling rate and a short source description."""
    name = uploaded_file.name.lower()
    blob = uploaded_file.getvalue()
    if name.endswith(".wav"):
        sample_rate, data = wavfile.read(BytesIO(blob))
        return _to_mono_float(data), float(sample_rate), "WAV audio signal"

    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(blob))
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("CSV needs at least one numeric column containing amplitude samples.")
        sample_col = next((c for c in numeric.columns if c.lower() in {"signal", "amplitude", "samples", "value", "iq_i"}), numeric.columns[0])
        sample_rate = fallback_sample_rate
        if "sample_rate" in df.columns:
            sample_rate = float(df["sample_rate"].iloc[0])
        elif "time" in numeric.columns and len(df) > 2:
            dt = float(np.median(np.diff(df["time"].to_numpy())))
            if dt > 0:
                sample_rate = 1 / dt
        return _to_mono_float(numeric[sample_col].to_numpy()), sample_rate, f"CSV column: {sample_col}"

    if name.endswith(".txt"):
        values = np.loadtxt(BytesIO(blob), delimiter=None)
        return _to_mono_float(values), float(fallback_sample_rate), "Text samples"
    raise ValueError("Supported formats are WAV, CSV, and TXT.")
