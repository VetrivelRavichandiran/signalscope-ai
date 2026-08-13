"""Digital filter design and application helpers."""
from __future__ import annotations

import numpy as np
from scipy import signal


def apply_filter(samples: np.ndarray, sample_rate: float, filter_type: str,
                 cutoff_1: float, cutoff_2: float | None = None, order: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a stable Butterworth SOS filter and return filtered samples and response."""
    nyquist = sample_rate / 2
    ftype = filter_type.lower()
    if ftype in {"bandpass", "bandstop"}:
        if cutoff_2 is None or not 0 < cutoff_1 < cutoff_2 < nyquist:
            raise ValueError("Band filters need two cutoffs where 0 < lower < upper < Nyquist.")
        wn = [cutoff_1 / nyquist, cutoff_2 / nyquist]
    else:
        if not 0 < cutoff_1 < nyquist:
            raise ValueError("Cutoff frequency must lie between 0 and the Nyquist frequency.")
        wn = cutoff_1 / nyquist
    sos = signal.butter(order, wn, btype=ftype, output="sos")
    filtered = signal.sosfiltfilt(sos, np.asarray(samples, dtype=float))
    response_f, response = signal.sosfreqz(sos, worN=2048, fs=sample_rate)
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
    return filtered, response_f, response_db
