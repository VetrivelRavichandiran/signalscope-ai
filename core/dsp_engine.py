"""Core signal-generation and DSP metrics for SignalScope AI."""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


def time_axis(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    return np.arange(len(samples), dtype=float) / float(sample_rate)


def normalize(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    return samples / peak if peak > EPS else samples.copy()


def add_awgn(samples: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise at an approximate requested SNR."""
    rng = rng or np.random.default_rng()
    power = float(np.mean(np.square(samples))) + EPS
    noise_power = power / (10 ** (snr_db / 10))
    return np.asarray(samples, dtype=float) + rng.normal(0, np.sqrt(noise_power), len(samples))


def generate_signal(kind: str, sample_rate: int = 16000, duration: float = 1.5,
                    carrier_hz: float = 1000.0, snr_db: float = 22.0,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible synthetic modulation/audio-like test signal."""
    rng = rng or np.random.default_rng(42)
    t = np.arange(int(sample_rate * duration), dtype=float) / sample_rate
    kind = kind.upper()
    message = 0.65 * np.sin(2 * np.pi * 7 * t)

    if kind == "SINE":
        clean = np.sin(2 * np.pi * carrier_hz * t)
    elif kind == "SQUARE":
        clean = signal.square(2 * np.pi * carrier_hz * t)
    elif kind == "AM":
        clean = (1 + 0.7 * message) * np.cos(2 * np.pi * carrier_hz * t)
    elif kind == "FM":
        phase = 2 * np.pi * carrier_hz * t + 4.5 * np.sin(2 * np.pi * 7 * t)
        clean = np.cos(phase)
    elif kind == "FSK":
        symbol_rate = 50
        bits = rng.integers(0, 2, int(np.ceil(duration * symbol_rate)))
        symbols = np.repeat(bits, int(np.ceil(sample_rate / symbol_rate)))[:len(t)]
        frequencies = np.where(symbols > 0, carrier_hz * 1.35, carrier_hz * 0.65)
        clean = np.sin(2 * np.pi * np.cumsum(frequencies) / sample_rate)
    elif kind == "CHIRP":
        clean = signal.chirp(t, f0=carrier_hz * 0.35, f1=carrier_hz * 1.8, t1=duration, method="linear")
    else:
        raise ValueError(f"Unsupported generated signal type: {kind}")

    return add_awgn(clean, snr_db, rng), clean


def compute_fft(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=float)
    if len(x) < 2:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(len(x), d=1 / sample_rate)
    magnitude_db = 20 * np.log10(np.abs(spectrum) / max(np.sum(window) / 2, EPS) + EPS)
    return freqs, magnitude_db


def estimate_noise_floor(samples: np.ndarray) -> float:
    """Robust amplitude-domain noise estimate using median absolute deviation."""
    x = np.asarray(samples, dtype=float)
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def calculate_snr_db(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate SNR by separating prominent spectral peaks from residual noise."""
    
    x = np.asarray(samples, dtype=float)

    if len(x) < 32 or np.std(x) < EPS:
        return float("nan")

    _, pxx = signal.periodogram(
        x,
        fs=sample_rate,
        scaling="spectrum"
    )

    if len(pxx) < 3:
        return float("nan")

    # find_peaks returns: peaks, properties
    peaks, _ = signal.find_peaks(
        pxx,
        prominence=np.max(pxx) * 0.01
    )

    keep = np.zeros_like(pxx, dtype=bool)

    # Keep up to five strongest frequency peaks
    strongest_peaks = peaks[np.argsort(pxx[peaks])[-5:]]

    for idx in strongest_peaks:
        keep[max(0, idx - 1): min(len(pxx), idx + 2)] = True

    signal_power = float(np.sum(pxx[keep]))
    noise_power = float(np.sum(pxx[~keep])) + EPS

    return float(
        10 * np.log10((signal_power + EPS) / noise_power)
    )

def spectral_features(samples: np.ndarray, sample_rate: float) -> dict[str, float]:
    freqs, mag_db = compute_fft(samples, sample_rate)
    linear = 10 ** (mag_db / 20)
    power = linear ** 2 + EPS
    power /= np.sum(power)
    centroid = float(np.sum(freqs * power))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power)))
    entropy = float(-np.sum(power * np.log2(power)))
    peak_idx = int(np.argmax(mag_db))
    threshold = np.max(mag_db) - 20
    active = freqs[mag_db >= threshold]
    occupied_bw = float(active[-1] - active[0]) if len(active) > 1 else 0.0
    return {
        "dominant_frequency_hz": float(freqs[peak_idx]),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "occupied_bandwidth_hz": occupied_bw,
        "spectral_entropy": entropy,
        "noise_floor": estimate_noise_floor(samples),
        "snr_db": calculate_snr_db(samples, sample_rate),
    }


def make_spectrogram(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(512, max(64, len(samples) // 8))
    freqs, times, spec = signal.spectrogram(samples, fs=sample_rate, nperseg=nperseg,
                                             noverlap=nperseg // 2, scaling="spectrum")
    return freqs, times, 10 * np.log10(spec + EPS)
