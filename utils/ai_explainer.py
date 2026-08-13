"""Offline engineering explanations. No API key or internet is required."""
from __future__ import annotations


def engineering_insight(metrics: dict, predicted_class: str, confidence: float) -> str:
    snr = metrics.get("snr_db", float("nan"))
    freq = metrics.get("dominant_frequency_hz", 0.0)
    bandwidth = metrics.get("occupied_bandwidth_hz", 0.0)
    entropy = metrics.get("spectral_entropy", 0.0)

    if snr != snr:
        quality = "Signal-to-noise ratio could not be reliably estimated for this short or nearly constant waveform."
    elif snr >= 20:
        quality = f"The signal is clean (estimated SNR {snr:.1f} dB), so its principal spectral components are clearly separated from noise."
    elif snr >= 8:
        quality = f"The signal has usable quality (estimated SNR {snr:.1f} dB), though noise may affect weak spectral components."
    else:
        quality = f"The signal is noise-dominated (estimated SNR {snr:.1f} dB); filtering or stronger acquisition conditions may be needed."

    complexity = "narrow-band and highly ordered" if entropy < 5 else "spectrally complex or wide-band"
    return (
        f"**Offline DSP assessment:** The classifier identifies the waveform as **{predicted_class}** "
        f"with **{confidence:.0%}** confidence. The strongest frequency component is near **{freq:,.1f} Hz**, "
        f"with an estimated occupied bandwidth of **{bandwidth:,.1f} Hz**. The spectrum is {complexity}. "
        f"{quality} This conclusion is a software analysis of the loaded/generated baseband waveform, not a certified RF measurement."
    )
