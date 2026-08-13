"""SignalScope AI — Interactive DSP and synthetic modulation analysis studio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.dsp_engine import compute_fft, generate_signal, make_spectrogram, normalize, time_axis
from core.filters import apply_filter
from core.ml_classifier import predict_signal
from utils.ai_explainer import engineering_insight
from utils.data_loader import load_uploaded_signal

st.set_page_config(page_title="SignalScope AI", page_icon="📡", layout="wide")


@st.cache_data(show_spinner=False)
def generated(kind: str, rate: int, duration: float, carrier: float, snr: float):
    return generate_signal(kind, rate, duration, carrier, snr)


def line_plot(x, y, title, x_label, y_label, color="#00D4FF"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.7)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_dark",
                      height=410, margin=dict(l=20, r=20, t=55, b=30))
    return fig


def metric_cards(metrics: dict, classification: str, confidence: float):
    cols = st.columns(5)
    snr = metrics["snr_db"]
    cols[0].metric("Detected waveform", classification)
    cols[1].metric("Model confidence", f"{confidence:.0%}")
    cols[2].metric("Dominant frequency", f"{metrics['dominant_frequency_hz']:,.1f} Hz")
    cols[3].metric("Estimated SNR", "N/A" if not np.isfinite(snr) else f"{snr:.1f} dB")
    cols[4].metric("Occupied bandwidth", f"{metrics['occupied_bandwidth_hz']:,.1f} Hz")


st.title("📡 SignalScope AI")
st.caption("Interactive DSP, filtering and ML-assisted waveform analysis — completely software-based and offline.")

with st.sidebar:
    st.header("Signal source")
    mode = st.radio("Choose input", ["Generate demo signal", "Upload WAV / CSV / TXT"])
    sample_rate = 16000
    source_label = ""

    if mode == "Generate demo signal":
        signal_type = st.selectbox("Waveform", ["FSK", "AM", "FM", "CHIRP", "SINE", "SQUARE"], index=0)
        sample_rate = st.select_slider("Sampling rate (Hz)", options=[8000, 12000, 16000, 22050, 44100], value=16000)
        duration = st.slider("Duration (seconds)", 0.5, 3.0, 1.5, 0.1)
        carrier = st.slider("Carrier / centre frequency (Hz)", 100, int(sample_rate / 3), min(1000, int(sample_rate / 4)), 25)
        requested_snr = st.slider("Injected SNR (dB)", 0, 40, 22)
        samples, clean = generated(signal_type, sample_rate, duration, carrier, requested_snr)
        source_label = f"Synthetic {signal_type} with injected AWGN"
    else:
        uploaded = st.file_uploader("Upload a signal", type=["wav", "csv", "txt"])
        fallback_rate = st.number_input("Fallback sample rate for CSV/TXT (Hz)", 1000, 192000, 16000, 1000)
        if uploaded is None:
            st.info("Upload a file, or switch to a generated demo signal.")
            st.stop()
        try:
            samples, sample_rate, source_label = load_uploaded_signal(uploaded, fallback_rate)
            clean = None
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    st.divider()
    st.header("Filter laboratory")
    enable_filter = st.toggle("Enable Butterworth filter", value=False)
    filter_type = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass", "bandstop"])
    nyquist = sample_rate / 2
    cutoff_1 = st.slider("Cutoff frequency (Hz)", 10.0, float(max(20, nyquist - 20)), float(min(1500, nyquist / 3)), 10.0)
    cutoff_2 = None
    if filter_type in {"bandpass", "bandstop"}:
        lower_max = float(max(20, nyquist - 60))
        cutoff_1 = min(cutoff_1, lower_max)
        cutoff_2 = st.slider("Upper cutoff (Hz)", float(cutoff_1 + 10), float(nyquist - 1), float(min(nyquist - 1, cutoff_1 * 2)), 10.0)
    filter_order = st.slider("Filter order", 2, 10, 4)

samples = normalize(samples)
processed = samples
response_f = response_db = None
if enable_filter:
    try:
        processed, response_f, response_db = apply_filter(samples, sample_rate, filter_type, cutoff_1, cutoff_2, filter_order)
    except ValueError as exc:
        st.warning(f"Filter not applied: {exc}")

try:
    predicted, confidence, metrics = predict_signal(processed, sample_rate)
except Exception as exc:
    st.error(f"Classifier could not process this signal: {exc}")
    st.stop()

st.success(f"Source: {source_label}  •  {len(processed):,} samples at {sample_rate:,.0f} Hz")
metric_cards(metrics, predicted, confidence)

tab_time, tab_frequency, tab_spectrogram, tab_filter, tab_ai, tab_export = st.tabs(
    ["⏱ Time domain", "📈 Frequency domain", "🌈 Spectrogram", "🎛 Filter lab", "🧠 AI insight", "⬇ Export"]
)

with tab_time:
    t = time_axis(processed, sample_rate)
    max_seconds = min(0.08, t[-1]) if len(t) else 0
    zoom = st.slider("Display window (seconds)", 0.01, max(0.02, float(min(1.0, t[-1]))), max_seconds, 0.01, key="time_zoom")
    n = min(len(processed), max(10, int(zoom * sample_rate)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[:n], y=samples[:n], name="Original", mode="lines", line=dict(color="#7F8C8D", width=1)))
    if enable_filter:
        fig.add_trace(go.Scatter(x=t[:n], y=processed[:n], name="Filtered", mode="lines", line=dict(color="#00D4FF", width=2)))
    fig.update_layout(template="plotly_dark", title="Waveform amplitude", xaxis_title="Time (s)", yaxis_title="Normalized amplitude", height=430)
    st.plotly_chart(fig, use_container_width=True)

with tab_frequency:
    freqs, magnitude = compute_fft(processed, sample_rate)
    fmax = st.slider("Maximum displayed frequency (Hz)", 100.0, float(sample_rate / 2), float(min(sample_rate / 2, 5000)), 100.0)
    mask = freqs <= fmax
    st.plotly_chart(line_plot(freqs[mask], magnitude[mask], "Windowed FFT magnitude", "Frequency (Hz)", "Magnitude (dB)"), use_container_width=True)
    st.caption("The FFT uses a Hann window to reduce spectral leakage. Dominant frequency and occupied bandwidth are derived from this spectrum.")

with tab_spectrogram:
    sf, stime, spec = make_spectrogram(processed, sample_rate)
    limit = st.slider("Spectrogram frequency ceiling (Hz)", 100.0, float(sample_rate / 2), float(min(sample_rate / 2, 5000)), 100.0, key="spec_limit")
    mask = sf <= limit
    fig = go.Figure(go.Heatmap(x=stime, y=sf[mask], z=spec[mask], colorscale="Turbo", colorbar=dict(title="dB")))
    fig.update_layout(template="plotly_dark", title="Time–frequency spectrogram", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)", height=470)
    st.plotly_chart(fig, use_container_width=True)

with tab_filter:
    st.markdown("Design a digital Butterworth filter in the sidebar and compare its effect with the original signal.")
    if response_f is not None:
        st.plotly_chart(line_plot(response_f, response_db, f"{filter_order}th-order {filter_type} response", "Frequency (Hz)", "Gain (dB)", "#FFB000"), use_container_width=True)
        st.plotly_chart(line_plot(time_axis(processed, sample_rate)[:min(2500, len(processed))], processed[:min(2500, len(processed))], "Filtered waveform preview", "Time (s)", "Amplitude", "#00D4FF"), use_container_width=True)
    else:
        st.info("Turn on the filter in the sidebar to design and inspect a response.")

with tab_ai:
    st.subheader("AI-assisted engineering explanation")
    st.markdown(engineering_insight(metrics, predicted, confidence))
    st.divider()
    st.subheader("Feature values used by the classifier")
    feature_table = pd.DataFrame({"Feature": list(metrics.keys()), "Value": list(metrics.values())})
    st.dataframe(feature_table, use_container_width=True, hide_index=True)
    st.caption("The classifier is a local Random Forest trained on synthetic Sine, Square, AM, FM, FSK and Chirp examples. It is educational, not a replacement for an SDR receiver or a calibrated RF instrument.")

with tab_export:
    table = pd.DataFrame({"time_s": time_axis(processed, sample_rate), "amplitude": processed})
    st.download_button("Download processed signal as CSV", table.to_csv(index=False).encode("utf-8"), "signalscope_processed.csv", "text/csv")
    report = pd.DataFrame([{"classification": predicted, "confidence": confidence, **metrics}])
    st.download_button("Download analysis metrics as CSV", report.to_csv(index=False).encode("utf-8"), "signalscope_metrics.csv", "text/csv")
