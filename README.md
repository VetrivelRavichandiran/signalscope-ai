# SignalScope AI

**SignalScope AI** is a fully software-based interactive DSP laboratory for exploring generated or uploaded waveforms. It is designed as a high-impact B.Tech ECE portfolio project: it combines digital signal processing, interactive visualisation, digital filters, and lightweight machine-learning classification.

## Features

- Generate noisy **Sine, Square, AM, FM, FSK, and Chirp** signals.
- Upload mono/stereo **WAV**, numeric **CSV**, or numeric **TXT** sample files.
- Inspect time-domain, windowed FFT, and interactive spectrogram views.
- Estimate dominant frequency, occupied bandwidth, spectral entropy, robust noise floor, and approximate SNR.
- Design low-pass, high-pass, band-pass, and band-stop Butterworth filters.
- Compare original and filtered waveforms and inspect filter frequency response.
- Classify waveforms locally with a Random Forest model trained automatically from synthetic data.
- Produce an offline, engineering-focused analysis narrative; no cloud API key is required.
- Export processed samples and analysis metrics as CSV.

## Installation

```bash
cd /opt/sandbox/workspace/signalscope-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints in your browser.

## Recommended live demonstration

1. Start with the built-in noisy FSK waveform and show its frequency hopping between two tones in the spectrogram.
2. Reduce injected SNR and show the SNR, noise floor, classifier confidence, and spectral plot changing together.
3. Switch to AM and FM and explain how the sidebands/spectral characteristics differ.
4. Enable a band-pass filter around the useful signal and overlay original versus filtered waveforms.
5. Export the final metrics CSV to demonstrate reproducible analysis.

## Technical scope and honest limitations

This application analyses sampled baseband/audio-style waveforms in software. It is **not** a calibrated spectrum analyser, an SDR receiver, or a general-purpose unknown RF-modulation recogniser. To analyse real RF transmissions, an authorised SDR or legal captured I/Q data source would be required. The present scope is deliberately standalone, safe, low-cost, and fully demonstrable without RF hardware.

## CSV format

The simplest supported CSV contains an amplitude column:

```csv
amplitude
0.12
0.15
-0.02
```

Optional `time` (seconds) enables sampling-rate inference; optional `sample_rate` provides an explicit rate.
