from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import chirp


# Folder where the sample signal CSV files will be saved
OUTPUT_FOLDER = Path("assets/sample_signals")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Common signal parameters
SAMPLE_RATE = 16000
DURATION = 2.0
time = np.arange(0, DURATION, 1 / SAMPLE_RATE)

# Fixed random seed: same samples are generated every time
rng = np.random.default_rng(42)


def add_noise(signal, snr_db):
    """
    Adds white Gaussian noise to a signal.
    snr_db: requested Signal-to-Noise Ratio in dB.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def save_signal(filename, signal_data):
    """
    Save a signal as a CSV file compatible with SignalScope AI.
    """
    dataframe = pd.DataFrame(
        {
            "time": time,
            "amplitude": signal_data,
            "sample_rate": SAMPLE_RATE,
        }
    )

    output_path = OUTPUT_FOLDER / filename
    dataframe.to_csv(output_path, index=False)

    print(f"Created: {output_path}")


# ---------------------------------------------------------
# 1. Noisy 1 kHz sine wave
# ---------------------------------------------------------
sine_signal = np.sin(2 * np.pi * 1000 * time)
sine_signal = add_noise(sine_signal, snr_db=10)

save_signal("sine_1khz_noisy.csv", sine_signal)


# ---------------------------------------------------------
# 2. AM: Amplitude Modulation
# Carrier = 1000 Hz, message frequency = 12 Hz
# ---------------------------------------------------------
carrier_frequency = 1000
message_frequency = 12

message_signal = 0.7 * np.sin(2 * np.pi * message_frequency * time)

am_signal = (
    1 + message_signal
) * np.cos(2 * np.pi * carrier_frequency * time)

am_signal = add_noise(am_signal, snr_db=20)

save_signal("am_signal.csv", am_signal)


# ---------------------------------------------------------
# 3. FM: Frequency Modulation
# ---------------------------------------------------------
fm_signal = np.cos(
    2 * np.pi * carrier_frequency * time
    + 5 * np.sin(2 * np.pi * message_frequency * time)
)

fm_signal = add_noise(fm_signal, snr_db=20)

save_signal("fm_signal.csv", fm_signal)


# ---------------------------------------------------------
# 4. FSK: Frequency Shift Keying
# Bit 0 = 650 Hz
# Bit 1 = 1350 Hz
# ---------------------------------------------------------
symbol_rate = 40
number_of_symbols = int(DURATION * symbol_rate)

bits = rng.integers(0, 2, number_of_symbols)

samples_per_symbol = int(SAMPLE_RATE / symbol_rate)
expanded_bits = np.repeat(bits, samples_per_symbol)

# Ensure bit array has same length as time array
expanded_bits = expanded_bits[:len(time)]

fsk_frequency = np.where(expanded_bits == 1, 1350, 650)

# Integrate frequency to create continuous phase FSK
fsk_signal = np.sin(
    2 * np.pi * np.cumsum(fsk_frequency) / SAMPLE_RATE
)

fsk_signal = add_noise(fsk_signal, snr_db=15)

save_signal("noisy_fsk.csv", fsk_signal)


# ---------------------------------------------------------
# 5. Chirp: frequency sweep from 300 Hz to 3000 Hz
# ---------------------------------------------------------
chirp_signal = chirp(
    time,
    f0=300,
    f1=3000,
    t1=DURATION,
    method="linear",
)

chirp_signal = add_noise(chirp_signal, snr_db=18)

save_signal("chirp_signal.csv", chirp_signal)


# ---------------------------------------------------------
# 6. Square wave for waveform-classifier testing
# ---------------------------------------------------------
square_signal = np.sign(
    np.sin(2 * np.pi * 800 * time)
)

square_signal = add_noise(square_signal, snr_db=18)

save_signal("square_wave.csv", square_signal)

print("\nAll sample signal files were generated successfully.")