# -*- coding: utf-8 -*-
"""
Life is what you make of it 

Written by Dinho_itt (this is my instagram id)
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to read raw data
def read_raw_file(filename):
    with open(filename, 'rb') as f:
        buffer = f.read()
    return np.frombuffer(buffer, dtype=np.int16)

# Apply Hamming Window
def apply_hamming_window(signal):
    N = len(signal)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return signal * window

# Compute DFT
def compute_dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

# Compute log magnitude spectrum
def compute_log_magnitude_spectrum(dft):
    return np.log(np.abs(dft))

# Apply lifter
def apply_lifter(cepstrum, L):
    N = len(cepstrum)
    liftered_cepstrum = np.zeros(N)
    liftered_cepstrum[:L] = cepstrum[:L]
    liftered_cepstrum[-L:] = cepstrum[-L:]
    return liftered_cepstrum

# Compute spectral envelope
def compute_spectral_envelope_fft(liftered_cepstrum):
    dft = np.fft.fft(liftered_cepstrum)
    return compute_log_magnitude_spectrum(dft)

# Function to downsample the data
def downsample(data):
    return data[::2]

# Read raw data
raw_signal = read_raw_file("C:/Users/Home/Desktop/Male.raw")
# Apply Hamming window
windowed_signal = apply_hamming_window(raw_signal)
# Compute DFT
dft_signal = compute_dft(windowed_signal)
# Downsample the DFT signal
dft_signal_ds = downsample(dft_signal)
# Compute log magnitude spectrum
log_mag_spectrum_ds = compute_log_magnitude_spectrum(dft_signal_ds)
# Compute cepstrum
cepstrum = np.real(compute_dft(log_mag_spectrum_ds))
# Apply lifter
liftered_cepstrum = apply_lifter(cepstrum, 15)
# Compute spectral envelope
spectral_envelope_fft = compute_spectral_envelope_fft(liftered_cepstrum)

# Reduce overall magnitude in half
spectral_envelope_fft_half = spectral_envelope_fft * 0.5
log_mag_spectrum_ds_half = log_mag_spectrum_ds * 0.5
# Selecting only the first 160 values
liftered_cepstrum = liftered_cepstrum[:160]
spectral_envelope_fft_half = spectral_envelope_fft_half[:160]

# Plot Liftered Cepstrum and Log Spectral Envelope
plt.figure(figsize=(14, 6))

# Log Spectral Envelope

plt.plot(spectral_envelope_fft_half)
plt.title('Log Spectral Envlope')

plt.tight_layout()
plt.show()



