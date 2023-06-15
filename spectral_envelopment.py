# -*- coding: utf-8 -*-
"""
Life is what you make of it 

Written by Dinho_itt (this is my instagram id)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd

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
def compute_spectral_envelope(liftered_cepstrum):
    dft = compute_dft(liftered_cepstrum)
    return compute_log_magnitude_spectrum(dft)

# Read raw data
raw_signal = read_raw_file("C:/Users/Home/Desktop/Male.raw")
# Apply Hamming window
windowed_signal = apply_hamming_window(raw_signal)
# Compute DFT
dft_signal = compute_dft(windowed_signal)
# Compute log magnitude spectrum
log_mag_spectrum = compute_log_magnitude_spectrum(dft_signal)
# Compute cepstrum
cepstrum = np.real(compute_dft(log_mag_spectrum))
# Apply lifter
liftered_cepstrum = apply_lifter(cepstrum, 15)
# Compute spectral envelope
spectral_envelope = compute_spectral_envelope(liftered_cepstrum)

# Plot log magnitude spectrum
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(log_mag_spectrum)
plt.subplot(1, 2, 2)
plt.plot(liftered_cepstrum, markerfacecolor='red')
plt.title('liftered_cepstrume')

# Plot spectral envelope
plt.subplot(1, 2, 2)
plt.plot(spectral_envelope)
plt.title('Spectral Envelope')
plt.show()




