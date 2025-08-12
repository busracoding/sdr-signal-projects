# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:14:25 2025

@author: stajyer1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import adi
import time

# SDR bağlantısı
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_rf_bandwidth = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 4096
sdr.gain_control_mode = "manual"
sdr.rx_hardwaregain_chan0 = 40

# Sweep ayarları
start_freq = int(70e6)
stop_freq = int(600e6)
step_hz = int(2e6)

all_freqs = []
all_magnitudes = []

print("Başlıyoruz...")

for freq in range(start_freq, stop_freq, step_hz):
    print(f"Frekans taranıyor: {freq/1e6:.1f} MHz")
    sdr.rx_lo = freq
    time.sleep(0.1)  # PLL otursun

    samples = sdr.rx()
    fft_data = fft(samples)
    power = 20 * np.log10(np.abs(fft_data) + 1e-6)

    # Frekans ekseni (offsetli)
    freqs = fftfreq(len(samples), 1/sdr.sample_rate) + freq
    all_freqs.extend(freqs)
    all_magnitudes.extend(power)

# Grafik çizimi
plt.figure(figsize=(12, 6))
plt.plot(np.array(all_freqs)/1e6, all_magnitudes)
plt.title("Geniş Bant Spektrum Tarama Sonucu")
plt.xlabel("Frekans (MHz)")
plt.ylabel("Güç (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
