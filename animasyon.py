# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:05:49 2025

@author: stajyer1
"""

import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time

# --- DLL & PATH ayarı ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# --- Kullanıcı ayarları ---
center_freq_mhz = 96     # Merkez frekans
sample_rate = 2e6        # Örnekleme hızı
buffer_size = 1024       # SDR RX buffer
refresh_interval = 0.3   # Grafik güncelleme süresi (saniye)

# --- SDR ayarla ---
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_lo = int(center_freq_mhz * 1e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = buffer_size

# --- Grafik başlat ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=1.5)
ax.set_title(f"Gerçek Zamanlı FFT - Merkez: {center_freq_mhz} MHz")
ax.set_xlabel("Frekans (MHz)")
ax.set_ylabel("Genlik")
ax.grid(True)

xf = fftfreq(buffer_size, 1 / sample_rate)[:buffer_size//2] / 1e6  # MHz
true_freqs = xf + center_freq_mhz
line.set_xdata(true_freqs)
ax.set_xlim(true_freqs[0], true_freqs[-1])
ax.set_ylim(0, 0.05)  # Gerekirse otomatik güncelle eklenebilir

# --- Döngü: canlı veri güncellemesi ---
try:
    while True:
        samples = sdr.rx()
        yf = fft(samples)
        yf_mag = 2.0 / buffer_size * np.abs(yf[:buffer_size // 2])
        line.set_ydata(yf_mag)

        # Gerekirse Y ekseni otomatik güncelle
        ax.set_ylim(0, np.max(yf_mag) * 1.2)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(refresh_interval)
except KeyboardInterrupt:
    print("🛑 Animasyon durduruldu.")
    plt.ioff()
    plt.show() 
    
    
    """gerçek zamanlı animasyon oluşturur"""
