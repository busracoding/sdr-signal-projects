# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:15:17 2025

@author: stajyer1
"""

import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- DLL yükle ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# --- SDR Ayarları ---
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

# --- Sweep bölgeleri: 3 parçaya bölünmüş 80–200 MHz ---
regions = {
    "Bölge 1 (80–120 MHz)": list(range(80, 120, 5)),
    "Bölge 2 (120–160 MHz)": list(range(120, 160, 5)),
    "Bölge 3 (160–200 MHz)": list(range(160, 200, 5)),
}

for region_name, frequencies in regions.items():
    print(f"\n🟩 {region_name}")
    
    for freq in frequencies:
        print(f"  📡 Sweep frekansı: {freq} MHz")

        # LO frekansını ayarla
        sdr.rx_lo = int(freq * 1e6)

        # Veri al
        samples = sdr.rx()

        # FFT hesapla
        N = len(samples)
        T = 1.0 / sdr.sample_rate
        xf = fftfreq(N, T)
        yf = fft(samples)

        xf_mhz = xf[:N//2] / 1e6
        yf_mag = 2.0/N * np.abs(yf[:N//2])

        # Grafik
        plt.figure(figsize=(8, 3))
        plt.plot(xf_mhz + freq, yf_mag)  # Frekansı LO'ya göre kaydır
        plt.title(f"{region_name} - LO: {freq} MHz (Gözlem: {freq-1}–{freq+1} MHz)")
        plt.xlabel("Frekans (MHz)")
        plt.ylabel("Genlik")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
