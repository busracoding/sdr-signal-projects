# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:44:58 2025

@author: stajyer1
"""

import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# DLL ayarı
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# SDR ayarı
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

# Sweep bölgeleri
start_freq = 70
end_freq = 6000
region_count = 10
region_width = (end_freq - start_freq) // region_count
peak_frequencies_all = []

# Sweep döngüsü
for i in range(region_count):
    region_start = start_freq + i * region_width
    region_end = region_start + region_width
    sweep_freqs = list(range(region_start, region_end, 5))

    print(f"\n🟣 Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.figure(figsize=(12, 4))

    for freq in sweep_freqs:
        sdr.rx_lo = int(freq * 1e6)
        samples = sdr.rx()

        N = len(samples)
        T = 1.0 / sdr.sample_rate
        xf = fftfreq(N, T)
        yf = fft(samples)

        xf_mhz = xf[:N//2] / 1e6
        yf_mag = 2.0 / N * np.abs(yf[:N//2])
        true_freq = xf_mhz + freq

        if np.max(yf_mag) < 0.01:
            continue

        # Çizim (saydam)
        plt.plot(true_freq, yf_mag, alpha=0.3)

        # 🔥 DAHA SIKI PEAK SEÇİMİ 🔥
        peaks, properties = find_peaks(
            yf_mag,
            height=np.max(yf_mag) * 0.7,   # sadece güçlü sinyaller
            distance=10,                   # çok yakınlar tek alınır
            prominence=0.01                # kenardan ayrışma
        )

        peak_freqs = true_freq[peaks]
        peak_heights = properties['peak_heights']

        # çok küçük tepe varsa atla
        strong_peaks = peak_freqs[peak_heights > 0.01]
        peak_frequencies_all.extend(strong_peaks)

    plt.title(f"Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- DBSCAN ile 20 kHz içinde grupla ---
print("\n📉 Toplam ham peak:", len(peak_frequencies_all))

if len(peak_frequencies_all) > 0:
    freqs_mhz = np.array(peak_frequencies_all).reshape(-1, 1)
    db = DBSCAN(eps=0.02, min_samples=2).fit(freqs_mhz)  # 20 kHz, en az 2 sinyal

    clusters = db.labels_
    grouped_peaks = [
        np.mean(freqs_mhz[clusters == i])
        for i in np.unique(clusters)
        if np.count_nonzero(clusters == i) > 1  # yalnızca gruplar
    ]

    grouped_peaks = np.round(sorted(grouped_peaks), 6)

    print(f"\n✅ Seçilen Temiz Peak Frekanslar ({len(grouped_peaks)} adet):")
    for f in grouped_peaks:
        print(f" - {f} MHz")
else:
    print("⚠️ Peak bulunamadı.")
