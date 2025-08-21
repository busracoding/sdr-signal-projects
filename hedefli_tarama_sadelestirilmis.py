
# -*- coding: utf-8 -*-
import os
import ctypes
import adi
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, firwin, lfilter, get_window
from sklearn.cluster import DBSCAN
from tkinter import ttk

# --- DLL ve SDR ayarı ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 2048

# Parametreler
n_bands = 8
band_width = (sdr.sample_rate // 2) // n_bands
filter_order = 128
window_type = 'hann'
peak_frequencies_all = []
max_power_peaks = []

# Araç türlerine göre frekans aralıkları
vehicle_freq_ranges = {
    "drone": [(2400, 2500), (5725, 5875)],
    "military_aircraft": [(960, 1215), (2000, 4000)],
    "ground_vehicle": [(70, 512), (890, 960)],
    "naval": [(225, 400), (3000, 6000)],
}

# Tarama tipi seçimi
print("🧭 Tarama tipi seçin:")
print("1. Genel Tarama (70–6000 MHz)")
print("2. Hedef Taraması (araç tipi seçerek)")

mode = int(input("Seçiminiz (1 veya 2): "))

if mode == 1:
    regions = [(70 + i * 593, 70 + (i + 1) * 593) for i in range(10)]
elif mode == 2:
    print("🎯 Tarama yapılacak hedef türünü seçiniz:")
    for i, vt in enumerate(vehicle_freq_ranges.keys(), 1):
        print(f"{i}. {vt}")
    choice = int(input("Seçiminizi yapın (sayı olarak): "))
    vehicle_types = list(vehicle_freq_ranges.keys())
    if 1 <= choice <= len(vehicle_types):
        target_type = vehicle_types[choice - 1]
        print(f"✅ Seçilen hedef tipi: {target_type}")
    else:
        target_type = "drone"
    regions = vehicle_freq_ranges[target_type]
else:
    print("⚠️ Geçersiz seçim, varsayılan: Genel tarama")
    regions = [(70 + i * 593, 70 + (i + 1) * 593) for i in range(10)]

# Tarama fonksiyonu
for i, (region_start, region_end) in enumerate(regions):
    sweep_freqs = list(range(region_start, region_end, 5))
    print(f"🔍 Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.figure(figsize=(12, 6))

    for freq in sweep_freqs:
        try:
            sdr.rx_lo = int(freq * 1e6)
            time.sleep(0.05)
            samples = sdr.rx()
            samples = samples - np.mean(samples)

            for b in range(n_bands):
                low = b * band_width
                high = low + band_width
                if low == 0: low = 1
                if high >= sdr.sample_rate // 2: high = (sdr.sample_rate // 2) - 1
                if low >= high: continue

                taps = firwin(filter_order, [low / (sdr.sample_rate/2), high / (sdr.sample_rate/2)], pass_zero=False, window=window_type)
                filtered = lfilter(taps, 1.0, samples)
                windowed = filtered * get_window(window_type, len(filtered))
                N = len(windowed)
                T = 1.0 / sdr.sample_rate
                xf = fftfreq(N, T)
                yf = fft(windowed)
                xf_mhz = xf[:N//2] / 1e6 + freq + (low / 1e6)
                yf_mag = 2.0 / N * np.abs(yf[:N//2])
                if np.max(yf_mag) < 0.01: continue

                plt.plot(xf_mhz, yf_mag, alpha=0.2)
                peaks, properties = find_peaks(yf_mag, height=np.max(yf_mag)*0.7, distance=10, prominence=1)
                peak_freqs = xf_mhz[peaks]
                peak_heights = properties['peak_heights']
                strong_peaks = peak_freqs[peak_heights > 0.01]
                peak_frequencies_all.extend(strong_peaks)

                max_idx = np.argmax(yf_mag)
                max_power_freq = xf_mhz[max_idx]
                max_power_peaks.append(max_power_freq)
        except Exception as e:
            print(f"⚠️ {freq} MHz alınamadı: {e}")
            continue

    plt.title(f"Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# DBSCAN ile frekans gruplama
print("\n📉 Toplam ham peak:", len(peak_frequencies_all))
grouped_peaks = []
if len(peak_frequencies_all) > 0:
    freqs_mhz = np.array(peak_frequencies_all).reshape(-1, 1)
    db = DBSCAN(eps=0.02, min_samples=2).fit(freqs_mhz)
    clusters = db.labels_
    grouped_peaks = [np.mean(freqs_mhz[clusters == i]) for i in np.unique(clusters) if np.count_nonzero(clusters == i) > 1]
    grouped_peaks = np.round(sorted(grouped_peaks), 6)
    print(f"\n✅ Seçilen Temiz Peak Frekanslar ({len(grouped_peaks)} adet):")
    for f in grouped_peaks:
        print(f" - {f} MHz")
else:
    print("⚠️ Peak bulunamadı.")
