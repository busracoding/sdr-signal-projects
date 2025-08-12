import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# --- DLL ve ortam ayarı ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# --- SDR ayarları ---
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

# --- Sweep bölge ayarları ---
start_freq = 70      # MHz
end_freq = 6000      # MHz
region_count = 10
region_width = (end_freq - start_freq) // region_count

peak_frequencies_all = []

# --- Sweep döngüsü ---
for i in range(region_count):
    region_start = start_freq + i * region_width
    region_end = region_start + region_width
    sweep_freqs = list(range(region_start, region_end, 5))

    print(f"\n🔷 Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.figure(figsize=(12, 4))

    for freq in sweep_freqs:
        sdr.rx_lo = int(freq * 1e6)
        samples = sdr.rx()

        N = len(samples)
        T = 1.0 / sdr.sample_rate
        xf = fftfreq(N, T)
        yf = fft(samples)

        xf_mhz = xf[:N//2] / 1e6
        yf_mag = 2.0/N * np.abs(yf[:N//2])
        true_freq = xf_mhz + freq

        # Zayıf sinyalleri komple atla
        if np.max(yf_mag) < 0.01:
            continue

        # Grafik çizimi (saydam)
        plt.plot(true_freq, yf_mag, alpha=0.4)

        # Daha az ve anlamlı peak bul
        peaks, properties = find_peaks(
            yf_mag,
            height=np.max(yf_mag) * 0.5,  # %50 eşik
            distance=3                   # min örnek mesafesi
        )
        peak_freqs = true_freq[peaks]
        peak_frequencies_all.extend(peak_freqs)

    plt.title(f"Bölge {i+1}: {region_start}–{region_end} MHz Spektrum")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 50 kHz içinde peak gruplama (daha sıkı) ---
print("\n🔍 Tüm Tespit Edilen Peak Sayısı (filtrelenmeden):", len(peak_frequencies_all))

if len(peak_frequencies_all) > 0:
    freqs_mhz = np.array(peak_frequencies_all).reshape(-1, 1)
    db = DBSCAN(eps=0.05, min_samples=1).fit(freqs_mhz)  # 50 kHz
    clusters = db.labels_
    grouped_peaks = [np.mean(freqs_mhz[clusters == i]) for i in np.unique(clusters)]

    grouped_peaks = np.round(sorted(grouped_peaks), 6)
    print("✅ Birleştirilmiş Peak Frekanslar (50 kHz içinde gruplanmış):")
    for f in grouped_peaks:
        print(f" - {f} MHz")
else:
    print("⚠️ Peak bulunamadı.")
