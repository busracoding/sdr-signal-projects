import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- DLL yükle ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# --- SDR ayarları ---
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

# --- 3 Bölge: 1 MHz adım aralığı ile sweep yapılacak ---
regions = {
    "Bölge 1 (80–120 MHz)": list(range(80, 121, 1)),
    "Bölge 2 (120–160 MHz)": list(range(120, 161, 1)),
    "Bölge 3 (160–200 MHz)": list(range(160, 201, 1)),
}

for region_name, frequencies in regions.items():
    print(f"\n🟪 {region_name}")
    plt.figure(figsize=(10, 4))

    for freq in frequencies:
        print(f"  📡 Sweep frekansı: {freq} MHz")
        sdr.rx_lo = int(freq * 1e6)
        samples = sdr.rx()

        N = len(samples)
        T = 1.0 / sdr.sample_rate
        xf = fftfreq(N, T)
        yf = fft(samples)

        xf_mhz = xf[:N//2] / 1e6
        yf_mag = 2.0/N * np.abs(yf[:N//2])
        true_freq = xf_mhz + freq

        # Saydam çizim, etiket yok → sade spektrum
        plt.plot(true_freq, yf_mag, alpha=0.4)

    # --- Grafik ayarları ---
    plt.title(f"{region_name} – Toplu Spektrum (1 MHz sweep)")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
