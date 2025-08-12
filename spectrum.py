import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# DLL yükle
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# SDR kur
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

frequencies = list(range(70, 110, 5))  # MHz cinsinden sweep (70, 75, 80, ..., 105 MHz)

for freq in frequencies:
    print(f"\n📡 Frekans: {freq} MHz")

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

    # Çiz
    plt.figure(figsize=(8, 3))
    plt.plot(xf_mhz, yf_mag)
    plt.title(f"Spektrum: {freq} MHz LO (Görünen Aralık: {freq - 1:.0f} – {freq + 1:.0f} MHz)")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


"""belli bi frekans aralığını tarayıp veri sağlar 5Mhz aralıklarla, 0-2Mhz farkla"""