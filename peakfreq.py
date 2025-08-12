import os
import ctypes
import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# --- DLL ve PATH ayarı ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

# --- Kullanıcıdan merkez frekans al ---
try:
    user_input = input("🎯 Lütfen merkez frekansı (MHz) girin (örn. 96): ")
    center_freq_mhz = float(user_input)
except:
    print("❗ Hatalı giriş. Varsayılan 96 MHz kullanılıyor.")
    center_freq_mhz = 96.0

# --- Diğer parametreler ---
sample_rate = 2e6       # Örnekleme hızı (Hz) → Gözlem genişliği ≈ 2 MHz
threshold_ratio = 0.3   # Peak için eşik (max'in % kaçı)

# --- SDR Ayarları ---
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_lo = int(center_freq_mhz * 1e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 1024

# --- Veri al ---
samples = sdr.rx()

# --- Zaman düzlemi: I/Q (gerçek) sinyal çizimi ---
plt.figure(figsize=(10, 3))
plt.plot(np.real(samples))
plt.title(f"I/Q Sinyal (Gerçek Kısmı) - Merkez: {center_freq_mhz} MHz")
plt.xlabel("Örnek")
plt.ylabel("Genlik")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- FFT hesaplama ---
N = len(samples)
T = 1.0 / sample_rate
xf = fftfreq(N, T)
yf = fft(samples)

xf_mhz = xf[:N//2] / 1e6
yf_mag = 2.0/N * np.abs(yf[:N//2])
true_freqs = xf_mhz + center_freq_mhz

# --- Spektrum çizimi ---
plt.figure(figsize=(10, 3))
plt.plot(true_freqs, yf_mag)
plt.title(f"Spektrum ({center_freq_mhz - sample_rate/2e6:.1f} - {center_freq_mhz + sample_rate/2e6:.1f} MHz)")
plt.xlabel("Frekans (MHz)")
plt.ylabel("Genlik")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Peak analiz ---
if len(yf_mag) == 0 or np.max(yf_mag) == 0:
    print("⚠️ Sinyal bulunamadı veya tüm genlikler sıfır.")
else:
    threshold = np.max(yf_mag) * threshold_ratio
    peaks, properties = find_peaks(yf_mag, height=threshold)
    peak_freqs = true_freqs[peaks]
    peak_heights = properties['peak_heights']

    plt.figure(figsize=(10, 3))
    plt.plot(true_freqs, yf_mag)
    plt.plot(peak_freqs, peak_heights, "ro")
    plt.title("Peak Frekanslar")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("🔍 Tespit edilen peak frekanslar (MHz):", np.round(peak_freqs, 4))
