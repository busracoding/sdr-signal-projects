# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
"""
AM Modülasyonlu Sweep Sinyali Gönderimi ve Spektral Analizi (435 MHz taşıyıcı)
@author: stajyer1
@date: 2025-07-30
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import adi
import time

# SDR Cihaz Ayarları
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(30.72e6)
sdr.rx_rf_bandwidth = int(30e6)
sdr.tx_rf_bandwidth = int(30e6)
#32768, 4096,1024 buffer size
sdr.rx_buffer_size = 4096 #buffer uzunluğu spektrum çözünürlüğünü(olumlu) ve süreye(olumsuz) göre belirlenebilir
#test frekansı 435MHz 
sdr.tx_lo = int(435e6)
sdr.rx_lo = int(435e6)
sdr.tx_enabled_channels = [0]
sdr.rx_enabled_channels = [0]
sdr.tx_cyclic_buffer = False
sdr.gain_control_mode = "manual"
sdr.rx_hardwaregain_chan0 = 50

# Sinyal Üretimi Ayarları
duration = 0.5  # saniye
fs = sdr.sample_rate
N = int(fs * duration)
t = np.arange(N) / fs

# Sweeping sinüs (4 kHz → 6 kHz)
f0 = 4000  # başlangıç frekansı (Hz)
f1 = 6000  # bitiş frekansı (Hz)
baseband = 0.5 * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)

# AM Modülasyon (baseband + taşıyıcı)
carrier = np.exp(2j * 2 * np.pi * 0 * t)  # taşıyıcı 0 Hz (baseband)
tx_signal = (1 + 0.8 * baseband) * carrier

# TX Gönderimi
print("TX başladı (Sweep sinyali)...")
sdr.tx(tx_signal.astype(np.complex64))
time.sleep(0.05)  # donanıma zaman tanı
sdr.tx_destroy_buffer()
print("TX bitti.")

# RX Alımı
print("RX başlıyor...")
samples = sdr.rx()
samples = samples - np.mean(samples)  # DC offset düzeltmesi

# FFT Analizi
fft_data = fft(samples)
power = 20 * np.log10(np.abs(fftshift(fft_data)) + 1e-3)
power = np.clip(power, a_min=0, a_max=None)
freqs = fftshift(fftfreq(len(samples), 1/fs)) + sdr.rx_lo

# 🎯 FFT Grafiği
plt.figure(figsize=(12, 5))
plt.plot(freqs / 1e6, power, color="royalblue")
plt.title("435 MHz AM Sinyal FFT")
plt.xlabel("Frekans (MHz)")
plt.ylabel("Güç (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Spektrogram (Zaman-Frekans) Analizi
plt.figure(figsize=(12, 4))
plt.specgram(np.real(samples), Fs=fs, NFFT=1024, noverlap=512, cmap="viridis")
plt.title("Zaman-Frekans (Spektrum) Analizi")
plt.xlabel("Zaman (saniye)")
plt.ylabel("Frekans (Hz)")
plt.tight_layout()
plt.show()

"""
Created on Thu Jul 31 10:04:23 2025

@author: stajyer1
"""

