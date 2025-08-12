# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:30:22 2025

@author: stajyer1
"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import adi
import time

# SDR başlat
sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_lo = int(435e6)
sdr.rx_rf_bandwidth = int(2e6)
sdr.gain_control_mode = "manual"
sdr.rx_hardwaregain_chan0 = 50
sdr.rx_buffer_size = 4096

# Audio ayarları
audio_fs = 48000
decim_factor = int(sdr.sample_rate / audio_fs)

print("Bir veri paketi alınıyor...")
samples = sdr.rx()  # IQ (kompleks) veri

# Grafik 1: IQ sinyali
plt.figure(figsize=(10,3))
plt.plot(np.real(samples[:1000]), label='I (Gerçek)')
plt.plot(np.imag(samples[:1000]), label='Q (İmajiner)')
plt.title("RX IQ Sinyali (ilk 1000 örnek)")
plt.legend()
plt.tight_layout()
plt.show()

# AM çözme (genlik detektörü), audio normalizasyonu
audio = np.abs(samples)
audio = audio - np.mean(audio)
audio = audio / np.max(np.abs(audio))  # Normalize et!
audio = audio * 0.8  # Güvenli ses seviyesi (maksimum 1.0)


# Grafik 2: Çözülmüş AM ses
plt.figure(figsize=(10,3))
plt.plot(audio[:1000])
plt.title("AM Çözülmüş Ham Ses (ilk 1000 örnek)")
plt.tight_layout()
plt.show()

# Ses örneklemesini hoparlöre uyarlama
audio_ds = audio[::decim_factor]

print("🎧 Şimdi kulaklığa/hoparlöre çalınıyor...")
sd.play(audio_ds, samplerate=audio_fs, blocking=True)
print("Bitti.")

# İstersen aşağıya while True ekleyip sürekli döngüde dinleyebilirsin.
