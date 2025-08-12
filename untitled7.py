# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:36:11 2025

@author: stajyer1
# %%

this code transmits the wav file on tx port and antsdr receives it on rx port. then the original sound and 
received sound are displayed.
then the sound signals are displayed.
"""

import numpy as np
import threading
import adi
import time
import sounddevice as sd
# %%
import soundfile as sf


sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.tx_rf_bandwidth = int(2e6)
sdr.rx_rf_bandwidth = int(2e6)
sdr.tx_lo = int(435e6)
sdr.rx_lo = int(435e6)
sdr.tx_enabled_channels = [0]
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 4096

def tx_func():
    # Ses dosyasını yükle
    filename = r"C:\Users\stajyer1\Downloads\file_example_WAV_1MG.wav"
    audio, samplerate = sf.read(filename)

    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    audio = audio / np.max(np.abs(audio))  # normalize

    # Ses dosyasını SDR sample rate’e uyarlamak için upsample
    interp_factor = int(sdr.sample_rate / samplerate)
    audio_upsampled = np.repeat(audio, interp_factor)

    t = np.arange(len(audio_upsampled)) / sdr.sample_rate
    carrier = np.exp(2j * 2 * np.pi * 1e3 * t)
    am_signal = (1 + 0.8 * audio_upsampled) * carrier

    print("TX başlıyor (WAV dosyası)")
    sdr.tx_cyclic_buffer = False
    sdr.tx(am_signal.astype(np.complex64))
    time.sleep(len(audio_upsampled) / sdr.sample_rate + 0.5)
    sdr.tx_destroy_buffer()
    print("TX durdu")


import time

def rx_func():
    print("RX başlıyor")

    # Ses dosyasının süresini ölç
    filename = r"C:\Users\stajyer1\Downloads\file_example_WAV_1MG.wav"
    audio, samplerate = sf.read(filename)
    duration = len(audio) / samplerate

    audio_fs = 48000
    decim_factor = int(sdr.sample_rate / audio_fs)
    audio_total = []

    start_time = time.time()

    while time.time() - start_time < duration + 1:  # 1 saniye pay
        samples = sdr.rx()
      # AM demodülasyon (faz çözüm)
        audio = np.angle(samples)  # Bu faz açısını verir (-π, π) aralığında
        audio = audio - np.mean(audio)  # DC offset çıkar
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        audio = audio * 0.95
            
        audio_ds = audio[::decim_factor]
        audio_total.extend(audio_ds)

    print("RX bitti. WAV dosyasına yazılıyor...")
    sf.write("loopback_full.wav", np.array(audio_total), samplerate=audio_fs)
    print("Kaydedildi: loopback_full.wav")
    #  d.play(np.array(audio_total), samplerate=audio_fs, blocking=True)


tx_thread = threading.Thread(target=tx_func)
rx_thread = threading.Thread(target=rx_func)  


tx_thread.start()
time.sleep(1)  # TX başlasın
rx_thread.start()

tx_thread.join()
rx_thread.join()
print("TX ve RX threadleri tamamlandı.")



# Orijinal gönderilen WAV dosyası
tx_audio, tx_sr = sf.read(r"C:\Users\stajyer1\Downloads\file_example_WAV_1MG.wav")
print("▶️ Orijinal ses çalınıyor...")
sd.play(tx_audio, samplerate=tx_sr, blocking=True)

# RX'te alınan loopback ses dosyası
rx_audio, rx_sr = sf.read("loopback_full.wav")
print("▶️ Loopback alınan ses çalınıyor...")
sd.play(rx_audio, samplerate=rx_sr, blocking=True)



import matplotlib.pyplot as plt
import numpy as np

# Normalize et ve zaman ekseni ayarla
tx_audio = tx_audio / np.max(np.abs(tx_audio))
rx_audio = rx_audio / np.max(np.abs(rx_audio))

min_len = min(len(tx_audio), len(rx_audio))
tx_audio = tx_audio[:min_len]
rx_audio = rx_audio[:min_len]
time_axis = np.linspace(0, min_len / tx_sr, min_len)

# Grafik
plt.figure(figsize=(12, 4))
plt.plot(time_axis, tx_audio, label="Gönderilen (TX)", alpha=0.7)
plt.plot(time_axis, rx_audio, label="Alınan (RX)", alpha=0.7)
plt.title("AM Ses Dalga Şekli Karşılaştırması (TX vs RX)")
plt.xlabel("Zaman (saniye)")
plt.ylabel("Genlik")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


