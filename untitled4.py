# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:23:45 2025

@author: stajyer1
"""
#TX CODE, TRANSMIT AUDIOWAVE
import numpy as np
import soundfile as sf
import adi
import time

filename = r"C:\Users\stajyer1\Downloads\file_example_WAV_1MG.wav"
audio, samplerate = sf.read(filename)

if audio.ndim > 1:
    audio = audio[:, 0]
audio = audio / np.max(np.abs(audio))

sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.tx_rf_bandwidth = int(2e6)
sdr.tx_lo = int(435e6)
sdr.tx_enabled_channels = [0]
sdr.tx_cyclic_buffer = False

interp_factor = int(sdr.sample_rate / samplerate)
audio_upsampled = np.repeat(audio, interp_factor)

N = len(audio_upsampled)
t = np.arange(N) / sdr.sample_rate
carrier = np.exp(2j * np.pi * 1e3 * t)
am_signal = (1 + 0.8 * audio_upsampled) * carrier

sdr.tx(am_signal.astype(np.complex64))
print("🔊 Yayın başladı (AM)...")
time.sleep(10)
sdr.tx_destroy_buffer()
print("🛑 Yayın durdu.")


