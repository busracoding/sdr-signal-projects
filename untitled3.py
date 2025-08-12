# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:36:09 2025

@author: stajyer1
"""

import adi
import numpy as np
import matplotlib.pyplot as plt

# Cihaza bağlan (USB bağlantıdaysa)
sdr = adi.ad9361("ip:192.168.2.1")  # ya da "usb:1.29.5" olabilir

# Ortak ayarlar
sdr.sample_rate = int(1e6)
sdr.rx_lo = int(435e6)  # Loopback frekansı
sdr.tx_lo = int(435e6)

sdr.tx_cyclic_buffer = True
sdr.tx_rf_bandwidth = int(1e6)
sdr.rx_rf_bandwidth = int(1e6)
sdr.gain_control_mode_chan0 = "slow_attack"
sdr.rx_enabled_channels = [0]
sdr.tx_enabled_channels = [0]
sdr.tx_hardwaregain_chan0 = -10  # düşük kazanç

# 1kHz sinüs dalgası üret
fs = int(sdr.sample_rate)
t = np.arange(1024) / fs
signal = 0.5 * np.exp(2j * np.pi * 1e3 * t)  # 1 kHz complex sinyal

# Yayına başla
sdr.tx(signal)

# RX ile oku (loopback varsa)
samples = sdr.rx()

# Görüntüle
plt.figure()
plt.psd(samples, NFFT=1024, Fs=fs / 1e6)
plt.title("RX Spektrumu (Loopback Testi)")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.show()
