import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import adi
import time

# SDR ayarları
sdr = adi.ad9361("ip:192.168.2.1")

# RX ayarları
sdr.rx_enabled_channels = [0]
sdr.sample_rate = int(1e6)
sdr.rx_lo = int(435e6)
sdr.rx_rf_bandwidth = int(1e6)
sdr.gain_control_mode = "manual"
sdr.rx_hardwaregain_chan0 = 40  # 0–70 arası; dikkatli yükselt

# Alım başlat
sdr.rx_buffer_size = 4096
time.sleep(0.5)  # Stabilizasyon için bekle

samples = sdr.rx()  # Kompleks veri (I/Q)

# Zaman domeni çizim
plt.figure(figsize=(10, 4))
plt.plot(np.real(samples), label="I")
plt.plot(np.imag(samples), label="Q")
plt.title("Zaman Domeninde RX Sinyali")
plt.xlabel("Örnek")
plt.ylabel("Genlik")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Frekans domeni (FFT)
N = len(samples)
freqs = fftfreq(N, 1 / sdr.sample_rate)
spectrum = 20 * np.log10(np.abs(fft(samples)) + 1e-6)

plt.figure(figsize=(10, 4))
plt.plot(freqs / 1e6, spectrum)
plt.title("Frekans Domeninde RX Sinyali (FFT)")
plt.xlabel("Frekans (MHz)")
plt.ylabel("Genlik (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()
