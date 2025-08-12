import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import adi, time

URI = "ip:192.168.2.1"   # gerekirse değiştir

sdr = adi.ad9361(URI)

# RX ayarları
sdr.rx_enabled_channels = [0]          # RX1
sdr.sample_rate = int(1e6)
sdr.rx_lo = int(435e6)
sdr.rx_rf_bandwidth = int(1e6)
sdr.gain_control_mode = "manual"
sdr.rx_hardwaregain_chan0 = 35         # 0–70 dB
sdr.rx_buffer_size = 4096

time.sleep(0.3)                         # PLL/gain otursun
iq = sdr.rx().astype(np.complex64)      # I/Q aldı

# Basit FFT göster
S = fftshift(fft(iq))
f = fftshift(fftfreq(len(iq), 1/sdr.sample_rate))
P = 20*np.log10(np.abs(S)+1e-9)

plt.figure(figsize=(10,4))
plt.plot(f/1e6, P)
plt.title("RX FFT (center = 0 Hz) @ {:.1f} MHz".format(sdr.rx_lo/1e6))
plt.xlabel("Frekans ofseti (MHz)"); plt.ylabel("dB"); plt.grid(True); plt.tight_layout()
plt.show()

sdr.rx_destroy_buffer()
del sdr
