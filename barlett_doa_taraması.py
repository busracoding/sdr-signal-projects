# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:46:34 2025

@author: stajyer1
"""

# doa_antsdr.py
# ANTSDR E310 (AD9361) ile 2 Rx DOA:
# - Faz-farkı (tek atış) açısı
# - Bartlett (delay-and-sum) taraması
# - Opsiyonel: Tx ile kalibrasyon tonu üretimi (lab kullanımına uygun)

import numpy as np
import matplotlib.pyplot as plt
import time
import math

try:
    import adi  # pyadi-iio
except ImportError:
    raise SystemExit("pyadi-iio gerekli: pip install pyadi-iio")

# ===================== KULLANICI AYARLARI =====================

SDR_IP           = "ip:192.168.2.1"     # E310/ANTSDR IP
CENTER_FREQ_HZ   = int(2.4e9)           # merkez RF (Hz) - ör: 2.4 GHz
SAMPLE_RATE_SPS  = int(2e6)             # örnekleme hızı
RX_BW_HZ         = int(2e6)             # RX analog bant genişliği
RX_GAIN_MODE     = "manual"             # "manual" ya da "slow_attack"
RX_GAIN_DB       = 30                   # manual mod için
BUF_SIZE         = 8192                 # donanım buffer (2^N)
N_BLOCKS         = 20                   # toplanacak blok sayısı (toplam N ≈ BUF_SIZE*N_BLOCKS)

# Anten aralığı ve d_norm ayarı
ANT_SPACING_M    = 0.0625               # antenler arası mesafe (metre) ör: ~6.25 cm
C_LIGHT          = 299792458.0          # m/s
WAVELENGTH_M     = C_LIGHT / CENTER_FREQ_HZ
D_NORM           = ANT_SPACING_M / WAVELENGTH_M

# Tarama ve grafik ayarları
THETA_MIN_DEG    = -90
THETA_MAX_DEG    =  90
N_THETA          = 721                  # çözünürlük
PLOT_SAMPLES     = 200                  # zaman grafiğinde gösterilecek örnek sayısı

# (Opsiyonel) TX kalibrasyon tonu – SADECE LAB/İZİNLE!
USE_TX_TONE      = False                # True yaparsan Tx yayın açılır ⚠️
TX_TONE_HZ       = int(50e3)            # baseband tonu
TX_BW_HZ         = int(2e6)
TX_GAIN_DB       = -20                  # çıkış gücü (dB) – küçük tut
TX_TONE_AMPL     = 0.25                 # I/Q genliği (0..1)

# ===============================================================


def configure_rx(sdr):
    sdr.sample_rate = SAMPLE_RATE_SPS
    sdr.rx_rf_bandwidth = RX_BW_HZ
    sdr.rx_lo = CENTER_FREQ_HZ

    # 2 Rx (I0Q0 ve I1Q1)
    sdr.rx_enabled_channels = [0, 1]
    sdr.rx_buffer_size = BUF_SIZE

    if RX_GAIN_MODE == "manual":
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
        sdr.rx_hardwaregain_chan1 = RX_GAIN_DB
    else:
        sdr.gain_control_mode_chan0 = "slow_attack"
        sdr.gain_control_mode_chan1 = "slow_attack"

def configure_tx(sdr):
    sdr.tx_enabled_channels = [0]               # tek çıkış yeterli
    sdr.tx_lo = CENTER_FREQ_HZ
    sdr.tx_rf_bandwidth = TX_BW_HZ
    sdr.tx_hardwaregain_chan0 = TX_GAIN_DB
    # temel ton oluştur
    N = BUF_SIZE
    t = np.arange(N)/SAMPLE_RATE_SPS
    tone = TX_TONE_AMPL * np.exp(2j*np.pi*TX_TONE_HZ*t)
    sdr.tx_cyclic_buffer = True
    sdr.tx(tone)  # cyclic buffer ile sürekli döner

def capture_blocks(sdr, n_blocks=N_BLOCKS):
    # ilk okuma flush olabilir; bir kez okuyup atmak iyi olur
    _ = sdr.rx()
    time.sleep(0.01)

    X_list = []
    for _ in range(n_blocks):
        d = sdr.rx()          # [ch0, ch1] tuple veya np.array gelebilir
        # pyadi-iio genelde tek array döndürür: shape (2, BUF_SIZE)
        x = np.vstack(d) if isinstance(d, (list, tuple)) else d
        X_list.append(x)
    X = np.hstack(X_list)     # (2, N)
    return X

def preprocess(X):
    # DC ofset kaldır + kanal ölçek eşitleme
    X = X - X.mean(axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-12
    X = X / std
    return X

def doa_phase_diff(X, d_norm=D_NORM):
    # Faz farkına dayalı tek kaynak açısı
    phi = np.angle(X[0,:] * np.conj(X[1,:]))      # [-pi,pi]
    phi_u = np.unwrap(phi)
    dphi  = np.mean(phi_u)
    s     = np.clip(dphi / (2*np.pi*d_norm), -1.0, 1.0)
    theta = np.arcsin(s)
    return np.degrees(theta)

def bartlett_scan(X, d_norm=D_NORM, n_theta=N_THETA,
                  th_min_deg=THETA_MIN_DEG, th_max_deg=THETA_MAX_DEG):
    thetas = np.linspace(np.radians(th_min_deg), np.radians(th_max_deg), n_theta)
    P = np.empty_like(thetas)
    Nr = X.shape[0]
    for i, th in enumerate(thetas):
        s = np.exp(-2j*np.pi*d_norm*np.array([0,1])*np.sin(th))
        w = s / Nr
        y = np.conj(w) @ X
        P[i] = np.var(y)  # güç metriği
    P_db = 10*np.log10(np.maximum(P, 1e-16))
    P_db -= P_db.max()
    return thetas, P_db

def main():
    print(f"[i] λ = {WAVELENGTH_M*100:.2f} cm, d = {ANT_SPACING_M*100:.2f} cm, d_norm = {D_NORM:.3f}")
    sdr = adi.ad9361(SDR_IP)

    try:
        configure_rx(sdr)

        if USE_TX_TONE:
            print("[!] TX AÇILIYOR (sadece izinli/lab ortamında kullan!)")
            configure_tx(sdr)
            time.sleep(0.2)  # LO/AGC yerleşsin

        print("[i] Veri toplanıyor...")
        X = capture_blocks(sdr)
        print(f"[i] X shape: {X.shape} (kanal x örnek)")

        X = preprocess(X)

        # Faz farkı DOA
        theta_phase = doa_phase_diff(X)
        print(f"[✓] Faz-farkı DOA: {theta_phase:.2f}°")

        # Bartlett taraması
        thetas, P_db = bartlett_scan(X)
        theta_bart = np.degrees(thetas[np.argmax(P_db)])
        print(f"[✓] Bartlett tepe açısı: {theta_bart:.2f}°")

        # Zaman domeni (gerçek kısım) – ayrı figür
        plt.figure()
        plt.plot(np.real(X[0, :PLOT_SAMPLES]), label="CH0 (real)")
        plt.plot(np.real(X[1, :PLOT_SAMPLES]), label="CH1 (real)")
        plt.xlabel("Örnek")
        plt.ylabel("Genlik")
        plt.title("İlk örnekler – 2 Rx (normalize)")
        plt.legend(); plt.grid(True)
        plt.show()

        # Bartlett eğrisi – ayrı figür
        plt.figure()
        plt.plot(np.degrees(thetas), P_db)
        plt.axvline(theta_bart, linestyle="--")
        plt.xlabel("Açı (derece)")
        plt.ylabel("Bartlett gücü (dB, norm.)")
        plt.title("Bartlett DOA taraması – 2 Rx")
        plt.grid(True)
        plt.show()

    finally:
        # TX açıksa kapat
        if USE_TX_TONE:
            sdr.tx_destroy_buffer()
        del sdr

if __name__ == "__main__":
    main()
