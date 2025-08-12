# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft, fftfreq
import adi, time

# ========= KULLANICI AYARI =========
SDR_URI   = "ip:192.168.2.1"
FS        = 2_000_000
FC        = 2_400_000_000      # 2.4 GHz örnek
N         = 200_000
RX_GAINS  = [35, 35]           # ikisini de aynı tut
TONE_HZ   = 100_000            # TX ton ofseti (FC ± 100 kHz)
D_LAMBDA  = 0.5                # d/λ ~ 0.5 (6–7 cm @ 2.4 GHz)
DO_TX     = True               # kontrollü CW istiyorsan True

# ========= TX: kontrollü CW (opsiyonel) =========
# TX'i FC ± TONE_HZ'de taşıyıcı + ton gönder
def start_tx(uri, fs, fc, tone_hz, tx_gain_db= -10):
    tx = adi.ad9361(uri)
    tx.tx_enabled_channels = [0]           # TX1A
    tx.tx_lo = int(fc)
    tx.sample_rate = int(fs)
    tx.tx_rf_bandwidth = int(fs)
    # Pluto/AD9361'de tx gain negatif dB skalası olabilir
    try:
        tx.tx_hardwaregain_chan0 = tx_gain_db
    except Exception:
        pass
    tx.tx_cyclic_buffer = True

    Ntx = 8192
    n = np.arange(Ntx)
    sig = np.exp(1j*2*np.pi*tone_hz*n/fs)  # saf kompleks ton
    # güç sınırlama
    sig = (sig * 0.3).astype(np.complex64)
    tx.tx(sig)
    return tx

# ========= RX: bağlan & al =========
def acquire_rx(uri, fs, fc, n, gains):
    sdr = adi.ad9361(uri)
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.rx_lo = int(fc)
    # Portları A'ya zorla (A_BALANCED). Eğer hata verirse yorum satırı yap.
    try:
        sdr.rx_rf_port = ["A_BALANCED", "A_BALANCED"]
    except Exception:
        try:
            sdr.rx_rf_port = "A_BALANCED"
        except Exception:
            pass
    sdr.gain_control_mode = ["manual","manual"]
    sdr.rx_hardwaregain_chan0 = gains[0]
    sdr.rx_hardwaregain_chan1 = gains[1]
    sdr.rx_buffer_size = int(n)

    # flush
    _ = sdr.rx(); _ = sdr.rx()
    x = sdr.rx()
    if isinstance(x, (list, tuple)):
        x0 = np.asarray(x[0]).astype(np.complex64)
        x1 = np.asarray(x[1]).astype(np.complex64)
    else:
        arr = np.asarray(x)
        assert arr.shape[0] == 2, f"Beklenmeyen şekil: {arr.shape}"
        x0, x1 = arr[0].astype(np.complex64), arr[1].astype(np.complex64)
    return x0, x1

def find_peak_bin(x, fs, guess_hz, search_bw_hz=20000):
    # yakın çevrede en güçlü bin’i bul
    N = x.size
    win = get_window("hann", N, fftbins=True)
    X = fft(x*win)
    f = fftfreq(N, d=1/fs)
    # arama bölgesi
    mask = (f>0) & (np.abs(f - guess_hz) < search_bw_hz/2)
    if not np.any(mask):
        return None, None, None
    idx = np.argmax(np.abs(X[mask]))
    abs_idx = np.where(mask)[0][idx]
    return abs_idx, f[abs_idx], X[abs_idx]

def main():
    tx = None
    if DO_TX:
        tx = start_tx(SDR_URI, FS, FC, TONE_HZ, tx_gain_db=-20)
        time.sleep(0.2)

    x0, x1 = acquire_rx(SDR_URI, FS, FC, N, RX_GAINS)
    rms0 = np.sqrt(np.mean(np.abs(x0)**2))
    rms1 = np.sqrt(np.mean(np.abs(x1)**2))
    print(f"[INFO] N={N}, CH0 RMS={rms0:.2f}, CH1 RMS={rms1:.2f}")

    # Zoom FFT ve ton kontrolü
    idx0, f0, X0pk = find_peak_bin(x0, FS, TONE_HZ, search_bw_hz=40000)
    idx1, f1, X1pk = find_peak_bin(x1, FS, TONE_HZ, search_bw_hz=40000)
    if (idx0 is None) or (idx1 is None):
        print("[UYARI] Beklenen ton çevresinde güçlü pik bulunamadı. FC/TONE_HZ/tx konumunu kontrol et.")
    else:
        print(f"[TON] CH0≈{f0:.1f} Hz, CH1≈{f1:.1f} Hz (beklenen {TONE_HZ} Hz)")

    # Korelasyon ve kompleks ölçek (dar bant: tonu çıkart)
    n = np.arange(N)
    osc = np.exp(-1j*2*np.pi*TONE_HZ*n/FS)
    y0, y1 = x0*osc, x1*osc

    # alpha: ch1 ≈ alpha*ch0
    alpha = (np.vdot(y0, y1) / (np.vdot(y0, y0) + 1e-12))
    corr_mag = np.abs(np.vdot(y0, y1)) / (np.linalg.norm(y0)*np.linalg.norm(y1) + 1e-12)
    print(f"[TEŞHİS] |alpha|={np.abs(alpha):.3f}, ∠alpha={np.degrees(np.angle(alpha)):.2f}°, |corr|={corr_mag:.3f}")

    # Eğer RMS çok farklıysa, kazancı eşitle:
    if rms0 > 0 and rms1/rms0 > 5:
        print("[NOT] CH1 çok daha güçlü görünüyor. TX konumu/kazanç/kablo/portu kontrol et; RX gainleri eşit.")

    # BF tarama
    lam = 3e8/FC
    d = D_LAMBDA * lam
    thetas = np.linspace(-90, 90, 721)
    nidx = np.array([0,1])

    # kalibrasyon: ch1'i alpha ile hizala
    X = np.vstack([y0, y1 * np.conj(alpha)])
    P = []
    for th in np.deg2rad(thetas):
        a = np.exp(-1j*2*np.pi*(d/lam)*nidx*np.sin(th))
        y = np.conj(a) @ X
        P.append(np.mean(np.abs(y)**2))
    P = np.array(P); P /= P.max()+1e-12
    peak_deg = thetas[np.argmax(P)]
    print(f"[BF] Tepe açı ≈ {peak_deg:.1f}°")

    # Görseller
    # 1) Zoom FFT
    Nfft = 32768
    win = get_window("hann", Nfft, fftbins=True)
    def zoom_fft(x):
        X = fft(x[:Nfft]*win)
        f = fftfreq(Nfft, d=1/FS)
        sel = (f>0) & (f<300000)
        return f[sel], 20*np.log10(np.abs(X[sel])+1e-12)
    fz0, Az0 = zoom_fft(x0); fz1, Az1 = zoom_fft(x1)
    plt.figure(); plt.plot(fz0/1e3, Az0, label="CH0"); plt.plot(fz1/1e3, Az1, '--', label="CH1")
    plt.axvline(TONE_HZ/1e3, linestyle=':', label="Beklenen ton")
    plt.title("Zoom FFT (0–300 kHz)"); plt.xlabel("kHz"); plt.ylabel("dB"); plt.grid(True); plt.legend(); plt.tight_layout()

    # 2) BF eğrisi
    plt.figure(); plt.plot(thetas, 10*np.log10(P+1e-12)); plt.grid(True)
    plt.title("Conventional BF (2 eleman)"); plt.xlabel("Açı (deg)"); plt.ylabel("dB"); plt.tight_layout()

    plt.show()

    if tx is not None:
        # TX açık kalabilir; istersen aşağıyı açıp durdur
        # tx.tx_destroy_buffer()
        # del tx
        pass

if __name__ == "__main__":
    main()
