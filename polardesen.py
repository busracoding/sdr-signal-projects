# -*- coding: utf-8 -*-
"""
E310 (AD9361) 2-RX Beamforming Starter - Spyder
Yapılanlar:
- SDR bağlan, iki RX kanaldan örnek al
- Splitter testi: iki kanal faz farkı (Δφ) ölç
- 2 elemanlı conventional beamforming (azimut tarama + polar/gain plot)
- Veriyi .npy olarak kaydet/oku (online & offline çalışma)

Gerekenler: pip install pyadi-iio numpy matplotlib scipy
Cihaz erişimi: adi.ad9361("ip:192.168.2.1") gibi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft, fftfreq
import datetime
import os

# ======= KULLANICI PARAMETRELERİ =======
SDR_URI          = "ip:192.168.2.1"   # E310/AD9361 IIO URI (seninkine göre değiştir)
FS               = 2_000_000          # örnekleme (Hz)
FC               = 2_400_000_000      # merkez frekans (Hz) - laboratuvar CW için ayarla
RX_BW            = FS                 # rx bant genişliği
N_SAMPLES        = 200_000            # alınacak örnek sayısı (her kanal)
RX_GAINS         = [40, 40]           # dB (manual gain; AGC kapalı)
SAVE_DIR         = "./captures"       # kayıt klasörü
LOAD_FROM_FILE   = False              # True ise aşağıdaki dosyadan oku, SDR'a bağlanma
LOAD_FILE_PATH   = ""                 # örn: "./captures/iq_2025-08-12_14-10-33.npy"

# Beamforming tarama parametreleri
THETA_DEG_MIN    = -90
THETA_DEG_MAX    =  90
THETA_STEPS      = 721                 # 0.25° çözünürlük için 721 ~ (-90..+90)
ELEMENT_SPACING  = 0.5                 # d/λ (normalize aralık), tipik 0.5 (aliasing yok)
TONE_HZ          = 100_000             # splitter test ve BF için beklenen CW ofseti (Hz). Sinyalin tonunu biliyorsan yaz.
WINDOW_TYPE      = "hann"              # FFT penceresi

# Modlar
DO_SDR_CAPTURE   = True and (not LOAD_FROM_FILE)   # SDR'dan al
DO_SPLITTER_TEST = True                            # Δφ ölç
DO_BF_SCAN       = True                            # tarama yap

# ======= YARDIMCI FONKSİYONLAR =======

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def connect_sdr(uri):
    import adi  # içerde import: Spyder restart'larında sürpriz yaşamamak için
    sdr = adi.ad9361(uri)
    # RX ayarları
    sdr.rx_enabled_channels = [0, 1]              # iki RX
    sdr.sample_rate = int(FS)
    sdr.rx_rf_bandwidth = int(RX_BW)
    sdr.rx_lo = int(FC)
    sdr.gain_control_mode = ["manual", "manual"]
    sdr.rx_hardwaregain_chan0 = RX_GAINS[0]
    sdr.rx_hardwaregain_chan1 = RX_GAINS[1]
    # buffer ayarı
    sdr.rx_buffer_size = int(N_SAMPLES)
    return sdr

def acquire_two_rx(sdr):
    """ iki kanaldan IQ al ve (2, N) döndür """
    x = sdr.rx()  # pyadi-iio bazen (N,) veya list döndürebilir
    # Farklı dönüş tiplerine karşı dayanıklı ol
    if isinstance(x, list) or isinstance(x, tuple):
        # list of arrays
        ch0 = np.array(x[0]).astype(np.complex64)
        ch1 = np.array(x[1]).astype(np.complex64)
    else:
        arr = np.array(x)
        if arr.ndim == 1:
            # tek kanal gelmiş olabilir -> hata
            raise RuntimeError("Sadece tek kanal veri geldi. sdr.rx_enabled_channels = [0,1] olduğundan emin ol.")
        elif arr.ndim == 2:
            # (channels, N) bekleriz
            if arr.shape[0] == 2:
                ch0, ch1 = arr[0].astype(np.complex64), arr[1].astype(np.complex64)
            else:
                raise RuntimeError(f"Beklenmeyen şekil: {arr.shape}")
        else:
            raise RuntimeError(f"Beklenmeyen veri boyutu: {arr.shape}")
    return np.vstack([ch0, ch1])

def save_npy(iq_2xN, meta, out_dir):
    ensure_dir(out_dir)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fpath = os.path.join(out_dir, f"iq_{ts}.npy")
    np.save(fpath, {"iq": iq_2xN, "meta": meta})
    print(f"[KAYIT] {fpath}")
    return fpath

def load_npy(fpath):
    d = np.load(fpath, allow_pickle=True).item()
    return d["iq"], d["meta"]

def estimate_tone_phase(iq, fs, tone_hz, window_type="hann"):
    """
    Tek kanaldaki fazı (radyan) ve genliği tespit et.
    Sinyal: beklenen tek ton (tone_hz) yakınında.
    """
    N = iq.size
    win = get_window(window_type, N, fftbins=True)
    xw = iq * win
    X = fft(xw)
    freqs = fftfreq(N, d=1/fs)
    # pozitif frekans tarafında en yakın bin:
    k = np.argmin(np.abs(freqs - tone_hz))
    ph = np.angle(X[k])
    amp = np.abs(X[k]) / (np.sum(win)/2.0)
    return ph, amp, freqs[k]

def splitter_phase_test(iq2, fs, tone_hz):
    """ iki kanalın faz farkını ölç: Δφ = φ1 - φ0 """
    ph0, a0, f0 = estimate_tone_phase(iq2[0], fs, tone_hz, WINDOW_TYPE)
    ph1, a1, f1 = estimate_tone_phase(iq2[1], fs, tone_hz, WINDOW_TYPE)
    dphi = np.unwrap([ph0, ph1])[1] - np.unwrap([ph0, ph1])[0]
    # -pi..pi normalize
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return dphi, (a0, a1), (f0, f1)

def conventional_bf_scan(iq2, fs, fc, element_spacing_norm, theta_deg_min, theta_deg_max, theta_steps, tone_hz=None):
    """
    2-elemanlı ULA için delay-and-sum (frekansta faz ağırlıkları) tarama.
    element_spacing_norm = d/λ
    """
    # Ton bilirsek dar bant yaklaşımla tek frekans bileşenini çıkarıp güç hesaplamak daha kararlı olur.
    if tone_hz is not None:
        # her kanaldan tonu demodüle et (complex tone remove)
        n = np.arange(iq2.shape[1])
        osc = np.exp(-1j * 2*np.pi * tone_hz * n / fs)
        X = np.vstack([iq2[0]*osc, iq2[1]*osc])
    else:
        X = iq2

    thetas = np.linspace(theta_deg_min, theta_deg_max, theta_steps)
    lam = 3e8 / fc
    d = element_spacing_norm * lam

    # Sadece iki eleman: n = [0, 1]
    n_idx = np.array([0, 1])

    P = np.zeros_like(thetas, dtype=float)
    for i, th_deg in enumerate(thetas):
        th = np.deg2rad(th_deg)
        # steering vector (broadside referans): a(theta) = exp(-j 2π/λ d n sin(theta))
        a = np.exp(-1j * 2*np.pi * (d/lam) * n_idx * np.sin(th))
        w = np.conj(a)  # delay-and-sum
        y = w @ X       # (2,) @ (2,N) -> (N,)
        P[i] = np.mean(np.abs(y)**2)

    # normalize + dB
    P /= (np.max(P) + 1e-12)
    P_dB = 10*np.log10(P + 1e-12)
    return thetas, P_dB

# ======= ANA AKIŞ =======
def main():
    if LOAD_FROM_FILE:
        iq2, meta = load_npy(LOAD_FILE_PATH)
        print(f"[OKU] {LOAD_FILE_PATH} | meta: {meta}")
    else:
        print("[SDR] Bağlanılıyor...")
        sdr = connect_sdr(SDR_URI)
        print("[SDR] Örnek alınıyor...")
        # “flush” için birkaç kez rx() çağırmak iyi olabilir
        _ = sdr.rx()
        iq2 = acquire_two_rx(sdr)  # (2, N)
        meta = dict(fs=FS, fc=FC, rx_bw=RX_BW, gains=RX_GAINS, uri=SDR_URI, n=iq2.shape[1])
        path = save_npy(iq2, meta, SAVE_DIR)

    # Spektrum önizleme
    N = iq2.shape[1]
    win = get_window(WINDOW_TYPE, N, fftbins=True)
    X0 = fft(iq2[0]*win); X1 = fft(iq2[1]*win)
    freqs = fftfreq(N, d=1/FS)

    plt.figure()
    # sadece pozitif yarı
    pos = freqs > 0
    plt.plot(freqs[pos]/1e3, 20*np.log10(np.abs(X0[pos]) + 1e-12), label="CH0")
    plt.plot(freqs[pos]/1e3, 20*np.log10(np.abs(X1[pos]) + 1e-12), label="CH1", linestyle="--")
    plt.title("Hızlı Spektrum Önizleme")
    plt.xlabel("Frekans (kHz)")
    plt.ylabel("|FFT| (dB, görece)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Splitter Test: Δφ
    if DO_SPLITTER_TEST:
        try:
            dphi, amps, fdet = splitter_phase_test(iq2, FS, TONE_HZ)
            print(f"[SPLITTER] Δφ(CH1-CH0) = {np.degrees(dphi):.2f}° | Ton: CH0≈{fdet[0]:.1f} Hz, CH1≈{fdet[1]:.1f} Hz")
        except Exception as e:
            print(f"[Uyarı] Splitter testi başarısız: {e}")

    # BF tarama
    if DO_BF_SCAN:
        thetas, P_dB = conventional_bf_scan(
            iq2, FS, FC, ELEMENT_SPACING,
            THETA_DEG_MIN, THETA_DEG_MAX, THETA_STEPS,
            tone_hz=TONE_HZ
        )

        # Kartezyen plot
        plt.figure()
        plt.plot(thetas, P_dB)
        plt.title("Conventional BF Tarama (2 Eleman)")
        plt.xlabel("Açı (deg)")
        plt.ylabel("Güç (dB, normalize)")
        plt.grid(True)
        plt.tight_layout()

        # Polar plot
        th_rad = np.deg2rad(thetas)
        P_lin = 10**(P_dB/20.0)  # görsel için lineer skala
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.plot(th_rad, P_lin)
        ax.set_theta_zero_location('N')   # 0° yukarı
        ax.set_theta_direction(-1)        # saat yönü artan
        ax.set_title("Polar Desen (Normalize)")
        plt.tight_layout()

    plt.show()
import numpy as np

# Kullanıcı: d/λ ve referans faz (broadside'ta ölçtüğün Δφ_ref)
d_over_lambda = 0.5
phi0_deg = -72.46   # ÖRNEK: senin referans ölçtüğün Δφ (derece). KENDİ DEĞERİNİ YAZ
# Her yeni ölçümde scriptin verdiği Δφ(deg):
phi_meas_deg = -117.72  # ÖR: yeni bir deneme

# Hesap
phi0 = np.deg2rad(phi0_deg)
phi  = np.deg2rad(phi_meas_deg)

# φ0'ı düş, sonra [-π,π] sarmala (güvenli hesap için)
phi_adj = np.angle(np.exp(1j*(phi - phi0)))

# AoA (arcsin bağıntısı). |phi_adj| <= 2π d/λ olmalı
theta_est_rad = np.arcsin( phi_adj / (2*np.pi*d_over_lambda) )
theta_est_deg = np.degrees(theta_est_rad)
print(f"θ̂ ≈ {theta_est_deg:.1f}°")

if __name__ == "__main__":
    main()
