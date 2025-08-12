# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:13:51 2025
# %%

# %%

E310’dan iki RX kanalı (iki anten) ile kompleks IQ alıyoruz.

TX’ten gönderdiğin tek ton (TONE_HZ) etrafında dar-bant çalışmak için sinyali demodüle ediyoruz (tonu DC’ye çekiyoruz).

Bir KERE yapılan kalibrasyonla (bf_cal.json) kablo/kanal sabit faz + genlik farkını düzeltiyoruz; böylece mekânsal faz farkı (AoA bilgisi) korunuyor.

Δφ → 
𝜃
θ dönüşümünden AoA (geliş açısı) kestiriyoruz.

Aynı veriden 2-elemanlı conventional (delay‑and‑sum) beamforming taraması yapıp, ışıma deseninde tepe açıyı buluyoruz.

(Senkron doğruysa) AoA ile BF tepe yakın çıkar; işaret farkı sadece eksen/kanal yönü konvansiyonu.
@author: stajyer1
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft, fftfreq
import json, os, time
import adi

# ======= KULLANICI =======
SDR_URI   = "ip:192.168.2.1"
FS        = 2_000_000
FC        = 2_400_000_000
N         = 200_000
RX_GAINS  = [35, 35]
TONE_HZ   = 100_000
D_LAMBDA  = 0.5              # ~6–7 cm @ 2.4 GHz
CAL_FILE  = "./bf_cal.json"
# %%
calc_mode = "measure"        # önce "calibrate" çalıştır, sonra "measure"


# ======= YARDIMCI =======
def sdr_capture(uri, fs, fc, n, gains):
    sdr = adi.ad9361(uri)
    sdr.rx_enabled_channels = [0,1]
    sdr.sample_rate = int(fs)
    sdr.rx_rf_bandwidth = int(fs)
    sdr.rx_lo = int(fc)
    try:
        sdr.gain_control_mode = ["manual","manual"]
    except:
        sdr.gain_control_mode = "manual"
    sdr.rx_hardwaregain_chan0 = gains[0]
    sdr.rx_hardwaregain_chan1 = gains[1]
    sdr.rx_buffer_size = int(n)
    _ = sdr.rx(); _ = sdr.rx()
    x = sdr.rx()
    if isinstance(x, (list,tuple)):
        ch0 = np.asarray(x[0]).astype(np.complex64)
        ch1 = np.asarray(x[1]).astype(np.complex64)
    else:
        arr = np.asarray(x)
        ch0, ch1 = arr[0].astype(np.complex64), arr[1].astype(np.complex64)
    return ch0, ch1

def zoom_tone_check(x, fs, guess, bw=40000):
    N = x.size
    win = get_window("hann", N, fftbins=True)
    X = fft(x*win); f = fftfreq(N, d=1/fs)
    mask = (f>0) & (np.abs(f-guess) < bw/2)
    if not np.any(mask): return None, None
    idx = np.argmax(np.abs(X[mask])); k = np.where(mask)[0][idx]
    return f[k], X[k]

def save_cal(phi_cal_deg, amp_ratio):
    d = {"phi_cal_deg": float(phi_cal_deg), "amp_ratio": float(amp_ratio)}
    with open(CAL_FILE, "w") as f: json.dump(d, f, indent=2)
    print(f"[CAL] yazıldı -> {CAL_FILE}: {d}")

def load_cal():
    if not os.path.exists(CAL_FILE):
        raise RuntimeError("Kalibrasyon dosyası yok. Önce calc_mode='calibrate' çalıştır.")
    with open(CAL_FILE) as f: d = json.load(f)
    return np.deg2rad(d["phi_cal_deg"]), d["amp_ratio"]

# ======= ANA =======
def main():
    ch0, ch1 = sdr_capture(SDR_URI, FS, FC, N, RX_GAINS)
    print(f"[INFO] N={N}, RMS0={np.sqrt(np.mean(np.abs(ch0)**2)):.2f}, RMS1={np.sqrt(np.mean(np.abs(ch1)**2)):.2f}")

    # Tonu demodüle et
    n = np.arange(N); osc = np.exp(-1j*2*np.pi*TONE_HZ*n/FS)
    y0, y1 = ch0*osc, ch1*osc

    # ---- MODE 1: KALİBRASYON (broadside ≈ 0° iken bir kez) ----
    if calc_mode.lower()=="calibrate":
        # sabit faz ofsetini ölç (donanım + kablo), AoA'nın kendisi ~0° kabul
        alpha = (np.vdot(y0, y1) / (np.vdot(y0, y0)+1e-12))
        phi_cal = np.angle(alpha)     # rad
        amp_ratio = np.abs(alpha)     # isteğe bağlı genlik düzeltme
        save_cal(np.degrees(phi_cal), amp_ratio)
        print("[CAL] Tamam. Şimdi calc_mode='measure' yapıp farklı açılarda ölç.")
        return

    # ---- MODE 2: ÖLÇÜM (AoA korunur) ----
    phi_cal, amp_ratio = load_cal()       # sabit!
    # Sadece SABİT kalibrasyonu uygula: ch1'i sabit faz/gain ile düzelt
    y1c = y1 * np.exp(-1j*phi_cal) / (amp_ratio + 1e-12)

    # Doğrudan Δφ ve AoA tahmini (kontrol amaçlı)
    # NOT: Burada anlık alpha HESAPLAMAYACAĞIZ, AoA'yı öldürür.
    # Δφ ≈ arg( sum(y1c * conj(y0)) )
    delta_phi = np.angle(np.vdot(y0, y1c))
    d_over_lambda = D_LAMBDA
    # arcsin argümanını güvene al
    arg = delta_phi / (2*np.pi*d_over_lambda)
    arg = np.clip(arg, -1.0, 1.0)
    theta_est = np.degrees(np.arcsin(arg))
    print(f"[AoA] Δφ={np.degrees(delta_phi):.2f}°, θ̂≈{theta_est:.1f}°")

    # Beamforming tarama (conventional)
    lam = 3e8/FC; d = D_LAMBDA * lam
    thetas = np.linspace(-90, 90, 721)
    nidx = np.array([0,1])
    X = np.vstack([y0, y1c])     # SADECE sabit kalibreli veriler

    P = []
    for th in np.deg2rad(thetas):
        a = np.exp(-1j*2*np.pi*(d/lam)*nidx*np.sin(th))
        y = np.conj(a) @ X
        P.append(np.mean(np.abs(y)**2))
    P = np.array(P); P /= P.max()+1e-12
    peak_idx = np.argmax(P); peak_deg = thetas[peak_idx]
    # debug: ilk 3 tepeyi yaz
    tops = np.argsort(P)[-3:][::-1]
    print("[BF] tepe(1)={:.1f}°, tepe(2)={:.1f}°, tepe(3)={:.1f}°".format(*thetas[tops]))

    # Plot
    plt.figure(); plt.plot(thetas, 10*np.log10(P+1e-12)); plt.grid(True)
    plt.title("Conventional BF (sabit kalibrasyon)"); plt.xlabel("Açı (deg)"); plt.ylabel("dB"); plt.tight_layout()

    # Polar
    plt.figure(); ax = plt.subplot(111, projection='polar')
    ax.plot(np.deg2rad(thetas), 10**( (10*np.log10(P+1e-12))/20 ))
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_title("Polar Desen"); plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
