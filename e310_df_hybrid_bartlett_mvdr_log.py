
# -*- coding: utf-8 -*-
"""
E310 (AD9361) Hybrid DF + Bartlett/MVDR (Stabilized) — with CSV logging
- Lock-in phasor with CFO correction
- Phase & amplitude offset calibration
- Optional amplitude pattern CSV
- Robust averaging, forward-backward covariance, diagonal loading
- Multiple measurements per run with running-median smoothing
- CSV logging of every measurement
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, savgol_filter
import csv, json, os, time
from datetime import datetime
import adi

# ========= USER SETTINGS =========
SDR_URI    = "ip:192.168.2.1"
FS         = 2_000_000                 # sample rate
FC         = 2_400_000_000             # RF
N_SAMP     = 200_000                   # samples per capture
RX_GAINS   = [35, 35]                  # dB
TX_ATTEN   = -30                       # dB
TONE_BB_HZ = 100_000                   # baseband tone for TX
D_LAMBDA   = 0.5                       # element spacing / lambda
CAL_FILE   = "./df_cal.json"
PATTERN_CSV= None                      # e.g., "./pattern_diffgain_vs_angle.csv"
MODE       = "measure"                 # "calibrate" or "measure"

# Stability / averaging
AVERAGES        = 8                    # captures averaged per measurement
MEAS_TIMES      = 5                    # how many measurements to do in a row
SLEEP_BETWEEN_S = 0.15                 # sleep between captures
RUNNING_MED_N   = 5                    # median window over AoA (odd)
CFO_CORRECT     = True
CFO_BLOCKS      = 16
WIN_NAME        = "hann"

# MVDR/Bartlett settings
PLOT_BEAMS      = True
SCAN_RES_DEG    = 0.25                 # scan resolution
LOADING_A       = 1e-2                 # diagonal loading
FORWARD_BACKWARD= True                 # forward-backward covariance averaging

# Logging
LOG_FILE        = "./df_measure_log.csv"

# ========= CONSTANTS =========
c  = 3e8
lam = c/FC
d   = D_LAMBDA * lam

def setup_sdr():
    sdr = adi.ad9361(SDR_URI)
    sdr.sample_rate = int(FS)
    sdr.rx_rf_bandwidth = int(min(FS, 0.8*FS))
    sdr.tx_rf_bandwidth = int(min(FS, 0.8*FS))
    sdr.rx_lo = int(FC)
    sdr.tx_lo = int(FC)
    sdr.rx_enabled_channels = [0, 1]
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(RX_GAINS[0])
    sdr.rx_hardwaregain_chan1 = int(RX_GAINS[1])

    sdr.tx_cyclic_buffer = True
    sdr.tx_enabled_channels = [0]
    sdr.tx_hardwaregain_chan0 = float(TX_ATTEN)

    t = np.arange(N_SAMP)/FS
    tx = 0.5*np.exp(1j*2*np.pi*TONE_BB_HZ*t)
    sdr.tx(tx.astype(np.complex64))

    sdr.rx_buffer_size = N_SAMP
    time.sleep(0.15)
    return sdr

def capture_iq(sdr):
    _ = sdr.rx()
    iq = sdr.rx()
    return iq[0].astype(np.complex64), iq[1].astype(np.complex64)

def phasor_lockin(x, fs, f0, win="hann"):
    N = len(x); t = np.arange(N)/fs
    osc = np.exp(-1j*2*np.pi*f0*t)
    if win:
        w = get_window(win, N, fftbins=True)
        ph = np.sum(x*osc*w)/np.sum(w)
    else:
        ph = np.mean(x*osc)
    return ph

def estimate_cfo(x, fs, f0, blocks=16, win="hann"):
    N = len(x); L = N//blocks
    if L < 64: return 0.0
    phs = []
    for i in range(blocks):
        seg = x[i*L:(i+1)*L]
        phs.append(np.angle(phasor_lockin(seg, fs, f0, win)))
    phs = np.unwrap(np.array(phs))
    dt = L/fs
    slope = np.polyfit(np.arange(blocks)*dt, phs, 1)[0]  # rad/s
    return float(slope/(2*np.pi))

def estimate_tone_phasor_stable(x, fs, f0, win="hann", cfo_correct=True, blocks=16):
    if cfo_correct:
        df = estimate_cfo(x, fs, f0, blocks=blocks, win=win)
        ph = phasor_lockin(x, fs, f0+df, win)
    else:
        df = 0.0; ph = phasor_lockin(x, fs, f0, win)
    return ph, df

def load_cal():
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE, "r") as f:
            return json.load(f)
    return None

def save_cal(cal):
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

def load_pattern_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path): return None
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data

def amp_ratio_to_angle(diff_gain_db, pattern_data):
    ang = pattern_data["angle_deg"]; dif = pattern_data["diff_gain_dB"]
    idx = np.argmin(np.abs(dif - diff_gain_db))
    return float(ang[idx])

def fb_average(R):
    """Forward-backward averaging for 2-element ULA."""
    J = np.array([[0,1],[1,0]])
    return 0.5*(R + J@R.conj()@J)

def estimate_bartlett_mvdr(X_snapshots, scan_deg, loading_alpha=1e-2, fb=True):
    M, K = X_snapshots.shape
    R = (X_snapshots @ X_snapshots.conj().T) / K
    if fb: R = fb_average(R)
    R += np.eye(M, dtype=complex) * (loading_alpha * np.trace(R).real / M)
    Rinv = np.linalg.pinv(R)

    PB, PM = [], []
    for th in np.radians(scan_deg):
        a = np.array([1.0, np.exp(-1j*2*np.pi*d*np.sin(th)/lam)], dtype=np.complex128).reshape(-1,1)
        pb = np.real((a.conj().T @ R @ a).squeeze())
        denom = (a.conj().T @ Rinv @ a).squeeze()
        pm = np.real(1.0 / denom) if np.abs(denom) > 1e-12 else 0.0
        PB.append(pb); PM.append(pm)
    PB = 10*np.log10(np.maximum(np.array(PB), 1e-12)); PB -= PB.max()
    PM = 10*np.log10(np.maximum(np.array(PM), 1e-12)); PM -= PM.max()
    return PB, PM

def ensure_log_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["timestamp","phase_diff_rad","amp_ratio_dB",
                        "CFO0_Hz","CFO1_Hz","phase_AoA_deg",
                        "amp_AoA_deg","hybrid_AoA_deg"])

def log_row(phase_diff, amp_ratio_db, df0, df1, th_phase, th_amp, th_hybrid):
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(timespec='seconds'),
                    f"{phase_diff:.6f}", f"{amp_ratio_db:.3f}",
                    f"{df0:.2f}", f"{df1:.2f}",
                    f"{th_phase:.2f}", "" if th_amp is None else f"{th_amp:.2f}",
                    f"{th_hybrid:.2f}"])

def main():
    sdr = setup_sdr()
    pattern = load_pattern_csv(PATTERN_CSV)
    cal = load_cal()
    phs_of, amp_of = (0.0, 0.0) if cal is None else (float(cal["phase_offset_rad"]), float(cal["amp_offset_db"]))

    if MODE == "calibrate":
        print("[i] Calibration started — point to a known angle.")
        phs_list, amp_list = [], []
        for _ in range(AVERAGES):
            x0, x1 = capture_iq(sdr)
            p0, df0 = estimate_tone_phasor_stable(x0, FS, TONE_BB_HZ, WIN_NAME, CFO_CORRECT, CFO_BLOCKS)
            p1, df1 = estimate_tone_phasor_stable(x1, FS, TONE_BB_HZ, WIN_NAME, CFO_CORRECT, CFO_BLOCKS)
            phs_list.append(np.angle(p1/p0))
            amp_list.append(20*np.log10(np.abs(p0)/np.abs(p1)))
            time.sleep(SLEEP_BETWEEN_S)
        phs_off = float(np.angle(np.mean(np.exp(1j*np.array(phs_list)))))
        amp_off = float(np.mean(amp_list))
        save_cal({"fc":FC,"fs":FS,"d_lambda":D_LAMBDA,
                  "phase_offset_rad":phs_off,"amp_offset_db":amp_off,
                  "timestamp": time.time()})
        print(f"[ok] Saved {CAL_FILE} | phase_off={phs_off:.4f} rad, amp_off={amp_off:.3f} dB")
        return

    ensure_log_header()
    scan_deg = np.arange(-90, 90+SCAN_RES_DEG, SCAN_RES_DEG)
    aoa_series = []

    for m in range(MEAS_TIMES):
        P0, P1, df0s, df1s = [], [], [], []
        for _ in range(AVERAGES):
            x0, x1 = capture_iq(sdr)
            p0, df0 = estimate_tone_phasor_stable(x0, FS, TONE_BB_HZ, WIN_NAME, CFO_CORRECT, CFO_BLOCKS)
            p1, df1 = estimate_tone_phasor_stable(x1, FS, TONE_BB_HZ, WIN_NAME, CFO_CORRECT, CFO_BLOCKS)
            P0.append(p0); P1.append(p1); df0s.append(df0); df1s.append(df1)
            time.sleep(SLEEP_BETWEEN_S)
        P0m = np.mean(np.array(P0)); P1m = np.mean(np.array(P1))
        phase_diff_raw = np.angle(P1m/P0m)
        phase_diff = np.angle(np.exp(1j*(phase_diff_raw - phs_of)))
        amp_ratio_db_raw = 20*np.log10(np.abs(P0m)/np.abs(P1m))
        amp_ratio_db = amp_ratio_db_raw - amp_of

        theta_phase_rad = np.arcsin(np.clip((phase_diff) * lam / (2*np.pi*d), -1.0, 1.0))
        theta_phase_deg = float(np.degrees(theta_phase_rad))

        theta_amp_deg = None
        if pattern is not None:
            theta_amp_deg = amp_ratio_to_angle(amp_ratio_db, pattern)

        theta_hybrid_deg = theta_phase_deg if theta_amp_deg is None else (0.75*theta_phase_deg + 0.25*theta_amp_deg)
        aoa_series.append(theta_hybrid_deg)

        # running median smoothing
        if len(aoa_series) >= RUNNING_MED_N and RUNNING_MED_N % 2 == 1:
            med = float(np.median(aoa_series[-RUNNING_MED_N:]))
        else:
            med = theta_hybrid_deg

        print(f"[{m+1}/{MEAS_TIMES}] AoA phase={theta_phase_deg:+.2f}°, "
              f"hybrid={theta_hybrid_deg:+.2f}° | median={med:+.2f}° | "
              f"CFO0={np.mean(df0s):+.1f} Hz, CFO1={np.mean(df1s):+.1f} Hz")

        log_row(phase_diff, amp_ratio_db, np.mean(df0s), np.mean(df1s),
                theta_phase_deg, theta_amp_deg, theta_hybrid_deg)

    # ---- Plot once using last capture for beam patterns ----
    if PLOT_BEAMS:
        x0, x1 = capture_iq(sdr)
        K = min(8192, len(x0))
        X = np.vstack([x0[:K], x1[:K]])
        PB, PM = estimate_bartlett_mvdr(X, scan_deg, loading_alpha=LOADING_A, fb=FORWARD_BACKWARD)

        th = np.radians(scan_deg + 90)
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(111, projection='polar')
        ax.plot(th, PB, label="Bartlett (norm, dB)")
        ax.plot(th, PM, label="MVDR (norm, dB)")

        def ang2pol(a_deg): return np.radians(a_deg + 90)
        rmin = min(PB.min(), PM.min())
        ax.plot([ang2pol(aoa_series[-1])]*2, [rmin, -1.0], linestyle='--', label=f"Hybrid AoA {aoa_series[-1]:+.1f}°")

        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_title("Bartlett & MVDR (2-eleman ULA) + AoA işaretleri")
        ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.1))
        plt.tight_layout()
        plt.show()

    try: sdr.tx_destroy_buffer()
    except Exception: pass

if __name__ == "__main__":
    main()
