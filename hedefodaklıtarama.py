# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:54:45 2025

@author: stajyer1
"""

import os
import ctypes
import adi
import time
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, firwin, lfilter, get_window
from sklearn.cluster import DBSCAN



# --- DLL ve SDR ayarı ---
os.environ["PATH"] += os.pathsep + "C:/Program Files/IIO Oscilloscope/bin"
ctypes.cdll.LoadLibrary("C:/Program Files/IIO Oscilloscope/bin/libiio.dll")

sdr = adi.ad9361("ip:192.168.2.1")
sdr.sample_rate = int(2e6)
sdr.rx_enabled_channels = [0]
sdr.rx_buffer_size = 2048  # FFT çözünürlüğü için artırıldı

# Filter bank parametreleri
n_bands = 8
band_width = (sdr.sample_rate // 2) // n_bands  # Toplam bant genişliği fs/2
filter_order = 128
#burda windowing yapıldı, hann window kullanıldı
window_type = 'hann'  
#frekans aralığı antsdr ile uyumlu olsun diye 70MHz-6GHz seçildi
start_freq = 70
end_freq = 6000
region_count = 10
region_width = (end_freq - start_freq) // region_count
peak_frequencies_all = []



#bu döngü 70-6k aralığında alınan verileri tarayıp örnekliyor (Sample) 
#önce her alt bandı (yaklaşık 600MHz lik bandlar) kendi içinde tarıyor 
#sonra FIR bandpass filtresi uyguluyor 
#Filter-bank + windowed FFT yöntemi uygulandı, sebebi daha çok aralığa bakmak ve detaylı veri analiz etmek
#polyphase fft filter bank ile daha hızlı olabilir ancak simulasyon için bu tercih edildi, geliştirilebilir kısmı burada
#daha hızlı veri analizi için fpga&python birlikte kullanılabilir, pythonda tüm verileri taraması uzun sürüyor
for i in range(region_count):
    region_start = start_freq + i * region_width
    region_end = region_start + region_width
    sweep_freqs = list(range(region_start, region_end, 5))

    print(f"\n🟣 Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.figure(figsize=(12, 6))
    max_power_peaks = []

    for freq in sweep_freqs:
        sdr.rx_lo = int(freq * 1e6)
        samples = sdr.rx()
        samples = samples - np.mean(samples)  # DC offset gider
        time.sleep(0.05)  # 50ms bekle, cihaz LO'yu sabitlesin (veri kaybolması hatasını gidermek için )
        # Her alt bandı tek tek analiz et
        for b in range(n_bands):
            low = b * band_width
            high = low + band_width

            # Düşük sınır sıfır olamaz, en az 1 Hz olsun
            if low == 0:
                low = 1

            # Yüksek sınır fs/2'yi aşamaz
            if high >= sdr.sample_rate // 2:
                high = (sdr.sample_rate // 2) - 1

            # Hatalı bandı atla
            if low >= high:
                continue

            low_hz = low
            high_hz = high

            # FIR bandpass filter tasarla
            taps = firwin(
                filter_order, [low_hz / (sdr.sample_rate/2), high_hz / (sdr.sample_rate/2)],
                pass_zero=False, window=window_type
            )
            filtered = lfilter(taps, 1.0, samples)

            # Pencere uygula
            window = get_window(window_type, len(filtered))
            windowed = filtered * window

            # FFT
            N = len(windowed)
            T = 1.0 / sdr.sample_rate
            xf = fftfreq(N, T)
            yf = fft(windowed)
            xf_mhz = xf[:N//2] / 1e6 + freq + (low_hz / 1e6)  # Frekans kaymasını da ekle
            yf_mag = 2.0 / N * np.abs(yf[:N//2])

            # Çok düşük genlikli bandları atla
            if np.max(yf_mag) < 0.01:
                continue

            plt.plot(xf_mhz, yf_mag, alpha=0.18, label=f"Band {b+1}" if freq == sweep_freqs[0] else None)

            # Peak bul ve kaydet
            peaks, properties = find_peaks(
                yf_mag,
                height=np.max(yf_mag) * 0.7,
                distance=10,
                prominence=1
            )
            peak_freqs = xf_mhz[peaks]
            peak_heights = properties['peak_heights']
            strong_peaks = peak_freqs[peak_heights > 0.01]
            peak_frequencies_all.extend(strong_peaks)
            # En güçlü peak frekansını kaydet (en yüksek genlikli olan)

    plt.title(f"Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- DBSCAN ile peak gruplama ---
print("\n📉 Toplam ham peak:", len(peak_frequencies_all))

if len(peak_frequencies_all) > 0:
    freqs_mhz = np.array(peak_frequencies_all).reshape(-1, 1)
    db = DBSCAN(eps=0.02, min_samples=2).fit(freqs_mhz)  # 20 kHz, en az 2 sinyal

    clusters = db.labels_
    grouped_peaks = [
        np.mean(freqs_mhz[clusters == i])
        for i in np.unique(clusters)
        if np.count_nonzero(clusters == i) > 1
    ]

    grouped_peaks = np.round(sorted(grouped_peaks), 6)

    print(f"\n✅ Seçilen Temiz Peak Frekanslar ({len(grouped_peaks)} adet):")
    for f in grouped_peaks:
        print(f" - {f} MHz")
else:
    print("⚠️ Peak bulunamadı.")
    
    
    
# Tanımlı araç türleri ve frekans bölgeleri
vehicle_freq_ranges = {
    "drone": [(2400, 2500), (5725, 5875)],
    "military_aircraft": [(960, 1215), (2000, 4000)],
    "ground_vehicle": [(70, 512), (890, 960)],
    "naval": [(225, 400), (3000, 6000)],
}

print("🎯 Tarama yapılacak hedef türünü seçiniz:")
for i, vt in enumerate(vehicle_freq_ranges.keys(), 1):
    print(f"{i}. {vt}")

choice = int(input("Seçiminizi yapın (sayı olarak): "))

vehicle_types = list(vehicle_freq_ranges.keys())
if 1 <= choice <= len(vehicle_types):
    target_type = vehicle_types[choice - 1]
    print(f"✅ Seçilen hedef tipi: {target_type}")
else:
    print("⚠️ Geçersiz seçim yapıldı, varsayılan: drone")
    target_type = "drone"

# Frekans bölgelerini al
target_regions = vehicle_freq_ranges[target_type]

peak_frequencies_all = []

for i, (region_start, region_end) in enumerate(target_regions):
    sweep_freqs = list(range(region_start, region_end, 5))
    print(f"\n🟣 Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.figure(figsize=(12, 6))

    for freq in sweep_freqs:
        try:
            sdr.rx_lo = int(freq * 1e6)
            samples = sdr.rx()
        except Exception as e:
            print(f"⚠️ Frekans {freq} MHz alınamadı: {e}")
            continue

        samples = samples - np.mean(samples)

        for b in range(n_bands):
            low = b * band_width
            high = low + band_width
            if low == 0: low = 1
            if high >= sdr.sample_rate // 2: high = (sdr.sample_rate // 2) - 1
            if low >= high: continue

            taps = firwin(
                filter_order,
                [low / (sdr.sample_rate/2), high / (sdr.sample_rate/2)],
                pass_zero=False, window=window_type
            )
            filtered = lfilter(taps, 1.0, samples)
            window = get_window(window_type, len(filtered))
            windowed = filtered * window

            N = len(windowed)
            T = 1.0 / sdr.sample_rate
            xf = fftfreq(N, T)
            yf = fft(windowed)
            xf_mhz = xf[:N//2] / 1e6 + freq + (low / 1e6)
            yf_mag = 2.0 / N * np.abs(yf[:N//2])

            if np.max(yf_mag) < 0.01:
                continue

            plt.plot(xf_mhz, yf_mag, alpha=0.18)

            peaks, properties = find_peaks(
                yf_mag,
                height=np.max(yf_mag) * 0.7,
                distance=10,
                prominence=1
            )
            peak_freqs = xf_mhz[peaks]
            peak_heights = properties['peak_heights']
            strong_peaks = peak_freqs[peak_heights > 0.01]
            peak_frequencies_all.extend(strong_peaks)
            max_idx = np.argmax(yf_mag)
            max_power_freq = xf_mhz[max_idx]
            max_power_peaks.append(max_power_freq)

    plt.title(f"Bölge {i+1}: {region_start}–{region_end} MHz")
    plt.xlabel("Frekans (MHz)")
    plt.ylabel("Genlik")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#tkinter GUI oluşturmak için kullanıldı 

from tkinter import ttk

# Kullanıcının girdiği frekans aralığında grouped_peaks içinden değerleri tabloya yazdıran fonksiyon
def show_peaks_in_range():
    try:
        fmin = float(entry_min.get())
        fmax = float(entry_max.get())
        filtered = [f for f in grouped_peaks if fmin <= f <= fmax]
        # Tabloyu temizle
        for item in peak_table.get_children():
            peak_table.delete(item)
        # Listeyi tabloya ekle
        for f in filtered:
            peak_table.insert("", "end", values=(f"{f:.3f} MHz",))
        if not filtered:
            result.set("No peaks found in this range.")
        else:
            result.set(f"{len(filtered)} peak(s) found.")
    except Exception as e:
        result.set("Please enter valid numbers.")
# Tabloyu ve giriş kutularını temizleyen fonksiyon
def clear_table():
    entry_min.delete(0, tk.END)
    entry_max.delete(0, tk.END)
    for item in peak_table.get_children():
        peak_table.delete(item)
    result.set("")
    # Max Power Peak listesine göre tabloyu güncelleyen fonksiyon
def show_max_power_peaks():
    # Tabloyu temizle
    for item in peak_table.get_children():
        peak_table.delete(item)
    # Max güç listesi
    for f in max_power_peaks:
        peak_table.insert("", "end", values=(f"{f:.3f} MHz",))
    result.set(f"{len(max_power_peaks)} max power peak(s) shown.")
# Sadece belirli frekans aralığının FFT grafiğini çizen fonksiyon
def plot_range_only():
    try:
        fmin = float(entry_min.get())
        fmax = float(entry_max.get())

        if fmin >= fmax:
            result.set("Min < Max olmalı.")
            return
        # Grafik başlat
        plt.figure(figsize=(10, 4))
        plt.title(f"FFT in Range Only: {fmin} – {fmax} MHz")
        plt.xlabel("Frekans (MHz)")
        plt.ylabel("Genlik")
        plt.grid(True)
        # Girilen aralıktaki her 5 MHz için veri topla ve FFT hesapla
        for freq in range(int(fmin), int(fmax), 5):
            try:
                sdr.rx_lo = int(freq * 1e6)
                time.sleep(0.05)
                samples = sdr.rx()
                samples = samples - np.mean(samples)
                windowed = samples * get_window(window_type, len(samples))
                N = len(windowed)
                T = 1.0 / sdr.sample_rate
                xf = fftfreq(N, T)
                yf = fft(windowed)
                xf_mhz = xf[:N // 2] / 1e6 + freq
                yf_mag = 2.0 / N * np.abs(yf[:N // 2])
                plt.plot(xf_mhz, yf_mag, alpha=0.3)
            except Exception as e:
                print(f"⚠️ {freq} MHz çizim hatası: {e}")
                continue

        plt.tight_layout()
        plt.show()
        result.set(f"Plotted {fmin}–{fmax} MHz")
    except Exception as e:
        result.set("Geçerli sayı gir.")

# --- Tkinter Arayüz 
root = tk.Tk()
root.title("Peak Frequency Finder")
root.geometry("330x450")
root.resizable(False, False)

# Giriş alanları (Min/Max Freq)
frm_top = tk.Frame(root, pady=10)
frm_top.pack()

tk.Label(frm_top, text="Min Freq (MHz):", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="e", padx=2)
entry_min = tk.Entry(frm_top, width=8, font=("Segoe UI", 11))
entry_min.grid(row=0, column=1, sticky="w", padx=2)

tk.Label(frm_top, text="Max Freq (MHz):", font=("Segoe UI", 11)).grid(row=1, column=0, sticky="e", padx=2)
entry_max = tk.Entry(frm_top, width=8, font=("Segoe UI", 11))
entry_max.grid(row=1, column=1, sticky="w", padx=2)

frm_btn = tk.Frame(root)
frm_btn.pack(pady=5)
# Butonlar
tk.Button(frm_btn, text="Show Peaks", font=("Segoe UI", 10), width=12, command=show_peaks_in_range).grid(row=0, column=0, padx=4)
tk.Button(frm_btn, text="Clear", font=("Segoe UI", 10), width=8, command=clear_table).grid(row=0, column=1, padx=4)
tk.Button(frm_btn, text="Max Power Peaks", font=("Segoe UI", 10), width=15, command=show_max_power_peaks).grid(row=1, column=0, columnspan=2, pady=4)
tk.Button(frm_btn, text="Plot Only", font=("Segoe UI", 10), width=15, command=plot_range_only).grid(row=2, column=0, columnspan=2, pady=4)
# Bilgilendirme etiketi (Sonuç mesajları burada görünür)
result = tk.StringVar()
tk.Label(root, textvariable=result, font=("Segoe UI", 11, "italic"), fg="#333").pack(pady=4)

# --- Peak Table + Scrollbar ---
frm_table = tk.Frame(root)
frm_table.pack(fill="both", expand=True, padx=10, pady=2)

peak_table = ttk.Treeview(frm_table, columns=("Frequency",), show="headings", height=14)
peak_table.heading("Frequency", text="Frequency (MHz)")
peak_table.column("Frequency", anchor="center", width=130)

scrollbar = ttk.Scrollbar(frm_table, orient="vertical", command=peak_table.yview)
peak_table.configure(yscroll=scrollbar.set)
peak_table.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

frm_table.rowconfigure(0, weight=1)
frm_table.columnconfigure(0, weight=1)
# Arayüzü çalıştır
root.mainloop()
