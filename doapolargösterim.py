# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:02:32 2025

@author: stajyer1
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametreler (örnek)
Nr = 2               # anten sayısı
d = 0.5              # anten aralığı (λ cinsinden)
Nsamples = 10000

# Simülasyon için sinyal (örnek: 20° geliş açısı)
theta_true = 20 * np.pi/180
X = np.zeros((Nr, Nsamples), dtype=complex)
for m in range(Nr):
    delay = m * d * np.sin(theta_true)
    X[m, :] = np.exp(1j * 2*np.pi * delay) + 0.1*np.random.randn(Nsamples)

# DOA taraması
theta_scan = np.linspace(-np.pi, np.pi, 1000)  # -180° ile +180°
results = []
for theta_i in theta_scan:
    w = np.exp(-2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))  # weight vector
    X_weighted = w.conj().T @ X
    results.append(10*np.log10(np.var(X_weighted)))

results = np.array(results)
results -= np.max(results)  # normalize

# Maksimum açıyı yazdır
theta_max_deg = theta_scan[np.argmax(results)] * 180/np.pi
print("Tahmin edilen DOA: {:.2f}°".format(theta_max_deg))

# Lineer plot (derece cinsinden)
plt.figure()
plt.plot(theta_scan*180/np.pi, results)
plt.xlabel("Theta [Derece]")
plt.ylabel("Bartlett Güç (dB, normalize)")
plt.title("DOA Tarama Eğrisi")
plt.grid()

# Polar plot (radyan cinsinden)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results)
ax.set_theta_zero_location('N')     # 0° yukarı
ax.set_theta_direction(-1)          # Saat yönünde artış
ax.set_rlabel_position(55)          # Yazıları yana kaydır
ax.set_title("DOA - Polar Gösterim", va='bottom')

plt.show()
