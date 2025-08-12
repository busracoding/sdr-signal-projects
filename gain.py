# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:59:53 2025

@author: stajyer1
"""

import numpy as np
import adi

sample_rate = 1e6 # Hz
center_freq = 100e6 # Hz
num_samps = 10000 # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.2.1')
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 70.0 # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

samples = sdr.rx() # receive samples off Pluto
print(samples[0:10])



"""The code below assumes you have the Pluto’s Python API installed.
 This code initializes the Pluto, sets the sample rate to 1 MHz, sets the center frequency to 100 MHz, 
 and sets the gain to 70 dB with automatic gain control turned off. Note it usually doesn’t matter the order
 in which you set the center frequency, gain, and sample rate. In the code snippet below, we tell the Pluto 
 that we want it to give us 10,000 samples per call to rx(). We print out the first 10 samples."""
 
