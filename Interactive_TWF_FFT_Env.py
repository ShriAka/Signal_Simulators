# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 10:36:20 2023

@author: shriaka
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 18,
          'figure.figsize': (25, 15),
         'axes.labelsize': 18,
         'axes.titlesize':24,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
from scipy import signal 
from scipy.fftpack import fft,ifft
from matplotlib.widgets import Slider, Button

#%%

'''
Multisine signal generation
'''
# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 5))
plt.subplots_adjust(bottom=0.35)
 
# Create plot for time waveform
fs = 10000
time = np.arange(0.0, 1.0, 1/fs)
x = 5 * np.cos(2 * np.pi * 25 * time)
twf, = ax1.plot(time, x )
ax1.set_ylim(-30, 30)
ax1.set_xlim(0, 1) 
ax1.set_title('Time waveform')

# FFT and frequency    
fft_y=abs(np.fft.rfft(x))*2/len(x)
freq=np.fft.rfftfreq(len(x), 1/(fs))

#create plot spectrum
spectrum, = ax2.plot(freq, fft_y)
ax2.set_ylim(0, 10)
ax2.set_xlim(0, 2000)
ax2.set_title('Spectrum')

# Envelope spectrum function
def env_sp(twf, fs, low_f, high_f):
    benv, aenv = signal.butter(3, [low_f, high_f], 'bandpass', analog=False, fs = fs)
    twf = signal.lfilter(benv, aenv, twf)
    N = len(twf)
    X = fft(twf,N)
    h = np.zeros(N)
    h[0] = 1
    h[N//2:N+1] = 2*np.ones(N//2)
    h[N//2] = 1    
    Z = X*h
    xa = ifft(Z,N)        
    x_envelope = np.abs(xa)
    xenv = x_envelope - np.mean(x_envelope)
        
    xenv = xenv*np.hanning(len(xenv))    
    fftxenv=2*abs(np.fft.rfft(xenv))*2/N   ### Amp. correction factor of 2 for hanning window
    freq=np.fft.rfftfreq(N, 1/fs)    
    
    return freq, fftxenv


# Envelope plot


fmin = 500
fmax = 1000
freq1, fftxenv1 = env_sp(x, fs, fmin, fmax)
#create plot envelope spectrum
env_spectrum, = ax3.plot(freq1, fftxenv1)
ax3.set_ylim(0, 10)
ax3.set_xlim(0, 500)
ax3.set_title('Envelope Spectrum')

# Create axes for sliders
axfreq_1 = plt.axes([0.1, 0.25, 0.2, 0.03]) #location input (start x, start y, width, hight)
axfreq_2 = plt.axes([0.1, 0.23, 0.2, 0.03])
axfreq_3 = plt.axes([0.1, 0.21, 0.2, 0.03])
axfreq_4 = plt.axes([0.1, 0.19, 0.2, 0.03])
axfreq_5 = plt.axes([0.1, 0.17, 0.2, 0.03])

axamplitude_1 = plt.axes([0.4, 0.25, 0.2, 0.03])
axamplitude_2 = plt.axes([0.4, 0.23, 0.2, 0.03])
axamplitude_3 = plt.axes([0.4, 0.21, 0.2, 0.03])
axamplitude_4 = plt.axes([0.4, 0.19, 0.2, 0.03])
axamplitude_5 = plt.axes([0.4, 0.17, 0.2, 0.03])

axphase_1 = plt.axes([0.7, 0.25, 0.2, 0.03])
axphase_2 = plt.axes([0.7, 0.23, 0.2, 0.03])
axphase_3 = plt.axes([0.7, 0.21, 0.2, 0.03])
axphase_4 = plt.axes([0.7, 0.19, 0.2, 0.03])
axphase_5 = plt.axes([0.7, 0.17, 0.2, 0.03])

'''
Create sliders
'''
# Create a slider for frequency
frequency_1 = Slider(axfreq_1, 'Frequency 1', 0.0, 2000, 25, valstep=10)
frequency_2 = Slider(axfreq_2, 'Frequency 2', 0.0, 2000, 0, valstep=10)
frequency_3 = Slider(axfreq_3, 'Frequency 3', 0.0, 2000, 0, valstep=10)
frequency_4 = Slider(axfreq_4, 'Frequency 4', 0.0, 2000, 0, valstep=10)
frequency_5 = Slider(axfreq_5, 'Frequency 5', 0.0, 2000, 0, valstep=10)

 
# Create a slider from 0.0 to 10.0 in axes axfreq
# with 5 as initial value and valsteps of 1.0
amplitude_1 = Slider(axamplitude_1, 'Amplitude 1', 0.0, 10.0, 5, valstep=1.0)
amplitude_2 = Slider(axamplitude_2, 'Amplitude 2', 0.0, 10.0, 0, valstep=1.0)
amplitude_3 = Slider(axamplitude_3, 'Amplitude 3', 0.0, 10.0, 0, valstep=1.0)
amplitude_4 = Slider(axamplitude_4, 'Amplitude 4', 0.0, 10.0, 0, valstep=1.0)
amplitude_5 = Slider(axamplitude_5, 'Amplitude 5', 0.0, 10.0, 0, valstep=1.0)

# Create a slider for phase
phase_1 = Slider(axphase_1, 'Phase 1',-180, 180, 0, valstep=10)
phase_2 = Slider(axphase_2, 'Phase 2',-180, 180, 0, valstep=10)
phase_3 = Slider(axphase_3, 'Phase 3',-180, 180, 0, valstep=10)
phase_4 = Slider(axphase_4, 'Phase 4',-180, 180, 0, valstep=10)
phase_5 = Slider(axphase_5, 'Phase 5',-180, 180, 0, valstep=10)
 
# Define function that will change output with slider values
 
def update_w_slider(val):
    f1 = frequency_1.val
    a1 = amplitude_1.val
    p1 = phase_1.val
    
    f2 = frequency_2.val
    a2 = amplitude_2.val
    p2 = phase_2.val

    f3 = frequency_3.val
    a3 = amplitude_3.val
    p3 = phase_3.val
    
    f4 = frequency_4.val
    a4 = amplitude_4.val
    p4 = phase_4.val

    f5 = frequency_5.val
    a5 = amplitude_5.val
    p5 = phase_5.val
    
    
    twf_y = a1*np.cos(2*np.pi*f1*time+p1*np.pi/180) + a2*np.cos(2*np.pi*f2*time+p2*np.pi/180) + a3*np.cos(2*np.pi*f3*time+p3*np.pi/180)\
        +a4*np.cos(2*np.pi*f4*time+p4*np.pi/180)+a5*np.cos(2*np.pi*f5*time+p5*np.pi/180)#variable TWF
        
    twf.set_ydata(twf_y)
    
    fft_y = abs(np.fft.rfft(twf_y))*2/len(twf_y) #variable spectrum
    spectrum.set_ydata(fft_y)
            
    freq1, fftxenv1 = env_sp(twf_y, fs, 500, 1000)
    env_spectrum.set_ydata(fftxenv1)
        
    ax1.set_ylim(-30, 30)
    ax2.set_ylim(0, 10)
    ax2.set_xlim(0, 2000)
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0, 10)
    
# Call function
frequency_1.on_changed(update_w_slider)
frequency_2.on_changed(update_w_slider)
frequency_3.on_changed(update_w_slider)
frequency_4.on_changed(update_w_slider)
frequency_5.on_changed(update_w_slider)

amplitude_1.on_changed(update_w_slider)
amplitude_2.on_changed(update_w_slider)
amplitude_3.on_changed(update_w_slider)
amplitude_4.on_changed(update_w_slider)
amplitude_5.on_changed(update_w_slider)

phase_1.on_changed(update_w_slider)
phase_2.on_changed(update_w_slider)
phase_3.on_changed(update_w_slider)
phase_4.on_changed(update_w_slider)
phase_5.on_changed(update_w_slider)

plt.show()

#%%
















