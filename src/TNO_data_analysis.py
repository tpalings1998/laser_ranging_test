#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:25:49 2023

@author: tijs
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy import fftpack
import matplotlib.pyplot as plt
from deglitcher import deglitcher
import scipy.integrate as integrate
from tqdm import tqdm
from constants import constants
import os

cwd = os.getcwd()
data_dir = os.path.join(os.path.split(cwd)[0],'data')

dt = 2E-8
c = constants().c
N = constants().N
#Fill in the path to the TUdelft_test.txt
path = os.path.join(data_dir,'TUdelft_test.txt')
file = open(path,'r')
data = []
for line in file.readlines():
    #data.append([line])
    s = []
    try:
        a = line.split(',')
        for i in a:
            s.append(np.float64(i))
        data.append(s)
    except:
        pass

file.close()
data = np.array(data)
#some filters to try to get rid of the noise, not really working as well as anticipated
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#normalizing the data to fit bits on the raw data, easier for automated detection
def bitfitter_norm(y,z=3):
    y_ = y#np.abs(y)
    avg = np.average(y_)
    std = np.std(y_)
    bits = np.zeros(len(y))
    t = avg+z*std
    bits[np.where(y>=t)] = 1
    bits[np.where(y<=-t)] = -1
    return bits


#splices the data into its traces to get rid ov overaveraging behaviour
def data_cutter(data,z=1):
    N = 1400
    K = len(data)
    m = K/N
    s = np.array([])
    for i in range(int(m)):
        splice = data[i*N:(i+1)*N]
        s = np.append(s,bitfitter_norm(splice,z=z))
    return s

def tocs(data):
    N=1400
    K = len(data)
    m = K/N
    avgs = np.array([])
    stds = np.array([])
    for i in range(int(m)):
        splice = data[i*N:(i+1)*N]
        avgs = np.append(avgs,np.average(splice))
        stds = np.append(stds,np.std(splice)**2)
    df = pd.DataFrame({"Averages":avgs,
                       "Stdevs":stds})
    df.to_excel('C:/Users/tijsp/Documents/TU/Thesis/TNO test/tracestats.xlsx')
#splits the large data into traces to generate delta points
def distance_est(a,b,typ='front'):
    N = 1400
    K = len(a)
    m = int(K/N)
    distances = []
    for i in range(m):
        splice1 = np.abs(a[i*N:(i+1)*N])
        splice2 = np.abs(b[i*N:(i+1)*N])
        if typ =='front':
            ind1 = np.where(splice1==1)[0][0]
            ind2 = np.where(splice2==1)[0][0]
        elif typ =='back':
            ind1 = np.where(splice1==1)[0][-1]
            ind2 = np.where(splice2==1)[0][-1]
        distances.append(np.abs(ind1-ind2))
    return np.array(distances)


#function to lookup traces individually for manual inspection
def trace_lookup(N,a,b):
    plt.figure(figsize=(10,6))
    t = np.arange(len(a))*2E-8*1E6
    t = t[N*1400:(N+1)*1400]
    plt.plot(a[N*1400:(N+1)*1400],color="blue",label="Tx")
    plt.plot(b[N*1400:(N+1)*1400],color='red',label="Rx")
    plt.xlabel(r"Time $(\mu s)$")
    plt.ylabel(r"Voltage (V)")
    plt.grid()
    plt.legend()
    plt.title("Trace: "+str(N))
    plt.tight_layout()
    return

def atmos_correction(P,T0,L,g_t,Rh,TL=0,TL2=0):
    e = Rh*6.11*10**((7.5*(T0-273.12))/(237.3 +T0-273.15))
    print(e)
    #n_emp = 1+ 83.11E-6 * P - 11.4E-6*e*(1/(T0+0.5*g_t*L))
    #n_emp = 1+(1/T0)*(83.11E-6 * P - 11.4E-6*e)*(2/(1+(TL/T0)))
    n_emp = 1+(1/TL2)*(83.11E-6 * P - 11.4E-6*e)*(1/(1+(1/3)*(T0-TL2)/TL2))
    return n_emp

def indiv_pulse(tx,rx,spacing=10):

    
    r1 = np.where(np.diff(tx)>=1)[0]+1
    r2 = np.where(np.diff(rx)>=1)[0]+1
    
    expected = np.zeros(len(rx))
    
    for i in range(40):
        expected[r2[0]+(i*spacing)] = 1
    
    da = np.where(((rx == expected)&(rx==1))==True)[0]
    
    
    
    dpulse = np.abs(r1-da)
    #plt.plot(r1)
    plt.plot(dpulse)
    plt.plot(expected)
    plt.plot(rx)
    
    return

def best_z_rx(data):
    N = 1400
    K = len(data)
    m = K/N
    s = np.array([])
    zs = np.linspace(0,3,1000)
    dz = zs[1] - zs[0]
    for i in tqdm(range(int(m))):
        splice = data[i*N:(i+1)*N]
        std = np.std(splice)
        _splice = np.abs(splice)
        
        t = np.arange(0,len(splice)*2E-8,2E-8)
        #s = np.append(s,bitfitter_norm(splice,z=z))
        sums = []
        for z in zs:
            _splice = _splice-(z*std)
            _splice[np.where(_splice<0)] = 0
            integral = integrate.simpson(_splice,t)
            #print(integral)
            sums.append(integral)
        if len(s)==0:
            s = np.array(sums)
        else:
            s = np.vstack((s,np.array(sums)))
    z_best = np.array([])
    for ss in s:
        dsdz = np.gradient(ss,dz)
        z_best = np.append(z_best,zs[np.where(np.abs(dsdz)==max(np.abs(dsdz)))])
    _s = np.array([])
    for i in tqdm(range(int(m))):
        splice = data[i*N:(i+1)*N]
        _s = np.append(_s,bitfitter_norm(splice,z=z_best[i]))
        
        
    
    return _s,s,zs,z_best
degg = deglitcher()

#raw data from txt
data_1540 = data[:100,1:].reshape(1,-1)[0] +1.25  #Rx: reshaping the data to center 0
data_1560 = data[100:200,1:].reshape(1,-1)[0] -2  #Tx reshaping the data to center 0

#tocs(data_1540-1.25)
#some properties to try filtering the data
lowcut, highcut = 0.45E7, 0.55E7
fs = 1/dt
nyq = 0.5*fs
time = np.linspace(0,dt*len(data_1540),len(data_1540))

sig_noise_fft = fftpack.fft(data_1540)
sig_noise_amp = 2/time.size * np.abs(sig_noise_fft)
sig_noise_freq = np.abs(fftpack.fftfreq(time.size,dt))

plt.figure()
plt.title("FFT of Rx data")
plt.plot(sig_noise_freq,sig_noise_amp,label="FFT")
plt.ylabel('Amplitude (-)')
plt.xlabel("Frequency (Hz)")
plt.grid()
plt.show()

plt.figure()
plt.title("Filtered rx data")
data_filtered = butter_bandpass_filter(data_1540, lowcut,highcut,fs)
plt.plot(data_filtered)
plt.grid()
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(2,1)

rx_bits             = data_cutter(data_1540,z=1.7)
#rx_bits_filtered    = bitfitter_norm(data_filtered,z=0.3)
tx_bits             = bitfitter_norm(data_1560,z=1)

tx_bits2 = tx_bits.copy()
tx_bits2[np.where(tx_bits2<0)] = 0
rx_bits2 = rx_bits.copy()
rx_bits2[np.where(rx_bits2<0)] = 0
r1 = np.where(np.diff(tx_bits2)>=1)[0]
r2 = np.where(np.diff(rx_bits2)>=1)[0]






ax[0].plot(rx_bits,label='rx bits')
ax[0].plot(tx_bits,label='tx bits')
ax[0].set_title("Fitted bits to data")
ax[0].grid()
ax[1].plot(data_1540,label='rx data')
ax[1].plot(data_1560,label='tx data')
ax[1].set_title("Raw data")
ax[1].grid()
plt.tight_layout()
plt.show()

fig2,ax2 = plt.subplots(2,1)
dpts = distance_est(rx_bits,tx_bits,typ='back')
distances = 0.5*dpts*dt*3E8
ax2[0].plot(dpts)
ax2[0].set_xlabel("Trace number")
ax2[0].set_ylabel("Delta points")
ax2[0].grid()
ax2[0].set_title("Delta points")
ax2[1].plot(distances)
ax2[1].set_xlabel("Trace number")
ax2[1].set_ylabel("Distance (m)")
ax2[1].set_title("Measured distance")
ax2[1].grid()
plt.tight_layout()
plt.show()

#correcting for atmosphere
P       = 1016.8
TL      = 285.7
T0      = 284.95
L       = 2442
g_t     = np.linspace(-0.008,0.008,10000)
Rh      = 0.93

dT = dpts*dt
n_emp = atmos_correction(P, T0, L, g_t, Rh,TL=TL,TL2=(T0+g_t*L))
L = c*dT/1.00028
#print(L/2)


plt.figure(figsize=(10,6))
n=2
#plt.plot(np.arange(len(data_1560[:n*1400]))*2E-2,data_1560[:n*1400]+2,color='blue',label="Tx")
plt.plot(np.arange(len(data_1540[:n*1400]))*2E-2,rx_bits[:n*1400],color='red',label="Rx")
plt.legend()
plt.grid()
plt.xlabel(r'time ($\mu$s)')
plt.ylabel('Bit value (-)')
plt.title("Magnified bit values of trace 2 ")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
n=15
plt.plot(np.arange(len(data_1560[:n*1400]))*2E-2,data_1560[:n*1400]+2,color='blue',label="Tx")
plt.plot(np.arange(len(data_1540[:n*1400]))*2E-2,data_1540[:n*1400]-1.25,color='red',label="Rx")
plt.legend()
plt.grid()
plt.xlabel(r'time $\mu$s')
plt.ylabel('Voltage (V)')
plt.title("Raw measurement data from the oscilloscope consisting of three randomly selected consecutive traces in the sequence")
plt.tight_layout()
plt.show()
zs = np.arange(0.7,1.75,0.05)
tx_bits             = bitfitter_norm(data_1560,z=1)
for z in zs:
    rx_bits             = data_cutter(data_1540,z=z)
    dpts_back = distance_est(rx_bits,tx_bits,typ='back')
    dpts_front = distance_est(rx_bits,tx_bits,typ='front')
    error = dpts_front -dpts_back




dpts_back = distance_est(rx_bits,tx_bits,typ='back')
dpts_front = distance_est(rx_bits,tx_bits,typ='front')


plt.figure(figsize=(10,6))
plt.plot(dpts_front*2E-8*1E6,color='black',label=r"$\Delta t_{rise}$",linestyle='dashed')
plt.plot(dpts_back*2E-8*1E6,color='black',label=r"$\Delta t_{fall}$")
plt.xlabel("Trace number")
plt.ylabel(r"$\Delta t$ $(\mu s)$")
plt.legend()
plt.grid()
plt.title("Time difference calculation automatically determined by algorithm z = 1.5")
plt.tight_layout()

plt.figure(figsize=(10,6))
plt.plot(np.arange(len(data_1560[:3*1400]))*2E-2,tx_bits[:3*1400],color='Blue',label="Tx")
#plt.plot(np.arange(len(data_1540[:3*1400]))*2E-2,data_1540[:3*1400]-1.25,color='red',label="Rx")
plt.legend()
plt.grid()
plt.xlabel(r'time $\mu$s')
plt.ylabel('Bit value (-)')
plt.title("Transmitter readings by the oscilloscope after applying a Gaussion distribution filter")
plt.tight_layout()
plt.show()


fig,ax = plt.subplots(2,1)
fig.set_figwidth(10)
fig.set_figheight(6)


ax[0].plot(dpts_front,color='black',label=r"$\Delta t_{rise}$",linestyle='dashed')
ax[0].plot(dpts_back,color='black',label=r"$\Delta t_{fall}$",linestyle='solid')

ax[1].plot(dpts_front*2E-8*1E6 -dpts_back*2E-8*1E6,color='black', label=r"$\Delta t_{rise} - \Delta t_{fall}$")
ax[0].set_xlabel("Trace number")
ax[0].set_ylabel(r"$\Delta t$ $(\mu s)$")
ax[1].set_xlabel("Trace number")
ax[1].set_ylabel(r"$\Delta t$ $(\mu s)$")
ax[0].grid()
ax[1].grid()
fig.suptitle("Time difference calculation automatically determined by algorithm")

ax[0].legend()
ax[1].legend()
plt.tight_layout()

"""
plt.figure(figsize=(10,6))
df = pd.read_excel('C:/Users/tijsp/Documents/TU/Thesis/TNO test/Test_results.xlsx',sheet_name='update')
plt.plot(df['Trace'],df['OLD']*2E-8*1E6,color='black',label="Before correction",linestyle="dashed")
plt.plot(df['Trace'],df['dpts(corrected)']*2E-8*1E6,color='black',label="After correction")
plt.ylabel(r'$\Delta t_{total}$ $(\mu s)$')
plt.xlabel("Trace number")
plt.legend()
plt.grid()
plt.title("Correction of the ToF measurements")
plt.tight_layout()
"""

plt.figure(figsize=(10,6))
df = pd.read_excel('C:/Users/tijsp/Documents/TU/Thesis/TNO test/Test_results.xlsx',sheet_name='update')
dRG58 = 10/(0.66*c)
dFIBERlow = 24/(c/N)
dFIBERhigh = 29/(c/N)

y1 = df['dpts(corrected)']*2E-8 - dRG58 - dFIBERlow
y2 = df['dpts(corrected)']*2E-8 - dRG58 - dFIBERhigh
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(df['Trace'],y1*1E6 ,color='black',label="Bounds",linestyle="solid")
ax1.plot(df['Trace'],y2*1E6 ,color='black',linestyle="solid")
ax2.plot(df["Trace"],y1*c/2,color='black',linestyle="solid")
ax2.plot(df["Trace"],y2*c/2,color='black',linestyle="solid")
ax1.fill_between(df["Trace"], y1*1E6, y2*1E6, interpolate=True, color='lightgray', alpha=0.5)
ax1.set_ylabel(r'$\Delta t$ $(\mu s)$')
ax1.set_xlabel("Trace number")
ax2.set_ylabel("Range (m)")
ax1.legend()
ax1.grid()
plt.title("ToF and range resulting from the measurements")
plt.tight_layout()

