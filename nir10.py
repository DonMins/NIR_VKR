import biosppy.signals.tools as f
import pandas as pd
from scipy.fftpack import fft, ifft
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy as sp
path1 = "s12w1.txt"
fig, (ax1, ax2,ax3) = plt.subplots(3,1, sharex=True)
data = pd.read_csv(path1, sep=" ", header=None)

t = np.linspace(0, 1, 100, False)  # 1 second

#

sos = signal.butter(8, 14, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, data[1][0:500])
# m1 = (55*8)/(2*np.pi)
# m2 = (55*14)/(2*np.pi)
# N= 11000
# for i in range(11000):
#     if not (m1<tmp[i]<m2) or (N-1-m2<tmp[i]<N-1-m1):
#         tmp[i]=0




path1 = "s12w1f.txt"
data2 = pd.read_csv(path1, sep=" ", header=None)

ax2.plot(t,filtered[0:100])
ax1.plot(t, data[1][0:100])
ax3.plot(t, data2[1][0:100])

plt.show()
#

from scipy.fftpack import rfft, irfft, fftfreq

time = np.linspace(0, 1, 500, False)  # 1 second
signal = data[1][0:11000]

W = fftfreq(signal.size, d=0.005)

f_signal =rfft(signal)

f_signal[(W>14)] = 0
f_signal[(W>14)] = 0


cut_signal = irfft(f_signal)
print(len(cut_signal))

ax2.plot(time,cut_signal[0:500])
ax1.plot(time, signal[0:500])
ax3.plot(time, data2[1][0:500])

plt.show()




# frate =200
#
# w = np.fft.fft(data[1][0:11000])
# freqs = np.fft.fftfreq(len(w))
# print(freqs.min(), freqs.max())
# # (-0.5, 0.499975)
#
# # Find the peak in the coefficients
# idx = np.argmax(np.abs(w))
# freq = freqs[idx]
# freq_in_hertz = abs(freq * frate)
# print(freq_in_hertz)

