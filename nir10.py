
import pandas as pd
from scipy.fftpack import fft, ifft
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy as sp
N= 11000
FD = 200

path1 = "s12w1.txt"
# fig, (ax1, ax2,ax3) = plt.subplots(3,1, sharex=True)
data = pd.read_csv(path1, sep=" ", header=None)

#t1 = np.linspace(0, 1, 11000, False)  # 1 second
t = np.linspace(0, 1, 200, False)  # 1 second
t1=[i for i in range(5501)]

m1 = (55*8)/(2*np.pi)
m2 = (55*14)/(2*np.pi)
#Исходный сигнал
plt.figure("Исходный сигнал")
t= np.arange(N)/FD
plt.grid()
plt.plot(t,data[1][0:N])


#спектр мощности
plt.figure("Спектр мощности")
ps = np.abs(np.fft.fft(data[1][0:N]))**2
time_step = 1 / FD
freqs = np.fft.fftfreq(N, time_step)
idx = np.argsort(freqs)
plt.grid()
plt.plot(freqs[idx], ps[idx])

print("Макс частота = ", max(freqs*2))

# спектр
plt.figure("Спектр")
ps = (np.fft.rfft(data[1][0:N]))
x = np.fft.rfftfreq(N, 1./FD)
plt.plot(x, ps)
plt.grid()

# фильтр
plt.figure("Фильтр")
for i in range(5501):
    if x[i]>14:
        ps[i]=0
    if x[i]<8:
        ps[i]=0
plt.plot(x, ps)
plt.grid()


# обратное фурье
path1 = "s12w1f.txt"
data2 = pd.read_csv(path1, sep=" ", header=None)

plt.figure("Отфильтрованный сигнал")
t= np.arange(N)/FD
os = np.fft.irfft(ps)
plt.grid()
plt.plot(t[0:100],os[0:100])
plt.plot(t[0:100] ,data2[1][0:100] , 'red')
plt.show()

fig, (ax1, ax2,ax3) = plt.subplots(3,1, sharex=True)
ax1.plot(t[0:200], data[1][0:200])
ax2.plot(t[0:200],os[0:200].real)
ax3.plot(t[0:200], data2[1][0:200])
plt.show()



plt.show()
print("Макс частота = ", max(freqs*2))




plt.figure(1)
plt.grid()
plt.plot(np.arange(N)/200,data[1][0:N])
plt.figure(2)
plt.grid()
plt.plot(np.fft.rfftfreq(N, 1./200), abs(tmp)/N)
plt.show()

for i in range(11000):
    if ((m1<i<m2) or (N-1-m2<i<N-1-m1)):
        tmp[i]=0

filtered=np.fft.ifft(tmp)

path1 = "s12w1f.txt"
data2 = pd.read_csv(path1, sep=" ", header=None)

ax1.plot(t, data[1][0:200])
ax2.plot(t,filtered[0:200].real)
ax3.plot(t, data2[1][0:200])

plt.show()


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
