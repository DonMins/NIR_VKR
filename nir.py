import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
import os.path as op
import numpy as np
from collections import deque
from mne.minimum_norm import read_inverse_operator, compute_source_psd
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, chirp
N_DATA = 11001
NUMBER_CHANNELS = 16
FREQUENCY = 400


def EegChart(data):
    mas = np.zeros((NUMBER_CHANNELS, N_DATA))

    for i in range(1, NUMBER_CHANNELS + 1):
        allMax = 0
        mmax = abs(max(data[i][0:N_DATA]))
        mmin = abs(min(data[i][0:N_DATA]))
        allMax = max(mmax, mmin)

        for j in range(N_DATA):
            mas[i - 1][j] = (10 * (data[i][j]) / allMax)  # для нормального отображение графиков сделаем нормировку

    data = np.array([mas[0], mas[1], mas[2], mas[3], mas[4], mas[5], mas[6], mas[7],
                     mas[8], mas[9], mas[10], mas[11], mas[12], mas[13], mas[14], mas[15]])
    info = mne.create_info(
        ch_names=['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg'],
        sfreq=FREQUENCY
    )
    raw = mne.io.RawArray(data, info)
    scalings = {'eeg': NUMBER_CHANNELS}

    calings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    raw.plot(n_channels=16, scalings=scalings, title='Auto-scaled Data from arrays',
             show=True, block=True)


def average(data, number):
    i1 = 0
    i2 = 0
    i3 = 0
    k = 0
    sum = 0
    Amplitude = []
    XAmplit = []

    for i in range(N_DATA):
        j = i + 1
        m = j + 1

        i1 = (data[number][i])
        i2 = (data[number][j])
        i3 = (data[number][m])

        if ((abs(i2) >= abs(i1)) & (abs(i2) >= abs(i3))):
            k = k + 1
            if(i2>0):
                Amplitude.append(i2)
                XAmplit.append(j)
            sum = sum + abs(i2)

    return sum /k ,XAmplit, Amplitude


def matWating(input):
    avg = 0
    for j in range(len(input)):
        avg += (input[j])
    return avg / N_DATA


# нахождение дисперсии
def dispers(input):
    disper = 0
    avg = matWating(input)
    for i in range(len(input)):
        disper += (input[i] - avg) ** 2
    return disper / (N_DATA - 1)


correlValues = []


# подсчёт корреляционной функции
def correlfunction(input):
    avg = matWating(input)
    for i in range((len(input)) - 1):
        temp = 0
        for j in range(len(input) - i):
            temp += (input[j] - avg) * (input[j + i] - avg)
        correlValues.append((1 / (N_DATA - i - 1)) * temp)


def AmplitudeAllPac(N):
    arrayNorm = []
    arrayShiz = []
    graficArrayNorm = []
    graficArrayShiz = []

    for i in range(1, N + 1):
        path1 = "NormAlpha\\" + str(i) + ".txt"
        path2 = "ShizAlpha\\" + str(i) + ".txt"
        tmp = 0
        tmp2 = 0
        norm = pd.read_csv(path1, sep=" ", header=None)
        shiz = pd.read_csv(path2, sep=" ", header=None)
        for k in range(1, NUMBER_CHANNELS + 1):
            av1 = average(norm, k)
            arrayNorm.append(av1[0])
            av2 = average(shiz, k)
            arrayShiz.append(av2[0])
        for j in range(NUMBER_CHANNELS):
            tmp = tmp + arrayShiz[j]
            tmp2 = tmp2 + arrayNorm[j]
        graficArrayShiz.append(tmp / len(arrayShiz))
        graficArrayNorm.append(tmp2 / len(arrayNorm))

        # print("Пациенты № " + str(i))
        # print("Норма                       Shiz ")
        # print("F7 = {0}    F7 = {1}:  ".format(arrayNorm[0], arrayShiz[0]))
        # print("F3 = {0}    F3 {1} :  ".format(arrayNorm[1], arrayShiz[1]))
        # print("F4 = {0}    F4 {1} :  ".format(arrayNorm[2], arrayShiz[2]))
        # print("F8 = {0}    F8 {1} :  ".format(arrayNorm[3], arrayShiz[3]))
        # print("T3 = {0}    T3 {1} :  ".format(arrayNorm[4], arrayShiz[4]))
        # print("C3 = {0}    C3 {1} :  ".format(arrayNorm[5], arrayShiz[5]))
        # print("Cz = {0}    Cz {1} :  ".format(arrayNorm[6], arrayShiz[6]))
        # print("C4 = {0}    C4 {1} :  ".format(arrayNorm[7], arrayShiz[7]))
        # print("T4 = {0}    T4 {1} :  ".format(arrayNorm[8], arrayShiz[8]))
        # print("T5 = {0}    T5 {1} :  ".format(arrayNorm[9], arrayShiz[9]))
        # print("P3 = {0}    P3 {1} :  ".format(arrayNorm[10], arrayShiz[10]))
        # print("Pz = {0}    Pz {1} :  ".format(arrayNorm[11], arrayShiz[11]))
        # print("P4 = {0}    P4 {1} :  ".format(arrayNorm[12], arrayShiz[12]))
        # print("T6 = {0}    T6 {1} :  ".format(arrayNorm[13], arrayShiz[13]))
        # print("O1 = {0}    O1 {1} :  ".format(arrayNorm[14], arrayShiz[14]))
        # print("O2 = {0}    O2 {1} :  ".format(arrayNorm[15], arrayShiz[15]))

        arrayNorm = []
        arrayShiz = []

    plt.figure("Amplitude EEG")
    x = [i for i in range(1, 40)]
    leg1, leg2 = plt.plot(x, graficArrayNorm, 'o', x, graficArrayShiz, 'o')
    plt.legend((leg1, leg2), ("Norm", "Shiz"))
    plt.grid(True)
    plt.show()

    plt.figure("Amplitude EEG2")
    leg1, leg2 = plt.plot(x, graficArrayNorm, x, graficArrayShiz)
    plt.legend((leg1, leg2), ("Norm", "Shiz"))
    plt.grid(True)
    plt.show()

    print(graficArrayShiz)
    print(graficArrayNorm)


def cor12(x, y):
    sum = 0


    for i in range(N_DATA):
        pr = x[i] * y[i]
        sum += pr

    return sum / N_DATA

def corrEnvelope(x, y):
    avgX = matWating(x)
    avgY = matWating(y)
    covXY = 0
    for i in range(N_DATA):
        covXY+=(x[i]-avgX)*(y[i]-avgY)
    denominator =0
    dispersionX =0
    dispersionY =0
    for i in range(N_DATA):
        dispersionX += (x[i]-avgX)**2
        dispersionY+= (y[i]-avgY)**2
    denominator = (dispersionX*dispersionY)**0.5

    return covXY/denominator

def plotEnvelope(data):

    amplitude_envelope = np.imag(hilbert(data))
    envelopeArray  = []
    for i in range(len(data)):
       envelopeArray.append(((data[i])**2 + (amplitude_envelope[i])**2)**0.5)

    #cглаживание 5 раз
    for k in range(5):
        for  i in range(len(data)-2):
            envelopeArray[i+1] = (envelopeArray[i]+2*envelopeArray[i+1]+envelopeArray[i+2])/4

    x = [i for i in np.arange(0,len(data)/200,0.005)]
    plt.figure("Огибающая")
    plt.plot(x[200:1200], data[200:1200],  color = '#000000')
    plt.plot(x[200:1200], envelopeArray[200:1200],  color = '#ff0000')
    plt.xlabel("Время (сек) ")
    plt.ylabel("Амплитуда")
    plt.grid()
    plt.show()

def avgTwoCanal(i,j):
    sum = 0

    for k in range(1, 40):
        path1 = "Norm\\" + str(k) + ".txt"
        path2 = "ShizBetta2\\" + str(k) + ".txt"
        norm = pd.read_csv(path2, sep=" ", header=None)
        sum+= round(corrEnvelope(norm[i][0:N_DATA],norm[j][0:N_DATA]), 2)

    return sum/39



if __name__ == "__main__":
    # сдвиг массива
    # g2 = [14.5, 5.4, 4.4, 9.0, 1.5, 5.2]
    # g2 = g2[2:] + g2[:2]
    # print(g2)
    # #[4, 9, 1, 5, 14, 5]

    path1 = "NormAlpha\\" + str(1) + ".txt"
    data = pd.read_csv(path1, sep=" ", header=None)
    # path2 = "Norm\\" +"1st.txt"
    # data2 = pd.read_csv(path2, sep=" ", header=None)

    # print("F3F4  = ", avgTwoCanal(2,3))
    # print("C3C4  = ", avgTwoCanal(6,8))
    # print("P3P4  = ", avgTwoCanal(11,13))
    # print("O1O2  = ", avgTwoCanal(15,16))
    #AmplitudeAllPac(39)

    times = np.arange(1, 100, 1)
    # If times is set to None only 10 regularly spaced topographies will be shown

    # plot magnetometer data as topomaps
    data[1][0:100].plot_topomap(times, ch_type='mag', time_unit='s')

   #  norm = [0.78,0.784,0.72,0.63]
   #  shiz = [0.77,0.76,0.65,0.53]
   #  x =[i for i in np.arange(0,1,0.25)]
   #  my_xticks = ['F3-4', 'C3-4', 'P3-4', 'O1-2']
   #  plt.figure("Межполушарная синхронность фильтрация в тета  диапазоне")
   #  #plt.grid()
   #  plt.xticks(x, my_xticks)
   #  #plt.title(r'$\alpha$')
   # # plt.title(r'$\beta1$')
   #  #plt.title(r'$\beta2$')
   #  plt.title(r'$\theta$')
   # # plt.title(r'$\delta$')
   #  plt.plot(x,norm,marker = 's',color = '#ff0000')
   #  plt.plot(x,shiz,marker = 'o',color = '#000000')
   #  plt.legend(("Норма","Шизофрения"))
   #  plt.grid()
   #  plt.show()


    # w = np.fft.fft(data[1][0:11000])
    # arg = []
    # A=[]
    # freq = []
    # for i in np.arange(0, 5500, 1):
    #     A.append((w[i].real ** 2 + (w[i].imag) ** 2))  # модуль амплитуды
    #
    # print ("A = " , A)
    #
    # for i in np.arange(0, 5500, 1):  # определение фазы
    #     if w[i].imag != 0:
    #         t = (-np.tanh((w[i].real) / (w[i].imag)))
    #         arg.append(t)
    #     else:
    #         arg.append(np.pi / 2)
    #     # радианы в градусы
    #     arg[i]=(arg[i]*180)/np.pi
    #     # получим частоты
    #     freq.append((FREQUENCY*i)/(N_DATA-1))
    # print("Частота " , freq)
    # print("Аргумент " , arg)
    # w = np.fft.fft(data[1][0:11000])
    # x = [i for i in range(11000)]
    #
    # plt.figure("БПФ")
    # plt.plot(x[0:5500], w[0:5500])
    # plt.xlabel("Частота")
    # plt.ylabel("Амплитуда")
    # plt.grid()
    #
    # plt.figure("Спектр мощности")
    # plt.plot(x[0:5500],A)
    # mmax = abs(max(w))
    # mmin = abs(min(w))
    # allMax = max(mmax, mmin)
    # mas = []
    # for j in range(10000):
    #     w[j] = ((2 * w[j]) / allMax)  # для нормального отображение графиков сделаем нормировку
    # plt.grid()
    #
    # data2 = np.array([w[0:5500]])
    # info = mne.create_info(
    #     ch_names=['Амплитуда'],
    #     ch_types=['eeg'],
    #     sfreq=FREQUENCY
    # )
    # raw = mne.io.RawArray(data2, info)
    # scalings = {'eeg': 1}
    # calings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    # raw.plot(n_channels=1, scalings=scalings, title='Auto-scaled Data from arrays',
    #          show=True, block=True)
    #
    # print(w)



    # arrayNorm = []
    # arrayShiz = []
    # graficArrayNorm = []
    # graficArrayShiz = []
    # namePara = {'F7F3    F3F4   F4F8  F8T3  T3C3  C3Cz  C4T4   T4T5  T5P3  P3Pz  PzP4  P4T6  T6O  O1O2 '}
    # namePara2 = { 'T5T6   P3P4    T3T4    C3C4    F7F8 '}
    # synchroMatrixNorm = np.zeros((40, 22))
    # synchroMatrixShiz = np.zeros((40, 22))
    #
    # for i in range(1, 40):
    #     path1 = "Norm\\" + str(i) + ".txt"
    #     path2 = "Shiz\\" + str(i) + ".txt"
    #     norm = pd.read_csv(path2, sep=" ", header=None)
    #     shiz = pd.read_csv(path2, sep=" ", header=None)
    #     for j in range(1,16):
    #         synchroMatrixNorm[i - 1][j - 1] = round(cor12(norm[j][0:N_DATA],norm[j+1][0:N_DATA]),2)
    #        # synchroMatrixShiz[i - 1][j - 1] = round(cor12(shiz[j][0:N_DATA],shiz[j+1][0:N_DATA]),2)
    #
    #     synchroMatrixNorm[i - 1][16] = round(cor12(norm[10][0:N_DATA], norm[14][0:N_DATA]), 2)
    #     synchroMatrixNorm[i - 1][17] = round(cor12(norm[11][0:N_DATA], norm[13][0:N_DATA]), 2)
    #     synchroMatrixNorm[i - 1][18] = round(cor12(norm[5][0:N_DATA], norm[9][0:N_DATA]), 2)
    #     synchroMatrixNorm[i - 1][19] = round(cor12(norm[6][0:N_DATA], norm[8][0:N_DATA]), 2)
    #     synchroMatrixNorm[i - 1][20] = round(cor12(norm[1][0:N_DATA], norm[4][0:N_DATA]), 2)
    #
    # print(namePara)
    # for i in range(39):
    #      print(*[synchroMatrixNorm[i, j] for j in range(15)])
    # print(namePara2)
    # for i in range(39):
    #     print(*[synchroMatrixNorm[i, j] for j in range(15,21,1)])




    # arrayNorm = []
    # arrayShiz = []
    # graficArrayNorm = []
    # graficArrayShiz = []
    # namePara = {'F7F3 F3F4 F4F8  F8T3 T3C3 C3Cz C4T4 T4T5  T5P3 P3Pz PzP4 P4T6 T6O O1O2 T5T6 P3P4 T3T4 C3C4 F7F8 '}
    # namePara2 = { 'T5T6 P3P4 T3T4 C3C4 F7F8 '}
    # synchroMatrixNorm = np.zeros((40, 22))
    # synchroMatrixShiz = np.zeros((40, 22))
    #
    # for i in range(1, 2):
    #     path1 = "Norm\\" + str(i) + ".txt"
    #     path2 = "Shiz\\" + str(i) + ".txt"
    #     norm = pd.read_csv(path1, sep=" ", header=None)
    #     shiz = pd.read_csv(path2, sep=" ", header=None)
    #     for j in range(1,16):
    #         synchroMatrixNorm[i - 1][j - 1] = round(corrEnvelope(norm[j][0:N_DATA],norm[j+1][0:N_DATA]), 2)
    #         synchroMatrixNorm[i - 1][15] = round(corrEnvelope(norm[10][0:N_DATA], norm[14][0:N_DATA]), 2)
    #         synchroMatrixNorm[i - 1][16] = round(corrEnvelope(norm[11][0:N_DATA], norm[13][0:N_DATA]), 2)
    #         synchroMatrixNorm[i - 1][17] = round(corrEnvelope(norm[5][0:N_DATA], norm[9][0:N_DATA]), 2)
    #         synchroMatrixNorm[i - 1][18] = round(corrEnvelope(norm[6][0:N_DATA], norm[8][0:N_DATA]), 2)
    #         synchroMatrixNorm[i - 1][19] = round(corrEnvelope(norm[1][0:N_DATA], norm[4][0:N_DATA]), 2)
    #        # synchroMatrixShiz[i - 1][j - 1] = round(cor12(shiz[j][0:N_DATA],shiz[j+1][0:N_DATA]),2)
    #
    #
    #
    # print(namePara)
    # for i in range(39):
    #      print(*[synchroMatrixNorm[i, j] for j in range(20)])
