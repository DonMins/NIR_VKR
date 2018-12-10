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
from scipy import signal
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import statistics

N_DATA = 11001
NUMBER_CHANNELS = 16
FREQUENCY = 200


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

def average(data,shiz, number):
    kNorm = 0
    sumNorm = 0
    kShiz=0
    sumShiz =0

    for i in range(N_DATA):
        j = i + 1
        m = j + 1

        i1 = (data[number][i])
        i2 = (data[number][j])
        i3 = (data[number][m])

        j1 = (shiz[number][i])
        j2 = (shiz[number][j])
        j3 = (shiz[number][m])

        if ((abs(i2) >= abs(i1)) & (abs(i2) >= abs(i3))):
            kNorm = kNorm + 1
            sumNorm = sumNorm + abs(i2)

        if ((abs(j2) >= abs(j1)) & (abs(j2) >= abs(j3))):
            kShiz = kShiz + 1
            sumShiz = sumShiz + abs(j2)

    return sumNorm/kNorm, sumShiz/kShiz


amplitudeTest = np.zeros((15, 16))
amplitude = np.zeros((63, 16))

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
            av = average(norm,shiz, k)
            arrayNorm.append(av[0])
            arrayShiz.append(av[1])

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

def corrEnvelope(x, y):
    amplitude_envelopeX = (hilbert(x))
    amplitude_envelopeY = (hilbert(y))
    envelopeArrayX = np.abs(amplitude_envelopeX)
    envelopeArrayY = np.abs(amplitude_envelopeY)
    corrPirson =np.corrcoef(envelopeArrayX,envelopeArrayY)[0,1]
    return corrPirson

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


corrAlpha = np.zeros((63, 120))
corrAlphaTest = np.zeros((15, 120))


def avgTwoCanal(i,j,index):
    sum = 0
    t = 0

    for k in range(1, 40):
        path2 = "NormBetta2\\" + str(k) + ".txt"
        path1 = "ShizBetta2\\" + str(k) + ".txt"
        norm = pd.read_csv(path2, sep=" ", header=None)
        shiz = pd.read_csv(path1, sep=" ", header=None)
        corrAlpha[t][index]=corrEnvelope(norm[i][0:N_DATA],norm[j][0:N_DATA])
        if (k<=24):
            corrAlpha[t+1][index]=corrEnvelope(shiz[i][0:N_DATA],shiz[j][0:N_DATA])
            t = t + 2
        else:
            t=t+1

def amplitide(fileName):
    sum = 0
    t = 0

    for k in range(1, 40):
        path2 = "NormAlpha\\" + str(k) + ".txt"
        path1 = "ShizAlpha\\" + str(k) + ".txt"
        norm = pd.read_csv(path2, sep=" ", header=None)
        shiz = pd.read_csv(path1, sep=" ", header=None)
        for m in range(1,NUMBER_CHANNELS+1):
            tmp =average(norm,shiz,m)
            amplitude[t][m-1] = tmp[0]
            if (k<=24):
                amplitude[t+1][m-1] = tmp[1]
        if (k<=24):
            t=t+2
        else:
            t = t+1
    np.savetxt(fileName, amplitude, fmt="%f")


def amplitideTest(fileNameTest):
    sum = 0
    t = 0

    for k in range(25, 40):
        path1 = "ShizAlpha\\" + str(k) + ".txt"
        shiz = pd.read_csv(path1, sep=" ", header=None)
        for m in range(1,NUMBER_CHANNELS+1):
            tmp =average(shiz,shiz,m)
            amplitudeTest[t][m-1] = tmp[0]
        t = t+1
    np.savetxt(fileNameTest, amplitudeTest, fmt="%f")

def avgTwoCanal3(i,j,index):
    sum = 0
    t =0
    for k in range(25, 40):
      #  path2 = "NormBetta2\\" + str(k) + ".txt"
        path1 = "ShizBetta2\\" + str(k) + ".txt"
       # norm = pd.read_csv(path2, sep=" ", header=None)
        shiz = pd.read_csv(path1, sep=" ", header=None)
        #corrAlphaTest[t][index]=corrEnvelope(norm[i][0:N_DATA],norm[j][0:N_DATA])
        corrAlphaTest[t][index]=corrEnvelope(shiz[i][0:N_DATA],shiz[j][0:N_DATA])
        #t=t+2
        t=t+1

def genY():
    y = []
    for i in range(31):
        y.append(0)
        if (i<24):
            y.append(1)
        else:
            y.append(0)
    y.append(0)
    return y

def genXandTest(fileName,fileNameTest):
    index = 0
    for i in range(1, 17):
        for j in range(i + 1, 17):
            avgTwoCanal(i, j, index)
            index += 1
    np.savetxt(fileName, corrAlpha, fmt="%f")

    index = 0
    for i in range(1, 17):
        for j in range(i + 1, 17):
            avgTwoCanal3(i, j, index)
            index += 1

    np.savetxt(fileNameTest, corrAlphaTest, fmt="%f")



if __name__ == "__main__":
    # сдвиг массива
    # g2 = [14.5, 5.4, 4.4, 9.0, 1.5, 5.2]
    # g2 = g2[2:] + g2[:2]
    # print(g2)
    # #[4, 9, 1, 5, 14, 5]

    # fileName = open('corrA.txt', 'w')
    # fileNameTest = open('corrAt.txt', 'w')
    # genXandTest(fileName,fileNameTest)

    # fileName = open('amplitude.txt', 'w')
    # fileNameTest = open('amplitudeTest.txt', 'w')
    # amplitide(fileName)
    # amplitideTest(fileNameTest)
# # # загрузить из файла
    x =  np.loadtxt('corrA.txt', delimiter=" ")
    x_test =  np.loadtxt('corrAt.txt', delimiter=" ")

    clf = LinearDiscriminantAnalysis()
    clf.fit(x, genY())
    test_predictions = clf.predict(x_test)
    print(test_predictions)

    x = np.loadtxt('amplitude.txt', delimiter=" ")
    x_test = np.loadtxt('amplitudeTest.txt', delimiter=" ")

    clf = LinearDiscriminantAnalysis()
    clf.fit(x, genY())
    test_predictions = clf.predict(x_test)
    print(test_predictions)





    # norm = pd.read_csv("27.txt", sep=" ", header=None)
    # norm2 = pd.read_csv("27f.txt", sep=" ", header=None)
    # x = [i for i in np.arange(1200)]
    # plt.figure("Огибающая")
    # plt.plot(x[200:800], norm[1][200:800],  color = '#000000')
    # plt.plot(x[200:800], norm2[1][200:800],  color = '#ff0000')
    # plt.xlabel("Время (сек) ")
    # plt.ylabel("Амплитуда")
    # plt.grid()
    # plt.show()
    # #
   #  print("------------Alpha_norm-----------------")
   #  print("------------F_L-----------------")
   #  print("F7F3  = ", avgTwoCanal(1,2))
   #  print("F7T3  = ", avgTwoCanal(1,5))
   #  print("F7C3  = ", avgTwoCanal(1,6))
   #  print("F3T3  = ", avgTwoCanal(2,5))
   #  print("F3C3  = ", avgTwoCanal(2,6))
   #  print("T3C3  = ", avgTwoCanal(5,6))
   #  print("------------F_R-----------------")
   #  print("F8F4  = ", avgTwoCanal(4,3))
   #  print("F8C4  = ", avgTwoCanal(4,8))
   #  print("F8T4  = ", avgTwoCanal(4,9))
   #  print("F4C4  = ", avgTwoCanal(3,8))
   #  print("F4T4  = ", avgTwoCanal(3,9))
   #  print("C4T4  = ", avgTwoCanal(8,9))
   #  print("------------C_L-----------------")
   #  #print("T3C3  = ", avgTwoCanal(5, 6))
   #  print("T3T5  = ", avgTwoCanal(5, 10))
   #  print("T3P3  = ", avgTwoCanal(5, 11))
   #  print("C3T5  = ", avgTwoCanal(6, 10))
   #  print("C3P3  = ", avgTwoCanal(6, 11))
   #  print("T5P3  = ", avgTwoCanal(10, 11))
   #  print("------------C_R-----------------")
   #  #print("C4T4  = ", avgTwoCanal(8, 9))
   #  print("C4P4  = ", avgTwoCanal(8, 13))
   #  print("C4T6  = ", avgTwoCanal(8, 14))
   #  print("T4P4  = ", avgTwoCanal(9, 13))
   #  print("T4T6  = ", avgTwoCanal(9, 14))
   #  print("P4T6  = ", avgTwoCanal(13, 14))
   #  print("------------O_l-----------------")
   # # print("T5P3  = ", avgTwoCanal(10, 11))
   #  print("T5O1  = ", avgTwoCanal(10, 15))
   #  print("P3O1  = ", avgTwoCanal(11, 15))
   #  print("------------O_r-----------------")
   #  #print("P4T6  = ", avgTwoCanal(13, 14))
   #  print("T6O2  = ", avgTwoCanal(14, 16))
   #  print("P4O2  = ", avgTwoCanal(13, 16))
   #  print("------------Межполушарная-----------------")
   #  print("F3F4  = ", avgTwoCanal(2,3))
   #  print("C3C4  = ", avgTwoCanal(6,8))
   #  print("P3P4  = ", avgTwoCanal(11,13))
   #  print("O1O2  = ", avgTwoCanal(15,16))
   #  print("-----------------------------")
   #  print("F7F8  = ", avgTwoCanal(1, 4))
   #  print("T3T4  = ", avgTwoCanal(5, 9))
   #  print("T5T6  = ", avgTwoCanal(10, 14))
   #  print("-----------------------------")
   #  print("F7T5  = ", avgTwoCanal(1, 10))
   #  print("F3P3  = ", avgTwoCanal(2, 11))
   #  print("F4P4  = ", avgTwoCanal(3, 13))
   #  print("F8T6  = ", avgTwoCanal(4, 14))
   #  print("-----------------------------")
   #  print("F3O1  = ", avgTwoCanal(2, 15))
   #  print("F4O2  = ", avgTwoCanal(3, 16))
   #  print("-----------------------------")
   #  print("F3P4  = ", avgTwoCanal(2, 10))
   #  print("F4P3  = ", avgTwoCanal(3, 11))
   #  print("F7T6  = ", avgTwoCanal(1, 14))
   #  print("F8T5  = ", avgTwoCanal(4, 10))
   #  # #AmplitudeAllPac(39)
   #
   #  print("------------Alpha_shiz-----------------")
   #  print("------------F_L-----------------")
   #  print("F7F3  = ", avgTwoCanal2(1, 2))
   #  print("F7T3  = ", avgTwoCanal2(1, 5))
   #  print("F7C3  = ", avgTwoCanal2(1, 6))
   #  print("F3T3  = ", avgTwoCanal2(2, 5))
   #  print("F3C3  = ", avgTwoCanal2(2, 6))
   #  print("T3C3  = ", avgTwoCanal2(5, 6))
   #  print("------------F_R-----------------")
   #  print("F8F4  = ", avgTwoCanal2(4, 3))
   #  print("F8C4  = ", avgTwoCanal2(4, 8))
   #  print("F8T4  = ", avgTwoCanal2(4, 9))
   #  print("F4C4  = ", avgTwoCanal2(3, 8))
   #  print("F4T4  = ", avgTwoCanal2(3, 9))
   #  print("C4T4  = ", avgTwoCanal2(8, 9))
   #  print("------------C_L-----------------")
   #  print("T3C3  = ", avgTwoCanal2(5, 6))
   #  print("T3T5  = ", avgTwoCanal2(5, 10))
   #  print("T3P3  = ", avgTwoCanal2(5, 11))
   #  print("C3T5  = ", avgTwoCanal2(6, 10))
   #  print("C3P3  = ", avgTwoCanal2(6, 11))
   #  print("T5P3  = ", avgTwoCanal2(10, 11))
   #  print("------------C_R-----------------")
   #  print("C4T4  = ", avgTwoCanal2(8, 9))
   #  print("C4P4  = ", avgTwoCanal2(8, 13))
   #  print("C4T6  = ", avgTwoCanal2(8, 14))
   #  print("T4P4  = ", avgTwoCanal2(9, 13))
   #  print("T4T6  = ", avgTwoCanal2(9, 14))
   #  print("P4T6  = ", avgTwoCanal2(13, 14))
   #  print("------------O_l-----------------")
   #  print("T5P3  = ", avgTwoCanal2(10, 11))
   #  print("T5O1  = ", avgTwoCanal2(10, 15))
   #  print("P3O1  = ", avgTwoCanal2(11, 15))
   #  print("------------O_r-----------------")
   #  print("P4T6  = ", avgTwoCanal2(13, 14))
   #  print("T6O2  = ", avgTwoCanal2(14, 16))
   #  print("P4O2  = ", avgTwoCanal2(13, 16))
   #  print("------------Межполушарная-----------------")
   #  print("F3F4  = ", avgTwoCanal2(2, 3))
   #  print("C3C4  = ", avgTwoCanal2(6, 8))
   #  print("P3P4  = ", avgTwoCanal2(11, 13))
   #  print("O1O2  = ", avgTwoCanal2(15, 16))
   #  print("-----------------------------")
   #  print("F7F8  = ", avgTwoCanal2(1, 4))
   #  print("T3T4  = ", avgTwoCanal2(5, 9))
   #  print("T5T6  = ", avgTwoCanal2(10, 14))
   #  print("-----------------------------")
   #  print("F7T5  = ", avgTwoCanal2(1, 10))
   #  print("F3P3  = ", avgTwoCanal2(2, 11))
   #  print("F4P4  = ", avgTwoCanal2(3, 13))
   #  print("F8T6  = ", avgTwoCanal2(4, 14))
   #  print("-----------------------------")
   #  print("F3O1  = ", avgTwoCanal2(2, 15))
   #  print("F4O2  = ", avgTwoCanal2(3, 16))
   #  print("-----------------------------")
   #  print("F3P4  = ", avgTwoCanal2(2, 10))
   #  print("F4P3  = ", avgTwoCanal2(3, 11))
   #  print("F7T6  = ", avgTwoCanal2(1, 14))
   #  print("F8T5  = ", avgTwoCanal2(4, 10))

   #  norm = [0.6,0.59,0.6,0.61,0.56,0.61]
   #  shiz = [0.63,0.64,0.58,0.61,0.6,0.69]
   #  x =[i for i in np.arange(0,3,0.5)]
   #  my_xticks = ['F_L', 'F_R', 'C_L', 'C_R', 'O_L', 'O_R']
   #  #my_xticks = [ ' F3-4 ' , ' C3-4 ' , ' P3-4 ' , ' O1-2 ' ]
   #  plt.figure("Региональные внутриполушарные различия в delta диапазоне")
   #  #plt.figure("Межполушарная синхронность  в дельта диапазоне")
   #  #plt.grid()
   #  plt.xticks(x, my_xticks)
   # # plt.title(r'$\alpha$')
   #  #plt.title(r'$\beta1$')
   # # plt.title(r'$\beta2$')
   #  #plt.title(r'$\theta$')
   #  plt.title(r'$\delta$')
   #  plt.plot(x,norm,marker = 's',color = '#ff0000')
   #  plt.plot(x,shiz,marker = 'o',color = '#000000')
   #  plt.legend(("Норма","Шизофрения"))
   #  plt.grid()
   #  plt.show()

    # path2 = "NormBetta1\\" + str(1) + ".txt"
    # data = pd.read_csv(path2, sep=" ", header=None)
    # # w = np.fft.fft(data[1][0:11000])
    # Hn = np.fft.fft(data[2][0:N_DATA])
    # freqs = (np.fft.fftfreq(len(Hn), 1 /FREQUENCY))
    # idx = np.argmax(np.abs(Hn))
    # print(np.abs(Hn))
    # freq_in_hertz = freqs[idx]
    # print('Main freq:', freq_in_hertz)
    #
    # xF = Hn[0:5500]
    # fr = np.linspace(0, FREQUENCY/2, 5500)
    # plt.subplot(2, 1, 1)
    # x = [i for i in range(2000)]
    # plt.plot(x,data[1][0:2000])
    # plt.subplot(2, 1, 2)
    # plt.plot(fr,abs(xF)**2)
    #
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD')
    #
    #
    #
    # plt.show()




    # path2 = "27.txt"
    # data = pd.read_csv(path2, sep=" ", header=None)
    # w = np.fft.fft(data[1][0:11000])
    #
    # arg = []
    # A=[]
    # freq = []
    # for i in np.arange(0, 2000, 1):
    #     A.append((w[i].real ** 2 + (w[i].imag) ** 2))  # модуль амплитуды
    #
    # freq = np.linspace(0, FREQUENCY / 2, 5500)
    # idx = np.argmax(np.abs(w))
    # freqs = (np.fft.fftfreq(len(w), 1 / FREQUENCY))
    # freq_in_hertz = freqs[idx]
    # print('Main freq:', freq_in_hertz)
    # w = np.fft.fft(data[1][0:11000])
    # plt.figure("БПФ")
    # plt.plot(freq[0:100], w[0:100])
    # plt.xlabel("Частота")
    # plt.ylabel("Амплитуда")
    # plt.grid()
    #
    # plt.figure("Спектр мощности")
    # plt.plot(freq[0:100],A[0:100])
    # plt.grid()
    # plt.show()




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

