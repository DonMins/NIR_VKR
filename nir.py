import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
import os.path as op
import numpy as np
from mne.minimum_norm import read_inverse_operator, compute_source_psd
N_DATA = 11001
NUMBER_CHANNELS = 16
FREQUENCY = 400

def EegChart(data):
    mas = np.zeros((NUMBER_CHANNELS, N_DATA))

    for i in range(1, NUMBER_CHANNELS+1):
        allMax = 0
        mmax = abs(max(data[i][0:N_DATA]))
        mmin = abs(min(data[i][0:N_DATA]))
        allMax = max(mmax, mmin)

        for j in range(N_DATA):
            mas[i - 1][j] = (10 * (data[i][j]) / allMax) # для нормального отображение графиков сделаем нормировку

    data = np.array([mas[0], mas[1], mas[2], mas[3], mas[4], mas[5], mas[6], mas[7],
                     mas[8], mas[9], mas[10], mas[11], mas[12], mas[13], mas[14], mas[15]])
    info = mne.create_info(
        ch_names=['F7', 'F3', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg'],
        sfreq= FREQUENCY
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

    for i in range(N_DATA + 1):
        j = i + 1
        m = j + 1

        i1 = (data[number][i])
        i2 = (data[number][j])
        i3 = (data[number][m])

        if ((abs(i2) > abs(i1)) & (abs(i2) > abs(i3))):
            k = k + 1
            sum = sum + abs(i2)

    return sum / k

def matOj(input):
    avg = 0
    for j in range(len(input)):
        avg += (input[j])
    return avg / N_DATA

# нахождение дисперсии
def dispers(input):
    disper = 0
    avg = matOj(input)
    for i in range(len(input)):
        disper += (input[i] - avg) ** 2
    return disper / (N_DATA - 1)

correlValues = []


# подсчёт корреляционной функции
def correlfunction(input):
    avg = matOj(input)
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

    for i in range(1, N+1):
        path1 = "Norm\\" + str(i) + ".txt"
        path2 = "Shiz\\" + str(i) + ".txt"
        tmp = 0
        tmp2 = 0
        norm = pd.read_csv(path1, sep=" ", header=None)
        shiz = pd.read_csv(path2, sep=" ", header=None)
        for k in range(1, NUMBER_CHANNELS+1):
            arrayNorm.append(average(norm, k))
            arrayShiz.append(average(shiz, k))
        for j in range(NUMBER_CHANNELS):
            tmp = tmp + arrayShiz[j]
            tmp2 = tmp2 + arrayNorm[j]
        graficArrayShiz.append(tmp / len(arrayShiz))
        graficArrayNorm.append(tmp2 / len(arrayNorm))

        print("Пациенты № " + str(i))
        print("Норма                       Shiz ")
        print("F7 = {0}    F7 = {1}:  ".format(arrayNorm[0], arrayShiz[0]))
        print("F3 = {0}    F3 {1} :  ".format(arrayNorm[1], arrayShiz[1]))
        print("F4 = {0}    F4 {1} :  ".format(arrayNorm[2], arrayShiz[2]))
        print("F8 = {0}    F8 {1} :  ".format(arrayNorm[3], arrayShiz[3]))
        print("T3 = {0}    T3 {1} :  ".format(arrayNorm[4], arrayShiz[4]))
        print("C3 = {0}    C3 {1} :  ".format(arrayNorm[5], arrayShiz[5]))
        print("Cz = {0}    Cz {1} :  ".format(arrayNorm[6], arrayShiz[6]))
        print("C4 = {0}    C4 {1} :  ".format(arrayNorm[7], arrayShiz[7]))
        print("T4 = {0}    T4 {1} :  ".format(arrayNorm[8], arrayShiz[8]))
        print("T5 = {0}    T5 {1} :  ".format(arrayNorm[9], arrayShiz[9]))
        print("P3 = {0}    P3 {1} :  ".format(arrayNorm[10], arrayShiz[10]))
        print("Pz = {0}    Pz {1} :  ".format(arrayNorm[11], arrayShiz[11]))
        print("P4 = {0}    P4 {1} :  ".format(arrayNorm[12], arrayShiz[12]))
        print("T6 = {0}    T6 {1} :  ".format(arrayNorm[13], arrayShiz[13]))
        print("O1 = {0}    O1 {1} :  ".format(arrayNorm[14], arrayShiz[14]))
        print("O2 = {0}    O2 {1} :  ".format(arrayNorm[15], arrayShiz[15]))

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

def cor12(x,y):
    sum=0
    for i in range(N_DATA):
        pr =x[i]*y[i]
        sum+=pr
    return sum/N_DATA



if __name__ == "__main__":

    path1 = "Shiz\\" + str(1) + ".txt"
    data = pd.read_csv(path1, sep=" ", header=None)
    arrayNorm = []
    arrayShiz = []
    graficArrayNorm = []
    graficArrayShiz = []
    namePara = {'F7F3    F3F4   F4F8  F8T3  T3C3  C3Cz  C4T4   T4T5  T5P3  P3Pz  PzP4  P4T6  T6O  O1O2'}
    synchroMatrixNorm = np.zeros((10, 15))
    synchroMatrixShiz = np.zeros((10, 15))

    for i in range(1, 10):
        path1 = "Norm\\" + str(i) + ".txt"
        path2 = "Shiz\\" + str(i) + ".txt"
        tmp = 0
        tmp2 = 0
        norm = pd.read_csv(path1, sep=" ", header=None)
        shiz = pd.read_csv(path2, sep=" ", header=None)


        for j in range(1,16):
            synchroMatrixNorm[i-1][j-1] = round(cor12(norm[j],norm[j+1]),2)
            #synchroMatrixShiz[i-1][j-1] = cor12(shiz[i],shiz[i+1])

    print(namePara)
    for i in range(9):
        print(*[synchroMatrixNorm[i, j] for j in range(15)])






