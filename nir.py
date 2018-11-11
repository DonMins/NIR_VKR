import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
import os.path as op
import numpy as np
N = 11001
time = []



# Numpy array of size 4 X 10000.

path1 = "Shiz\\" + str(1) + ".txt"
norm = pd.read_csv(path1, sep=" ", header=None)

mas = np.zeros((16, N))


for i in range(1,17):
    allMax=0
    mmax = abs(max(norm[i][0:N]))
    mmin = abs(min(norm[i][0:N]))
    allMax = max(mmax,mmin)

    for j in range(N):
        mas[i-1][j]=(10*(norm[i][j])/allMax)


data = np.array([mas[0],mas[1],mas[2],mas[3],mas[4],mas[5],mas[6],mas[7],
                 mas[8],mas[9],mas[10],mas[11],mas[12],mas[13],mas[14],mas[15]])
# Definition of channel types and names.

info = mne.create_info(
    ch_names=['F7','F3','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
    ch_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'],
    sfreq=400
)
raw = mne.io.RawArray(data, info)
scalings = {'eeg': 16}

calings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=16, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)


def average(data, number):
    i1 = 0
    i2 = 0
    i3 = 0
    k = 0
    sum = 0

    for i in range(N+1):
        j = i + 1
        m = j + 1

        i1 = (data[number][i])
        i2 = (data[number][j])
        i3 = (data[number][m])

        if ((abs(i2) > abs(i1)) & (abs(i2) > abs(i3))):
            k = k + 1
            sum = sum + abs(i2)

    return sum / k

arrayNorm = []
arrayShiz = []
graficArrayNorm=[]
graficArrayShiz=[]
#
# for i in range(1,40):
#
#     path1 = "Norm\\"+str(i)+".txt"
#     path2 = "Shiz\\"+str(i)+".txt"
#     tmp=0
#     tmp2=0
#     norm = pd.read_csv(path1, sep=" ", header=None)
#     shiz = pd.read_csv(path2, sep=" ", header=None)
#     for k in range(1,17):
#         arrayNorm.append(average(norm,k))
#         arrayShiz.append(average(shiz,k))
#     for j in range(16):
#         tmp = tmp + arrayShiz[j]
#         tmp2 = tmp2 + arrayNorm[j]
#     graficArrayShiz.append(tmp/len(arrayShiz))
#     graficArrayNorm.append(tmp2/len(arrayNorm))
#
#     print("Пациенты № "+str(i))
#     print("Норма                       Shiz ")
#     print("F7 = {0}    F7 = {1}:  ".format(arrayNorm[0],arrayShiz[0]))
#     print("F3 = {0}    F3 {1} :  ".format(arrayNorm[1],arrayShiz[1]))
#     print("F4 = {0}    F4 {1} :  ".format(arrayNorm[2],arrayShiz[2]))
#     print("F8 = {0}    F8 {1} :  ".format(arrayNorm[3],arrayShiz[3]))
#     print("T3 = {0}    T3 {1} :  ".format(arrayNorm[4],arrayShiz[4]))
#     print("C3 = {0}    C3 {1} :  ".format(arrayNorm[5],arrayShiz[5]))
#     print("Cz = {0}    Cz {1} :  ".format(arrayNorm[6],arrayShiz[6]))
#     print("C4 = {0}    C4 {1} :  ".format(arrayNorm[7],arrayShiz[7]))
#     print("T4 = {0}    T4 {1} :  ".format(arrayNorm[8],arrayShiz[8]))
#     print("T5 = {0}    T5 {1} :  ".format(arrayNorm[9],arrayShiz[9]))
#     print("P3 = {0}    P3 {1} :  ".format(arrayNorm[10],arrayShiz[10]))
#     print("Pz = {0}    Pz {1} :  ".format(arrayNorm[11],arrayShiz[11]))
#     print("P4 = {0}    P4 {1} :  ".format(arrayNorm[12],arrayShiz[12]))
#     print("T6 = {0}    T6 {1} :  ".format(arrayNorm[13],arrayShiz[13]))
#     print("O1 = {0}    O1 {1} :  ".format(arrayNorm[14],arrayShiz[14]))
#     print("O2 = {0}    O2 {1} :  ".format(arrayNorm[15],arrayShiz[15]))
#
#     norm=[]
#     shiz=[]
#     arrayNorm=[]
#     arrayShiz=[]
#
#
# # plt.figure(1)
# plt.figure("MRI_with_EEG")
# x = [i for i in range(1,40)]
# leg1, leg2=plt.plot(x,graficArrayNorm ,'o' ,x,graficArrayShiz, 'o' )
# plt.legend((leg1,leg2),("Norm","Shiz"))
# plt.grid(True)
# plt.show()
#
# plt.figure("MRI_with_EEG2")
# leg1, leg2=plt.plot(x,graficArrayNorm ,x,graficArrayShiz )
# plt.legend((leg1,leg2),("Norm","Shiz"))
# plt.grid(True)
# plt.show()

#
# plt.figure(2)
# plt.plot(shiz[0][0:100], shiz[1][0:100])
# plt.grid()
#



def matOj(input):
    avg = 0
    for j in range(len(input)):
        avg += (input[j])
    return avg / N

#
# print("Мат.ожидание норм [1] ", matOj(norm[1]))
# print("Мат.ожидание shiz [1] ", matOj(shiz[1]))
# print("Мат.ожидание норм [2] ", matOj(norm[2]))
# print("Мат.ожидание shiz [2] ", matOj(shiz[2]))


# нахождение дисперсии
def dispers(input):
    disper = 0
    avg = matOj(input)
    for i in range(len(input)):
        disper += (input[i] - avg) ** 2
    return disper / (N - 1)


# print("Дисп норм [1] ", dispers(norm[1]))
# print("Дисп shiz [1] ", dispers(shiz[1]))
# print("Дисп норм [2] ", dispers(norm[2]))
# print("Дисп shiz [2] ", dispers(shiz[2]))
correlValues = []


# подсчёт корреляционной функции
def correlfunction(input):
    avg = matOj(input)
    for i in range((len(input)) - 1):
        temp = 0
        for j in range(len(input) - i):
            temp += (input[j] - avg) * (input[j + i] - avg)
        correlValues.append((1 / (N - i - 1)) * temp)





# print("Норма                      Shiz ")
# print("F7 = {0}    F7 = {1}:  ".format(average(norm, 1), average(shiz, 1)))
# print("F3 = {0}    F3 {1} :  ".format(average(norm, 2), average(shiz, 2)))
# print("F4 = {0}    F4 {1} :  ".format(average(norm, 3), average(shiz, 3)))
# print("F8 = {0}    F8 {1} :  ".format(average(norm, 4), average(shiz, 4)))
# print("T3 = {0}    T3 {1} :  ".format(average(norm, 5), average(shiz, 5)))
# print("C3 = {0}    C3 {1} :  ".format(average(norm, 6), average(shiz, 6)))
# print("Cz = {0}    Cz {1} :  ".format(average(norm, 7), average(shiz, 7)))
# print("C4 = {0}    C4 {1} :  ".format(average(norm, 8), average(shiz, 8)))
# print("T4 = {0}    T4 {1} :  ".format(average(norm, 9), average(shiz, 9)))
# print("T5 = {0}    T5 {1} :  ".format(average(norm, 10), average(shiz, 10)))
# print("P3 = {0}    P3 {1} :  ".format(average(norm, 11), average(shiz, 11)))
# print("Pz = {0}    Pz {1} :  ".format(average(norm, 12), average(shiz, 12)))
# print("P4 = {0}    P4 {1} :  ".format(average(norm, 13), average(shiz, 13)))
# print("T6 = {0}    T6 {1} :  ".format(average(norm, 14), average(shiz, 14)))
# print("O1 = {0}    O1 {1} :  ".format(average(norm, 15), average(shiz, 15)))
# print("O2 = {0}    O2 {1} :  ".format(average(norm, 16), average(shiz, 16)))

# def chast():
#     i1=0
#     list1=[]
#     list0 = []
#     tmp=[]
#     i2=0
#     st = ''
#     st0 = ''
#
#     for i in range(200):
#       j = i + 1
#       i1 = (data[16][i])
#       i2 = (data[16][j])
#
#       if(i2>=i1):
#           if (len(st0)!= 0):
#               list0.append(st0)
#               st0 = ''
#           st= st+'1'
#       if(i2<i1):
#           if(len(st)!=0):
#               list1.append(st)
#               st=''
#           st0 = st0+'0'


# mas = []
# for i in range (100):
#     mas.append(data[1][i])
#
# plt.figure(2)
# plt.grid()
# masf = (fft(mas,len(mas),axis=-1))
# ms = [value.real for value in masf]
#
# print (ms)
# plt.plot(data[0][0:100],ms)
# plt.figure(3)
# masf3=irfft(masf, len(ms), axis=-1)
#
# plt.plot(data[0][0:100],masf3)
# # plt.grid()
# # plt.show()
#
