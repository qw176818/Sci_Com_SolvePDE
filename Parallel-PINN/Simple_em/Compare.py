import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:

        y.append(eval(row[1]))
        x.append(eval(row[0]))
    return x, y

if __name__ == '__main__':
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    plt.axes(yscale = "log")
    fig = plt.figure()
    x2, y2 = readcsv("./log/loss_w_1.csv")
    plt.plot(x2, y2, color='red', label='PINN(1, 2, 20)')
    plt.plot(x2, y2, '.', color='red')

    x, y = readcsv("./log/PINN_loss_w_20_2_20.csv")
    for i in range(len(y)):
        y[i] /= 100

    plt.plot(x, y, 'green', label='PINN(12, 2, 20)')

    x1, y1 = readcsv("./log/PINN_loss_w_20_4_40.csv")
    for i in range(len(y1)):
        y1[i] /= 100
    plt.plot(x1, y1, color='blue', label='PINN(12, 4, 40)')

    x4, y4 = readcsv("./log/PINN_loss_w_20_6_60.csv")
    for i in range(len(y4)):
        y4[i] /= 100
    plt.plot(x4, y4, color='orange', label='PINN(12, 6, 60)')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale("log")
    plt.ylim(0, 1.0)
    plt.xlim(0, 51000)

    plt.xlabel('Iter', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.legend(fontsize=13)
    plt.savefig("loss_cmp.pdf")
    plt.savefig("loss_cmp.eps")