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
    x, y = readcsv("./log/PINN_loss.csv")
    plt.plot(x, y, color='red', label='PINN')

    x1, y1 = readcsv("./log/PINN_loss_ada_fun.csv")
    plt.plot(x1, y1, 'green', label='PINN+ADF')

    x2, y2 = readcsv("./log/PINN_loss_ada_fun_loss.csv")
    plt.plot(x2, y2, color='blue', label='PINN+ADL')

    x3, y3 = readcsv("./log/PINN_Ada_Loss_Fun_loss.csv")
    plt.plot(x3, y3, color='orange', label='AdaPINN')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale("log")
    plt.ylim(0, 1.0)
    plt.xlim(0, 51000)

    plt.xlabel('Iter', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.legend(fontsize=13)
    plt.savefig("./figures/loss_com.pdf")
    plt.savefig("./figures/loss_com.eps")