#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot(x, labels):
    colors = ListedColormap(['b', 'g'])
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap=colors)
    plt.show()

def random_j(i, m):
    j = i
    while j == i:
        j = np.random.randint(0, m, 1)
    return j[0]

def clipped(aj, l, h):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def get_b(alphai, alphaj, b1, b2, C):
    if (0 < alphai) and (C > alphai):
        return b1
    elif (0 < alphaj) and (C > alphaj):
        return b2
    else:
        return (b1 + b2) / 2.0


def smoSimple(data, labels, C, tolerance, iters):
    m, n = data.shape
    alphas = np.zeros(m)
    b = 0
    iter = 0
    while iter < iters:
        alpha_changed = 0
        for i in range(m):
            # 步骤1： 找一个不满足KKT 条件的i，并计算误差Ei
            fxi = sum(alphas * labels * data.dot(data[i])) + b
            ei = fxi - labels[i]
            if (labels[i] * ei < -tolerance and alphas[i] < C) or (labels[i] * ei > tolerance and alphas[i] > 0):
                # 步骤2： 随机选取一个与i不同的j， 并计算误差
                j = random_j(i, m)
                fxj = sum(alphas * labels * data.dot(data[j])) + b
                ej = fxj - labels[j]

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # 步骤3： 计算边界
                if labels[i] != labels[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(C, C + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - C)
                    high = min(C, alphas[j] + alphas[i])
                if low == high:
                    print('low == high')
                    continue

                # 步骤4： 计算eta,  这里的eta是公式推导中的-eta
                eta = 2.0 * data[i].dot(data[j]) - data[i].dot(data[i]) - data[j].dot(data[j])
                if eta >= 0:           #公式推导中eta<=0的情况
                    print('eta >= 0')
                    continue

                # 步骤5： 计算并修剪alpha_j
                alphas[j] -= labels[j] * (ei - ej) / eta
                alphas[j] = clipped(alphas[j], low, high)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("alpha_j变化太小")
                    continue

                # 步骤6：更新alpha_i
                alphas[i] += labels[j] * labels[i] * (alpha_j_old - alphas[j])

                # 步骤7：计算b1，b2， b
                b1 = b - ei- labels[i]*(alphas[i]-alpha_i_old)*data[i].dot(data[i]) - labels[j]*(alphas[j]-alpha_j_old)*data[i].dot(data[j])
                b2 = b - ej- labels[i]*(alphas[i]-alpha_i_old)*data[i].dot(data[j]) - labels[j]*(alphas[j]-alpha_j_old)*data[j].dot(data[j])

                #步骤8：根据b_1和b_2更新b
                b = get_b(alphas[i], alphas[i], b1, b2, C)

                alpha_changed += 1

        if alpha_changed == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas


def show(dataArr, labelArr, w, b):
    plt.plot(dataArr[np.nonzero(labelArr == -1)[0]][:, 0], dataArr[np.nonzero(labelArr == -1)[0]][:, 1], 'or')
    plt.plot(dataArr[np.nonzero(labelArr == 1)[0]][:, 0], dataArr[np.nonzero(labelArr == 1)[0]][:, 1], 'Dg')
    minY = min(dataArr[:, 1])
    maxY = max(dataArr[:, 1])
    plt.plot([-(b + w[1] * minY) / w[0], -(b + w[1] * maxY) / w[0]], [minY, maxY])
    plt.plot([( 1 - b - w[1] * minY) / w[0], ( 1 - b - w[1] * maxY) / w[0]], [minY, maxY])
    plt.plot([(-1 - b - w[1] * minY) / w[0], (-1 - b - w[1] * maxY) / w[0]], [minY, maxY])
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[0, 1])
    labels = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[2])

    # plot(data, labels)
    b, alphas = smoSimple(data, labels, 0.6, 0.001, 40)
    w = (alphas * labels).dot(data)
    print(w, b)
    show(data, labels, w, b)


