#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def kernel(X, A, kTup):
    """
    通过核函数将数据转换更高维的空间
    Parameters：
        X - 数据矩阵
        A - 单个数据的向量
        kTup - 包含核函数信息的元组
    Returns:
        K - 计算的核K
    """
    m, n = np.shape(X)
    K = np.zeros((m, 1))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积。
    elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            K[j] = (X[j, :] - A).dot(X[j, :] - A)
        K = np.exp(K / (-1 * kTup[1] ** 2))  # 计算高斯核K
    else:
        raise NameError('核函数无法识别')

    return K.flatten()  # 返回计算的核K


class global_params:
    def __init__(self, data, labels, C, toler):
        self.data = data
        self.m = len(self.data)
        self.labels = labels
        self.C = C
        self.toler = toler
        self.alphas = np.zeros(len(self.data))
        self.b = 0
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):  # 计算所有数据的核K
            self.K[:, i] = kernel(self.data, self.data[i, :], kTup=("rbf", 1.3))


def random_select(ii, os):
    jj = ii
    while ii == jj:
        jj = np.random.randint(0, len(os.data), 1)
    return jj[0]


def cal_err(k, os):
    fxk = sum(os.alphas * os.labels * os.K[:,k]) + os.b
    return fxk - os.labels[k]

def select_j(i, ei, os):
    maxk = 0
    max_delta_err = -np.inf
    ej = 0
    valid_ecache_list = np.nonzero((os.alphas > 0) * (os.alphas < os.C))[0]  # 在违反KKT中去找
    if len(valid_ecache_list) > 0:            # 有违反KKT的样本
        for k in valid_ecache_list:
            if i == k:
                continue
            ek = cal_err(k, os)
            delta_err = abs(ek - ei)
            if delta_err > max_delta_err:
                max_delta_err = delta_err
                maxk = k
                ej = ek
        return maxk, ej
    else:                                  # 没有违反KKT的样本， 则随机选取
        jj = random_select(i, os)
        ej = cal_err(jj, os)
        return jj, ej


def clipped(aj, l, h):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def inner(i, os):
    # 步骤1： 找到不满足KKT条件的i， 并计算其误差Ei
    ei = cal_err(i, os)
    if (os.labels[i] * ei < -os.toler and os.alphas[i] < os.C) or (os.labels[i] * ei > os.toler and os.alphas[i] > 0):

        # 步骤2： 按maxEi选取j， 并计算误差
        j, ej = select_j(i, ei, os)
        alpha_i_old = os.alphas[i].copy()
        alpha_j_old = os.alphas[j].copy()

        # 步骤3： 计算边界
        if os.labels[i] != os.labels[j]:
            low = max(0, os.alphas[j] - os.alphas[i])
            high = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            low = max(0, os.alphas[j] + os.alphas[i] - os.C)
            high = min(os.C, os.alphas[j] + os.alphas[i])
        if low == high:
            print('low == high')
            return 0

        # 步骤4： 计算eta,  这里的eta是公式推导中的-eta
        eta = 2.0 * os.K[i,j] - os.K[i,i] - os.K[j,j]
        if eta >= 0:  # 公式推导中eta<=0的情况
            print('eta >= 0')
            return 0

        # 步骤5： 计算并修剪alpha_j
        os.alphas[j] -= os.labels[j] * (ei - ej) / eta
        os.alphas[j] = clipped(os.alphas[j], low, high)
        if abs(os.alphas[j] - alpha_j_old) < 0.00001:
            print("alpha_j变化太小")
            return 0

        # 步骤6：更新alpha_i
        os.alphas[i] += os.labels[j] * os.labels[i] * (alpha_j_old - os.alphas[j])

        #步骤7：计算b1，b2， b
        b1 = os.b - ei - os.labels[i] * (os.alphas[i] - alpha_i_old) * os.K[i,i] - os.labels[j] * (
                os.alphas[j] - alpha_j_old) * os.K[i,j]
        b2 = os.b - ej - os.labels[i] * (os.alphas[i] - alpha_i_old) * os.K[i,j] - os.labels[j] * (
                os.alphas[j] - alpha_j_old) * os.K[j,j]

        # 步骤8：根据b_1和b_2更新b
        if (0 < os.alphas[i]) and (os.C > os.alphas[i]):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.C > os.alphas[j]):
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0



def smo(data, labels, C, toler, maxiter):
    '''首先遍历整个样本集，选择违反KKT条件的αi作为第一个变量，接着依据相关规则选择第二个变量(见下面分析),对这两个变量采用上述方法进行优化。
    当遍历完整个样本集后，遍历非边界样本集(0<αi<C)中违反KKT的αi作为第一个变量，同样依据相关规则选择第二个变量，对此两个变量进行优化。
    当遍历完非边界样本集后，再次回到遍历整个样本集中寻找，即在整个样本集与非边界样本集上来回切换，寻找违反KKT条件的αi作为第一个变量。
    直到遍历整个样本集后，没有违反KKT条件αi，然后退出。'''
    os = global_params(data, labels, C, toler)
    iter = 0
    entire = True
    alpha_changed = 0
    while iter < maxiter and (alpha_changed > 0 or entire):
        alpha_changed = 0
        if entire:
            for i in range(len(data)):
                alpha_changed += inner(i, os)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_changed))
            iter += 1
        else:
            non_boundary = np.nonzero((os.alphas > 0) * (os.alphas < os.C))[0]  # *表示取两个条件的交集, 两个条件一定要括起来
            for i in non_boundary:
                alpha_changed += inner(i, os)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alpha_changed))
            iter += 1

        if entire:
            entire = False
        elif alpha_changed == 0:
            entire = True
        print("迭代次数: %d" % iter)
    return os.alphas, os.b


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
    # data = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[0, 1])
    # labels = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[2])
    # alphas, b = smo(data, labels, C=0.6, toler=0.001, maxiter=400)
    # w = (alphas * labels).dot(data)
    # print(w, b)
    # show(data, labels, w, b)

    data = np.loadtxt('svm_kernel1.txt', dtype=float, delimiter=' ', usecols=[0, 1])
    labels = np.loadtxt('svm_kernel1.txt', dtype=float, delimiter=' ', usecols=[2])
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()

    alphas, b = smo(data, labels, C=0.6, toler=0.001, maxiter=400)
    w = (alphas * labels).dot(data)
    print(w, b)

    x = data
    for i in range(len(data)):
        if labels[i]*(x[i].dot(w) + b) > 1:
            plt.scatter(x[i, 0], x[i, 1], c='r')
        else:
            plt.scatter(x[i, 0], x[i, 1], c='g')
    plt.show()

