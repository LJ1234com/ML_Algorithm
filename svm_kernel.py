import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def svm(data, labels, maxiter, lr):
    m, n = data.shape
    x = np.c_[np.ones(m), data]
    theta = np.zeros(n + 1)

    minx = 2  #-1
    maxx = 8  #2
    for iter in range(maxiter):
        for i in range(m):
            if labels[i] * x[i].dot(theta) > 1:
                deriviate = 0
            else:
                deriviate = -x[i] * labels[i]
            theta -= lr * deriviate
        plt.clf()
        plt.scatter(x[:, 1], x[:, 2], c=labels)
        plt.plot([minx, maxx],
                 [(1 - theta[0] - minx * theta[1]) / theta[2], (1 - theta[0] - maxx * theta[1]) / theta[2]])
        plt.pause(0.1)
    plt.show()

    for i in range(m):
        if x[i].dot(theta) > 1:
            plt.scatter(x[i, 1], x[i, 2], c='r')
        else:
            plt.scatter(x[i, 1], x[i, 2], c='g')
    plt.show()
    # print(1-data.dot(theta))
    print(theta)


def svm_soft(data, labels, maxiter, lr, C, llambda):
    m, n = data.shape
    x = np.c_[np.ones(m), data]
    theta = np.zeros(n + 1)

    minx = 2  #-1
    maxx = 8  #2
    for iter in range(maxiter):
        for i in range(m):
            if labels[i] * x[i].dot(theta) > 1:
                deriviate = 0
            else:
                deriviate = -x[i] * labels[i]
            theta -= C * lr * deriviate
            theta[1:] -= llambda * theta[1:]
        plt.clf()
        plt.scatter(x[:, 1], x[:, 2], c=labels)
        plt.plot([minx, maxx],
                 [(1 - theta[0] - minx * theta[1]) / theta[2], (1 - theta[0] - maxx * theta[1]) / theta[2]])
        plt.pause(0.1)
    plt.show()

    for i in range(m):
        if x[i].dot(theta) > 1:
            plt.scatter(x[i, 1], x[i, 2], c='r')
        else:
            plt.scatter(x[i, 1], x[i, 2], c='g')
    plt.show()
    # print(1-data.dot(theta))
    print(theta)

def svm_kernel(data, labels, maxiter, lr):
    m, n = data.shape
    l = data
    F = np.zeros((m, m))   # K=[f1, f2, ..., fm]
    sigma = 1
    for i in range(m):
        for j in range(m):
            f = np.exp(-sum((data[i]-l[j])**2)/(2*sigma**2))
            F[i, j] = f
    print(F)

    X = F.T
    x = np.c_[np.ones(m), X]
    theta = np.zeros(m + 1)


    for iter in range(maxiter):
        for i in range(m):
            if labels[i] * x[i].dot(theta) > 1:
                deriviate = 0
            else:
                deriviate = -x[i] * labels[i]
            theta -=  lr * deriviate
    #     plt.clf()
    #     plt.scatter(x[:, 1], x[:, 2], c=labels)
    #     plt.plot([minx, maxx],
    #              [(1 - theta[0] - minx * theta[1]) / theta[2], (1 - theta[0] - maxx * theta[1]) / theta[2]])
    #     plt.pause(0.1)
    # plt.show()
    #

    w = (theta[1:] * labels).dot(data)
    for i in range(m):
        if labels[i]*(data[i].dot(w)+theta[0]) > 1:
            plt.scatter(data[i, 0], data[i, 1], c='r')
        else:
            plt.scatter(data[i, 0], data[i, 1], c='g')
    plt.show()
    print(theta)

    ## OR:
    for i in range(m):
        if x[i].dot(theta) > 1:
            plt.scatter(data[i, 0], data[i, 1], c='r')
        else:
            plt.scatter(data[i, 0], data[i, 1], c='g')
    plt.show()



if __name__ == '__main__':
    data = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[0, 1])
    labels = np.loadtxt('svm.txt', dtype=float, delimiter=' ', usecols=[2])
    # data, labels = datasets.make_classification(n_samples=300, n_features=2, n_redundant=0, n_classes=2,
    #                             class_sep=1.5, random_state=225, n_clusters_per_class=1)
    # labels[np.nonzero(labels==0)[0]] = -1

    # svm(data, labels, 100, 0.01)
    # svm_soft(data, labels, 50, 0.01, C=0.1, llambda=0)

    data = np.loadtxt('svm_kernel1.txt', dtype=float, delimiter=' ', usecols=[0, 1])
    labels = np.loadtxt('svm_kernel1.txt', dtype=float, delimiter=' ', usecols=[2])
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()

    svm_kernel(data, labels, 10000, 0.01)