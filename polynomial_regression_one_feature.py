import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import copy
from numba import jit
import time


def sgn(x):
    if x >= 0:
        return 1.0
    else:
        return -1.0

def sklearn_linear(x,y, d=None):
    model1 = LinearRegression()
    model1.fit(x, y)

    poly = PolynomialFeatures(d, interaction_only=False, include_bias=True)
    X = poly.fit_transform(x)
    model2 = LinearRegression()
    model2.fit(X, y)

    m = x.shape[0]
    lr = 0.01
    theta = np.zeros(X.shape[1])

    for iter in range(100000):
        error = X.dot(theta) - y
        theta -= lr * X.T.dot(error) / m
        theta[1:] -=  0.01 * theta[1:] / m    # L2 Regulization
        # for i in range(1, d+1):             # L1 Regulization
        #     theta[i] -= 0.1 * sgn(theta[i]) / m


    print(theta)
    x_new = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    X_new = poly.fit_transform(x_new)
    predict = X_new.dot(theta)

    plt.scatter(x, y, c='green')
    plt.plot(x_new, model1.predict(x_new), 'r', linewidth=2)
    plt.plot(x_new, model2.predict(X_new), 'b', linewidth=2)
    plt.plot(x_new, predict, '--', c='r', linewidth=2)

    # plt.show()


if __name__ == '__main__':
    t1 = time.time()
    x, y = datasets.make_regression(n_samples=10, n_features=1, noise=10, n_targets=1, random_state=13)
    sklearn_linear(x, y, d=6)
    print(time.time()-t1)














