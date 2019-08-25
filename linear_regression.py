import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets


def use_sklearn(x, y):
    x=x[:, np.newaxis]
    linear = LinearRegression()
    linear.fit(x, y)
    print([linear.intercept_, linear.coef_])

    predict = linear.predict(x)
    plt.scatter(x, y, c='g', marker='o', linewidths=1)
    plt.plot(x, predict, c='r', linewidth=3)
    plt.show()

def normal_equation(x, y):
    # X = np.hstack((np.ones((len(x), 1)), x[:, np.newaxis]))
    X = np.c_[np.ones((len(x), 1)), x[:, np.newaxis]]
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(theta)

    predict = X.dot(theta)
    plt.scatter(x, y, c='g', marker='o', linewidths=1)
    plt.plot(x, predict, c='r', linewidth=3)
    plt.show()

def gradient_decent1(x, y, max_iter, alpha):
    # X = np.hstack((np.ones((len(x), 1)), x[:, np.newaxis]))
    X = np.c_[np.ones((len(x), 1)), x[:, np.newaxis]]

    m = X.shape[0]
    theta = np.zeros(X.shape[1])

    for iter in range(max_iter):
        error = X.dot(theta) - y
        theta -= alpha * X.T.dot(error) / m
        predict = X.dot(theta)

        plt.clf()
        plt.scatter(x, y, c='g', marker='o', linewidths=1)
        plt.plot(x, predict, c='r', linewidth=3)
        plt.pause(0.09)

    print(theta)
    plt.show()

def gradient_decent2(x, y, max_iter, alpha):
    # X = np.hstack((np.ones((len(x), 1)), x[:, np.newaxis]))
    X = np.c_[np.ones((len(x), 1)), x[:, np.newaxis]]
    Y = y[:, np.newaxis]

    m = X.shape[0]
    theta = np.zeros((X.shape[1], 1))

    for iter in range(max_iter):
        error = X.dot(theta) - Y
        # theta -= alpha * np.sum(X * error, axis=0)[:, np.newaxis] / m
        theta -= alpha * X.T.dot(error) / m

        predict = X.dot(theta)
        plt.clf()
        plt.scatter(x, y, c='g', marker='o', linewidths=1)
        plt.plot(x, predict, c='r', linewidth=3)
        plt.pause(0.05)

    print(theta)
    plt.show()




if __name__ == '__main__':
    x = np.loadtxt('data.txt', delimiter=' ', dtype=float, usecols=[0])
    y = np.loadtxt('data.txt', delimiter=' ', dtype=float, usecols=[1])

    use_sklearn(x, y)
    normal_equation(x, y)
    # gradient_decent1(x, y,20, 0.5)
    # gradient_decent2(x, y, 20, 0.5)






