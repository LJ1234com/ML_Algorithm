import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets


def use_sklearn(x, y):
    model = LinearRegression()
    model.fit(x, y)
    print(model.intercept_, model.coef_)

def normal_equation(x, y):
    X = np.c_[np.ones((len(x), 1)), x]
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(theta)


def gradient_decent(x, y, maxiter, lr, type, batch_size=None):
    m = x.shape[0]
    X = np.c_[(np.ones((m, 1)), x)]
    theta = np.zeros(X.shape[1])

    if type == 'BGD':
        for iter in range(maxiter):
            error = X.dot(theta) - y
            # for i in range(len(theta)):
            #         theta[i] -= lr * X.T[i].dot(error) / m
            theta -= lr * X.T.dot(error) / m
        print(theta)

    if type == 'SGD':
        for iter in range(maxiter):
            for i in range(m):
                error = X[i].dot(theta) - y[i]
                # theta -= lr * error * X[i]
                theta -= lr * X[i].T.dot(error)
        print(theta)

    if type == 'mini_BGD':
        batches = [[X[i:i+batch_size], y[i:i+batch_size]] for i in range(0, m, batch_size)]
        for x_b, y_b in batches:
            error = x_b.dot(theta) - y_b
            theta -= lr * x_b.T.dot(error) / batch_size
        print(theta)



if __name__ == '__main__':
    x, y = datasets.make_regression(n_samples=10000, n_features=3, n_targets=1, noise=25)

    use_sklearn(x, y)
    normal_equation(x, y)
    gradient_decent(x, y, 100, 0.1, 'BGD')
    gradient_decent(x, y, 100, 0.005, 'SGD')
    gradient_decent(x, y, 100, 0.1, 'mini_BGD', batch_size=100)



