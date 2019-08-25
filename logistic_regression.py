import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.e ** (-z))

def logistic(x, label, mat_iter, lr):
    m, n = x.shape
    X = np.c_[(np.ones((m, 1)), x)]

    theta = np.zeros(n+1)

    for iter in range(mat_iter):
        error = sigmoid(X.dot(theta)) - label
        theta -= lr * X.T.dot(error) / m
    print(theta)

    for i in range(m):
        if label[i] == 0 :
            plt.scatter(x[i][0], x[i][1], c='red')
        else:
            plt.scatter(x[i][0], x[i][1], c='green')

    xnew = np.linspace(0, 10, 50)
    x_ = np.c_[(np.ones((50, 1)), xnew)]
    y_ = -x_.dot(theta[:-1]) / theta[-1]

    plt.plot(xnew, y_, '-g')
    plt.axis([0, 10, 0, 10])
    plt.grid()
    plt.show()

if __name__ == '__main__':
    X = np.array([[1, 2],[2,2],[3,2],[4,2],
    [1, 3],[1,4],[1,5],[1,6],
    [1.5, 2.5],[2,3.5],[2,4.5],[3,4],
    [2.5, 3],[3,3],[1.5,4],[1.5,5],
    [4.4,3.4],[5.3,2.3],[2.4,5.6],[3.9,3.0],[4.7,2.6],
    [3.7,5.2],[2,6],[3,5],[4,4],[5,3],[3.1,5.8],

    [1, 7],
    [1, 8],[2,8],[3,8],[4,8],[5,8],[6,8],
    [6, 2],[6,3],[6,4],[6,5],[6,6],[6,7],
    [2.5,7],[3,6.5],[3.7,7],[5,6.7],[4,6],
    [5, 4],[4.5,5],[5,5.5],
    [1.7,6.9],[5.3,4.7]

    ])

    label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
          1,1,1,1,1,1,1,1,1,1,1,1,1
            ,1,1,1,1,1,1,1,1,1,1
     ]

    logistic(X, label, 1000, 0.1)















