import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA(x, y, nComponents=None):
    m = len(x)
    labels = list(set(y))
    n_class = len(labels)

    classes = []
    for i in range(n_class):
        index = np.nonzero(y==i)
        classes.append(x[index])

    ## cal mean
    mean_all = np.mean(x, axis=0)
    mean_classes = [np.mean(t, axis=0) for t in classes]

    ## X Zero-centered
    x_zeros = x - mean_all
    x_classes_zeros = [classes[i] - mean_classes[i] for i in range(n_class)]

    ## 全局散度矩阵
    st = x_zeros.T.dot(x_zeros)

    ## cal 类内散度矩阵（within-class scatter matrix）
    sw = sum(x_classes_zeros[i].T.dot(x_classes_zeros[i]) for i in range(n_class))

    ## cal 类间散度矩阵（between-class scatter matrix）
    sb = st - sw

    ## cal eigen
    p = np.linalg.inv(sw).dot(sb)
    eigen_vals, eigen_vecs = np.linalg.eig(p)
    print(eigen_vals)
    print(eigen_vecs)

    sorted = np.argsort(eigen_vals)
    w = eigen_vecs[:, sorted[::-1][:nComponents]]
    # w = eigen_vecs[:, list(reversed(sorted))[:nComponents]]
    print(w)

    x_new = x.dot(w)
    data_plot2d(x_new, y)

def sklearn_lda(x, y, nComponent=None):
    lda = LinearDiscriminantAnalysis(n_components=nComponent)
    lda.fit(X, y)
    newx = lda.transform(X)
    data_plot2d(newx, y)



def data_plot2d(x, y):
    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], c='r')
        if y[i] == 1:
            plt.scatter(x[i][0], x[i][1], c='b')
        if y[i] == 2:
            plt.scatter(x[i][0], x[i][1], c='g')
    plt.show()



def data_plot3d(x, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax = plt.axes(projection='3d')
    for i in range(len(x)):
        if y[i] == 0:
            ax.scatter(x[i][0], x[i][1], x[i][2], c='r')
        if y[i] == 1:
            ax.scatter(x[i][0], x[i][1], x[i][2], c='b')
        if y[i] == 2:
            ax.scatter(x[i][0], x[i][1], x[i][2], c='g')
    plt.show()

if '__main__' == __name__:
    X, y = datasets.make_classification(n_samples=500, n_features=3, n_redundant=0, n_classes=3,
                               n_informative=2, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target

    # data_plot3d(X, y)
    LDA(X, y, nComponents=2)
    sklearn_lda(X, y, nComponent=2)


