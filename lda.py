import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import sklearn.datasets as datasets

def LDA(x, y, nComponents=None):
    m = len(x)
    X1 = np.array([x[i] for i in range(m) if y[i] == 0])
    X2 = np.array([x[i] for i in range(m) if y[i] == 1])
    X3 = np.array([x[i] for i in range(m) if y[i] == 2])

    ## cal centroid for each group and global centroid
    u = np.mean(x, axis=0)
    u1 = np.mean(X1, axis=0)
    u2 = np.mean(X2, axis=0)
    u3 = np.mean(X3, axis=0)

    ## X Zero-centered
    X_cen = x - u
    X1_cen = X1 - u1
    X2_cen = X2 - u2
    X3_cen = X3 - u3

    ## 全局散度矩阵
    st = X_cen.T.dot(X_cen)

    ## cal 类内散度矩阵（within-class scatter matrix）
    sw = X1_cen.T.dot(X1_cen) + X2_cen.T.dot(X2_cen) + X3_cen.T.dot(X3_cen)

    ## cal 类间散度矩阵（between-class scatter matrix）
    # sb = len(X1) * (u1-u).T.reshape(len(u), 1).dot((u1-u).reshape(1, len(u))) + len(X2) * (u2-u).T.reshape(len(u), 1).dot((u2-u).reshape(1, len(u))) + len(X3) * (u3-u).T.reshape(len(u), 1).dot((u3-u).reshape(1, len(u)))
    sb = st - sw

    p = np.linalg.inv(sw).dot(sb)
    eigen_vals, eigen_vecs = np.linalg.eig(p)
    sorted = np.argsort(eigen_vals)

    w = eigen_vecs[:, sorted[::-1][:nComponents]]
    # w = eigen_vecs[:, list(reversed(sorted))[:nComponents]]
    newX = x.dot(w)

    plt.scatter(newX[:,0],newX[:,1],c=y,marker='o')
    plt.show()


if '__main__' == __name__:
    # X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
    #                            n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    LDA(X, y, nComponents=2)



