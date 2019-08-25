import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def cal_dist(train, test):
    dist_list = []
    for i in range(len(test)):
        dists = []
        for j in range(len(train)):
            dist = np.sqrt(sum((test[i] - train[j]) ** 2))
            dists.append(dist)
        dist_list.append(dists)
    return dist_list

def KNN(x_train, x_test, y_train, y_test, K=3):
    n_test = len(x_test)
    labels = list(set(y_train))
    dist_list = cal_dist(x_train, x_test)                               # 计算各test样本到train样本的距离
    dist_sorted_index = [np.argsort(xx) for xx in dist_list]            # 找出按距离由小到大，距离的索引

    k_near_index = [tt[:K] for tt in dist_sorted_index]                 # 取出最近K个距离最小样本的索引
    k_near_label = [y_train[label.tolist()] for label in k_near_index]  # 获取最近K个样本的label

    count_near_label = []
    for near_label in k_near_label:                                     # 统计最近K个样本的label出现的次数
        counts = {}
        for label in labels:
            counts[label] = near_label.tolist().count(label)
        count_near_label.append(counts)

    count_sorted = [sorted(tt.items(), key=lambda p: p[1], reverse=True) for tt in count_near_label]  # 最近K个样本的label出现的次数降序排列
    results = np.array([tt[0][0] for tt in count_sorted])                                             # 找出出现次数对多样本对应的label

    right = sum(results == y_test)
    print("test: %d, right: %d, accuracy: %.3f" %(n_test, right, right/n_test))
    # print("test: {}, right: {}, accuracy: {:.4f}".format(n_test, right, right/n_test))
    # print("test: {}, right: {}, accuracy: {}/{}".format(n_test, right, right, n_test))

def sklearn_KNN(x_train, x_test, y_train, y_test, K=3):
    n_test = len(x_test)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    right = sum(predict == y_test)
    print("sklearn-- test: %d, right: %d, accuracy: %.3f" %(n_test, right, right/n_test))


if __name__ == '__main__':
    data = datasets.load_iris()
    x, y = data.data, data.target

    for i in range(20):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3)
        KNN(x_tr, x_te, y_tr, y_te, 3)
        sklearn_KNN(x_tr, x_te, y_tr, y_te, 3)




















