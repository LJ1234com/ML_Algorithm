import numpy as np
import matplotlib.pyplot as plt

def init_centroids(data, k):
    random_index = np.random.randint(0, len(data), k)
    return data[random_index].tolist()

def cal_dist(data, centroids):
    dist_list = []
    for i in range(len(data)):
        dists = []
        for centroid in centroids:
            dist = np.sqrt(sum((data[i] - centroid) ** 2))
            dists.append(dist)
        dist_list.append(dists)
    return dist_list

def cal_group(m, dist_list):
    group = [0 for i in range(m)]
    for i in range(m):
        group[i] = np.argsort(dist_list[i])[0]
    return group

def cal_centroids(data, group):
    new_centroids = []
    groups = list(set(group))
    for i in range(len(groups)):
        group_index = np.nonzero(group == groups[i])
        centroid = np.mean(data[group_index], axis=0).tolist()
        new_centroids.append(centroid)
    return new_centroids



def kMeans(data, K=3):
    m = len(data)
    centroids = init_centroids(data, K)
    changed = True
    group = []
    while changed:
        dist_list = cal_dist(data, centroids)       # 计算每个样本到各中心的距离
        group = cal_group(m, dist_list)             # 根据距离判断样本属于哪一类
        new_centroids = cal_centroids(data, group)  # 根据新类， 更新中心
        if new_centroids == centroids:              # 判断质心有无变化，有变化则继续循环
            changed = False
        else:
            centroids = new_centroids

        plt.clf()
        plt.scatter(data[:, 0], data[:, 1], c=group)
        plt.pause(0.5)
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt('kmeans.txt', dtype=float, delimiter=',', usecols=[0, 1])
    kMeans(data, K=3)
