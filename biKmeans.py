import numpy as np
import matplotlib.pyplot as plt

def cal_dist(x, cen):
    return np.sqrt(np.sum((x - cen) ** 2))

def cal_dist2(x, cens):
    dists = []
    for i in range(len(x)):
        tmp = []
        for cen in cens:
            dist = cal_dist(x[i], cen)
            tmp.append(dist)
        dists.append(tmp)
    return dists

def init_centroids(x, k):
    random_index = np.random.randint(0, len(x), k)
    init_centroids = x[random_index].tolist()
    return init_centroids

def get_group(dists):
    dist_sorted = np.argsort(np.array(dists))
    new_dist = np.array(dists)[:, dist_sorted[:, 0]]
    return dist_sorted[:, 0], new_dist

def cal_centroids(x, group):
    labels = list(set(group))
    new_centroids = []
    for label in labels:
        new_centroids.append(np.mean(x[np.nonzero(group==label)], axis=0).tolist())
    return  new_centroids

def kMeans(x, groups=2):
    bi_cens = init_centroids(x, groups)
    changed = True
    group = []
    bi_dists = np.zeros(x.shape)
    while changed:
        dists = cal_dist2(x, bi_cens)
        group, new_dist = get_group(dists)
        new_centroids = cal_centroids(x, group)
        if new_centroids == bi_cens:
            changed = False
            bi_dists[:, 0], bi_dists[:, 1]= group, new_dist[0]
        else:
            bi_cens = new_centroids
    return bi_cens, bi_dists

def biKmeans(data, K=None):
    m, n = data.shape
    cluster_list = np.zeros((m, n))             # 第一列为类别， 第二列为距离
    centroid0 = np.mean(data, axis=0)
    centroids = [centroid0]
    for i in range(m):
        cluster_list[i, 1] = cal_dist(data[i], centroid0)

    while len(centroids) < K:
        min_sse = np.inf
        best_cluster = None
        best_cen = None
        best_split = None
        for i in range(len(centroids)):
            data2 = data[np.nonzero(cluster_list[:, 0]==i)[0]]   # 从data中取出类别为i的数据
            bi_cen, bi_cluster = kMeans(data2, 2)
            bi_sse = np.sum(bi_cluster[:, 1])
            notsplit_sse = np.sum(cluster_list[np.nonzero(cluster_list[:, 0]!=i)[0], 1])
            if bi_sse + notsplit_sse < min_sse:
                min_sse = bi_sse + notsplit_sse
                best_split = i
                best_cen = bi_cen
                best_cluster = bi_cluster

        best_cluster[np.nonzero(best_cluster[:, 0]==1)[0], 0] = len(centroids)
        best_cluster[np.nonzero(best_cluster[:, 0]==0)[0], 0] = best_split

        cluster_list[np.nonzero(cluster_list[:, 0]==best_split)[0], :] = best_cluster
        centroids[best_split] = best_cen[0]
        centroids.append(best_cen[1])
    return centroids,cluster_list


def plot(x, cens, dists):
    plt.scatter(x[:, 0], x[:, 1], c=final_dist[:, 0])
    plt.scatter(np.array(cens)[:, 0], np.array(cens)[:, 1], c='r')
    plt.show()


if __name__ == '__main__':
    data = np.loadtxt('kmeans.txt', dtype=float, delimiter=',', usecols=[0, 1])
    final_cens, final_dist = biKmeans(data, 3)
    plot(data, final_cens, final_dist)
