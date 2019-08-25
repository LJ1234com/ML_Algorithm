import numpy as np
import matplotlib.pyplot as plt


def cal_dist_centroid(mat1, mat2):
    cen1 = np.mean(mat1, axis=0)
    cen2 = np.mean(mat2, axis=0)
    dist = np.sqrt(sum((cen1 - cen2) ** 2))
    return dist

def get_dist_mat(data, cal_dist_type):
    dist_mat = np.zeros((len(data), len(data)))


    for i in range(len(data)):
        for j in range(len(data)):
            if i <= j:
                dist = cal_dist_type(data[[i]], data[[j]])  #data[i] is 1D array; data[[i]] is 2D array
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
            else:
                continue
    return dist_mat

def find_min_dist(dist_mat, group_len):
    min_dist = np.inf
    I, J = 0, 0
    for p in range(group_len):
        for q in range(p+1, group_len):
            if dist_mat[p, q] < min_dist:
                min_dist = dist_mat[p, q]
                I, J = p, q
    return I, J



def AGNES(x, K=3, cal_dist_type=cal_dist_centroid):
    group = [[i] for i in range(len(x))]                # 用index表示每一个样本
    group_len = len(group)
    dist_mat = get_dist_mat(data, cal_dist_type)        # 计算两两样本间的距离

    while group_len > K:
    # for i in range(1):
        i, j = find_min_dist(dist_mat, group_len)       # 找到距离矩阵中的最小值
        group[i].extend(group[j])                       # 把大的索引放在小的索引中
        del group[j]                                    # 删除大的索引
        dist_mat = np.delete(dist_mat, j, axis=0)       # 在距离矩阵中，删除大的索引所在的行
        dist_mat = np.delete(dist_mat, j, axis=1)       # 删除大的索引，删除大的索引所在的列

        for k in range(len(group)):
            dist_mat[i, k] = cal_dist_type(x[group[i]], x[group[k]])  # 更新距离矩阵中的第i行和第i列
            dist_mat[k, i] = dist_mat[i, k]
        group_len -= 1
    return group

def plot(data, group):
    labels = np.zeros(len(data))
    for j in range(len(group)):
        labels[group[j]] = j
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()


if __name__ == '__main__':
    data = np.loadtxt('kmeans.txt', dtype=float, delimiter=',', usecols=[0, 1])
    group = AGNES(data, 3, cal_dist_centroid)
    plot(data, group)


