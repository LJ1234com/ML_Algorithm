import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_pts_list(x, r):
    pts_list = []
    for i in range(len(x)):
        pts = []
        for j in range(len(x)):
            dist = np.sqrt(sum((x[i] - x[j]) ** 2))
            if i !=j and dist <= r:
                pts.append(j)
        pts_list.append(pts)
    return pts_list

def DBSCAN(data, r=3.0, minpts=1):
    m = len(data)
    pts_list = get_pts_list(data, r)                                # 找出每个点r半径内的所有点

    core_list = [i for i in range(m) if len(pts_list[i])>= minpts]  # 周围大于minpts个点，即为核心点
    border_list = []
    for i in range(m):                                              # 本身不是核心点，但其周围有核心点
        if i not in core_list:
            for j in pts_list[i]:
                if j in core_list:
                    border_list.append(i)
                    break

    noise_list = [i for i in range(m) if i not in core_list and i not in border_list]    #噪声点既不是核心点，也不是border点

    group = [i for i in range(m)]
    for i in range(m):
        for j in pts_list[i]:
            if i in core_list and j in core_list and i < j:   # 密度可达的核心点 分为同一类，且用索引小的表示
                for k in range(m):
                    if group[k] == group[j]:
                        group[k] = group[i]

    for i in range(m):                                    # border点划到其周围的核心点所在的类
        for j in pts_list[i]:
            if i in border_list and j in core_list:
                group[i] = group[j]
    return group, noise_list

def plot(data, group, noise):
    index_without_noise = [i for i in range(len(group)) if group[i] not in noise]
    data2 = data[index_without_noise]
    data_noise = data[noise]

    group_without_noise = [i for i in group if i not in noise]
    for i in range(len(group_without_noise)):                    # to plot data with right color
        if group_without_noise[i] == 103:
            group_without_noise[i] = 1

    cmap_bold = ListedColormap(['b', 'g', 'r'])
    plt.scatter(data2[:, 0], data2[:, 1], c=group_without_noise, cmap=cmap_bold)
    plt.scatter(data_noise[:, 0], data_noise[:, 1], c='black')
    plt.show()


if __name__ == '__main__':
    data = np.loadtxt('kmeans.txt', dtype=float, delimiter=',', usecols=[0, 1])
    group, noise = DBSCAN(data, r=0.8, minpts=10)
    plot(data, group, noise)