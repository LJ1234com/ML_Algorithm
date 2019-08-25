import numpy as np

def get_cut_pts(x):
    m = len(x)
    cut_pts = []
    for i in range(m - 1):
        cut_pts.append((x[i] + x[i+1])/2)
    return cut_pts


def find_optimized_pt(x, y_new, cut_pts):
    min_loss = np.inf
    c1 = 0
    c2 = 0
    opt_pt = 0
    y_new_tmp = np.zeros_like(y_new)
    for cut in cut_pts:   # 根据分割点， 将数据分成r1， r2两部分
        r1 = [index for index, value in enumerate(x) if x[index] < cut]
        r2 = [index for index, value in enumerate(x) if x[index] > cut]
        y1 = np.mean(y_new[r1])     # r1 部分的值用平均值表示
        y2 = np.mean(y_new[r2])     # r1 部分的值用平均值表示
        yy1 = y_new[r1] - y1        # 新的y值用原来的y值减去均值表示
        yy2 = y_new[r2] - y2
        loss = sum(yy1**2) + sum(yy2**2)

        if loss < min_loss:
            min_loss = loss
            c1 = y1
            c2 = y2
            opt_pt = cut
            y_new_tmp = np.r_[yy1, yy2]
    return opt_pt, c1, c2, min_loss, y_new_tmp




def CART(x, y, cut_pts):
    y_new = y
    loss = np.inf
    treeList  = []
    while loss > 0.2:
        # 返回最佳分割点，R1/R2区域的值， loss，新的y值
        optimized_pt, y1, y2, loss, y_new = find_optimized_pt(x, y_new, cut_pts)
        treeList.append([optimized_pt, y1, y2, loss])
    print(treeList)

    final_cuts = list(set([l[0] for l in treeList]))
    final_cuts_bk = final_cuts.copy()
    final_cuts.insert(0, -np.inf)
    final_cuts.append(np.inf)
    print(final_cuts)

    values = []
    for i in range(len(final_cuts)-1):
        tmp= 0
        for tree in treeList:
            if tree[0] >= final_cuts[i+1]:  # cut at the right of the region
                tmp += tree[1]
            if tree[0] <= final_cuts[i]:  # cut at the left of the region
                tmp += tree[2]
        values.append(tmp)

    resuls = [final_cuts_bk, values]
    print(resuls)

    loss = 0
    final_cuts_bk.append(np.inf)
    for i in range(len(x)):
        for j in range(len(final_cuts_bk)):
            if x[i]< final_cuts_bk[j]:
                print(x[i], values[j])
                loss += (y[i] - values[j])**2
                break
    print(loss)          # 最后的loss与while循环中的loss相同


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
    cut_pts = get_cut_pts(X)   # 根据x，指定分割点
    print(cut_pts)
    CART(X, Y, cut_pts)
