import numpy as np
from random import seed
import time
from sklearn.ensemble import RandomForestClassifier



def random_forest(data, max_depth, min_size, n_trees, n_features):
    m = len(data)
    trees = []
    accus = []
    predictions_list = []
    for i in range(n_trees):
        index_train = np.random.randint(0, m, m)      # 从m个样本中有放回抽样，抽取m个样本
        sample = data[index_train]
        sample1 =[row.tolist() for row in sample]    # list processing is faster than array processing
        tree = build_tree(sample1, max_depth, min_size, n_features)

        index_oob = [ii for ii in range(m) if ii not in index_train]
        test = data[index_oob]              # 用oob样本作为测试集

        predictions = [bagging_predict([tree], row) for row in test]
        predictions_list.append(dict(zip(index_oob, predictions)))
        actual = test[:, -1]
        accu = sum(predictions == actual) / len(test)
        accus.append(accu)
    print(accus)
    print(np.mean(accus))


    results = []
    is_oob = []
    for i in range(m):
        predict = []
        for p in predictions_list:
            if i in p.keys():       # 如果第i个样本是第p棵树的oob， 则获取其预测值
                predict.append(p[i])

        if len(predict) > 0:          # 大于零是由于第i个样本可能每次都被抽中作为train样本
            results.append(max(predict, key=predict.count))  # 多数投票原则， 判断该样本作为oob样本的预测值
            is_oob.append(i)

    print(len(results))
    print(len(is_oob))
    print(sum(results == data[is_oob][:, -1])/ len(is_oob))



def build_tree(sample, max_depth, min_size, n_features):
    root = get_split(sample, n_features)                      # 找到最佳切分向量和最佳切分点进行划分为两个子集
    split(root, max_depth, min_size, n_features, 1)
    return root


def get_split(sample, n_features):
    # sample = [row.tolist() for row in sample]
    labels = list(set([row[-1] for row in sample]))

    b_index, b_value, min_gini, b_groups = np.inf, np.inf, np.inf, None
    features = []
    while len(features) < n_features:                        # 随机抽取两个不同的特征
        index = np.random.randint(len(sample[0])-1)
        if index not in features:
            features.append(index)

    for feature_id in features:                              # 遍历随机抽取的feature， 找到最小的gini系数
        for x in sample:
            # 根据cut point（x[fearure_id]）把数据分成两部分， 然后算gini系数
            groups = test_split(feature_id, x[feature_id], sample)
            gini = cal_gini(groups, labels)
            if gini < min_gini:
                b_index, b_value, min_gini, b_groups = feature_id, x[feature_id], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def test_split(feature_id, value, sample):
    left, right = [], []
    for xx in sample:
        if xx[feature_id] < value:
            left.append(xx)
        else:
            right.append(xx)
    return left, right

def cal_gini(groups, labels):
    gini = 0
    for label in labels:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(label) / float(size)
            gini += proportion * (1 - proportion)
    return gini


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']    # 左右两部分数据取出来
    del (node['groups'])

    if not left or not right:     # 左右有一个为空，则用相同类别表示
        # print(node)
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(outcomes, key=outcomes.count)

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(predictions, key=predictions.count)

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']




if __name__ == "__main__":
    seed(1)
    data = np.loadtxt('iris.txt', delimiter=',', dtype=float, usecols=[0, 1, 2, 3, 4])

    max_depth = 10     # 树的最大深度
    min_size = 10      #
    n_features = int(np.sqrt(data.shape[1] - 1))


    for n_trees in [5]: # 每次种5棵树
        np.random.shuffle(data)
        random_forest(data, max_depth, min_size, n_trees, n_features)