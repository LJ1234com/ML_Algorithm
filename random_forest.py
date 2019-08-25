import numpy as np
from random import seed
import time

def evaluate_algorithm(data, algorithm, n_folds, *args):
    scores = []
    fold_size = int(len(data) / n_folds)
    folds = [data[k:k+fold_size] for k in range(0, len(data), fold_size)]
    for i in range(n_folds):
        test = folds[i]
        folds_bk = folds.copy()
        del folds_bk[i]

        train = folds_bk[0]
        for ff in folds_bk[1:]:
            train = np.r_[train, ff]

        actual = test[:, -1]  # 真实的分类值列表
        test = test[:, :-1]

        predicted = algorithm(train, test, *args)  # 调用随机森林算法，预测的分类值列表
        accu = sum(actual == predicted) / len(actual) * 100
        scores.append(accu)
    return scores

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = []
    for i in range(n_trees):
        ## train 有120个样本，在这120个中有放回抽120个，随机采样保证了每棵决策树训练集的差异性
        sample = train[np.random.randint(0, len(train), len(train))]
        sample1 =[row.tolist() for row in sample]    # list processing is faster than array processing
        tree = build_tree(sample1, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


def build_tree(sample, max_depth, min_size, n_features):
    # 找到最佳切分向量和最佳切分点进行划分为两个子集
    root = get_split(sample, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def get_split(sample, n_features):
    # sample = [row.tolist() for row in sample]
    labels = list(set([row[-1] for row in sample]))

    b_index, b_value, b_score, b_groups = np.inf, np.inf, np.inf, None
    features = []
    while len(features) < n_features:   # 随机抽取两个不同的特征
        index = np.random.randint(len(sample[0])-1)
        if index not in features:
            features.append(index)

    for feature_id in features:   # 遍历随机抽取的feature， 找到最小的gini系数
        for x in sample:
            # 根据cut point（x[fearure_id]）把数据分成两部分， 然后算gini系数
            groups = test_split(feature_id, x[feature_id], sample)
            gini = cal_gini(groups, labels)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = feature_id, x[feature_id], gini, groups
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
    left, right = node['groups']
    del (node['groups'])

    if not left or not right:
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
    # print(data)

    n_folds = 5  # 把数据分成5份
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = int(np.sqrt(data.shape[1] - 1))

    for n_trees in [10, 10 , 10, 10, 10]: # 每次种一棵树
        np.random.shuffle(data)
        scores = evaluate_algorithm(data, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('/Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
