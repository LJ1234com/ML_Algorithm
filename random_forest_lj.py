import numpy as np


def choose_features(n, n_features):
    ret = []
    while len(ret) < n_features:
        index = np.random.randint(n)
        if index not in ret:
            ret.append(index)
    return ret


def cal_gini(groups, labels):
    gini = 0
    for label in labels:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            else:
                pro = [r[-1] for r in group].count(label) / size
                gini += pro * (1 - pro)
    return gini


def bi_split(train, feature_id, value):
    left, right = [], []
    for xx in train:
        if xx[feature_id] < value:
            left.append(xx)
        else:
            right.append(xx)
    return left, right


def get_split(train, n_features):
    n = len(train[0]) - 1  # the last column is label
    opt_id, opt_value, min_gini, opt_groups = np.inf, np.inf, np.inf, None
    labels = list(set(data[:, -1]))
    features = choose_features(n, n_features)
    for x in train:
        for feature_id in features:
            groups = bi_split(train, feature_id, x[feature_id])
            gini = cal_gini(groups, labels)
            if gini < min_gini:
                opt_id, opt_value, min_gini, opt_groups = feature_id, x[feature_id], gini, groups
    return {'index': opt_id, 'value': opt_value, 'groups': opt_groups}


def split(node, n_features, max_depth, min_size, depth):
    left, right = node['groups']
    del node['groups']

    if not left or not right:
        node['left'] = node['right'] = get_label(left + right)
        return
    if depth > max_depth:
        node['left'], node['right'] = get_label(left), get_label(right)
        return
    if len(left) < min_size:
        node['left'] = get_label(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], n_features, max_depth, min_size, depth+1)
    if len(right) < min_size:
        node['right'] = get_label(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], n_features, max_depth, min_size, depth+1)


def get_label(group):
    outcomes = [row[-1] for row in group]
    return max(outcomes, key=outcomes.count)


def build_tree(train, n_features, max_depth, min_size):
    root = get_split(train, n_features)
    split(root, n_features, max_depth, min_size, 1)
    return root


def random_forest(data, n_trees, max_depth, min_size):
    m = data.shape[0]
    accu_list = []
    predict_list = []
    for i in range(n_trees):
        train_index = np.random.randint(0, m, m)
        oob_index = np.array([i for i in range(m) if i not in train_index])
        train = [x.tolist() for x in data[train_index]]  # list is faster than array
        test = data[oob_index]

        tree = build_tree(train, n_features, max_depth, min_size)
        predictions = bagging_predict([tree], test)
        accu_list.append(sum(predictions==test[:, -1])/len(test))
        predict_list.append(dict(zip(oob_index, predictions)))

    print(accu_list)
    print(np.mean(accu_list))

    results = []
    is_oob = []
    for i in range(m):
        predict = []
        for p in predict_list:
            if i in p.keys():
                predict.append(p[i])

        if len(predict) > 0:
            results.append(max(predict, key=predict.count))
            is_oob.append(i)
    print(len(results), len(is_oob))
    print(sum(results == data[is_oob][:, -1])/len(results))


def bagging_predict(trees, test):
    predictions = []
    for x in test:
        tmp = [predict(tree, x) for tree in trees]
        predictions.append(max(tmp, key=tmp.count))
    return predictions


def predict(node, x):
    if x[node['index']] < node['value']:
        if isinstance(node['left'], dict):  # 判断node['left']是不是dict类型
            return predict(node['left'], x)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], x)
        else:
            return node['right']


if __name__ =='__main__':
    data = np.loadtxt('iris.txt', delimiter=',', dtype=float)

    max_depth = 10
    min_size = 1
    n_features = int(np.sqrt(data.shape[1] - 1))
    for n_trees in [10]:
        random_forest(data, n_trees, max_depth, min_size)