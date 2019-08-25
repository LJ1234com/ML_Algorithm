# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
import time

def load_csv(filename):
    dataset = list()
    data = open(filename, "r")
    for line in data.readlines():
        line = line.strip("\n").split(",")
        dataset.append(line)
    return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) / n_folds

    for i in range(n_folds):
        fold = list()
        while len(fold) <= fold_size - 1:
            # print(len(dataset_copy))
            # index = randrange(len(dataset_copy))
            index = np.random.randint(len(dataset_copy))

            # print(index)
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataset
def get_split(dataset, n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:  ##产生n_features个随机数
        # index = randrange(len(dataset[0]) - 1)
        index = np.random.randint(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:  ##随机抽取两个特征
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # 把样本分成 left/right两部分
            gini = gini_index(groups, class_values)
            if gini < b_score:  ##找出min gini
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):  # 输出group中出现次数较多的标签
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)  # 取set(outcomes)中的元素在outcomes中出现次数最大值


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


# Make a prediction with a decision tree
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


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):  # dataset是train data
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)

        tree = build_tree(sample, max_depth, min_size, n_features)

        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    # for i in folds:
    # print(len(i))
    scores = list()
    # 每次循环从folds从取出一个fold作为测试集，其余作为训练集，遍历整个folds，实现交叉验证
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
        test_set = list()
        for row in fold:  # fold表示从原始数据集dataset提取出来的测试集
            row_copy = list(row)
            test_set.append(row_copy[:-1])
            # row_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)  # 调用随机森林算法，预测的分类值列表

        actual = [row[-1] for row in fold]  # 真实的分类值列表
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def build_tree(train, max_depth, min_size, n_features):
    # 找到最佳切分向量和最佳切分点进行划分为两个子集
    t1 = time.time()
    root = get_split(train, n_features)
    print(time.time() - t1)
    split(root, max_depth, min_size, n_features, 1)

    return root


# Test the random forest algorithm
if __name__ == "__main__":
    seed(1)
    filename = 'iris.txt'
    dataset = load_csv(filename)

    # evaluate algorithm
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_features = int(sqrt(len(dataset[0]) - 1))

    for n_trees in [10, 10, 10, 10, 10]:
        np.random.shuffle(dataset)
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees,
                                    n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))