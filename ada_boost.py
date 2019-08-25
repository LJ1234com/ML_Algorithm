import numpy as np
import matplotlib.pyplot as plt

def stumpClassify(X, id, thresh, inequal):
    ret = np.ones(len(X))
    if inequal == 'lt':
        ret[X[:, id] <= thresh] = -1
    else:
        ret[X[:, id] > thresh] = -1
    return ret


def buildStump(X, Y, D):
    m, n = X.shape
    steps = 10
    best_stump = {}                                              #最佳单层决策树信息
    best_estimate = np.zeros(m)                                  #最佳的分类结果
    min_error = np.inf
    for j in range(n):
        min = np.min(X[:, j])
        max = np.max(X[:, j])
        step_size = (max - min) / steps
        for k in range(-1, steps + 1):
            thresh = min + k* step_size
            for inequal in ['lt', 'gt']:                         # 两种赋值方式， 一种是小于阈值赋-1， 一种是大于阈值赋-1
                errors = np.ones(m)                              #初始化误差矩阵
                predict = stumpClassify(X, j, thresh, inequal)
                errors[predict == Y] = 0                         #分类正确的,赋值为0
                weight_error = D.dot(errors)
                if weight_error < min_error:
                    min_error = weight_error
                    best_estimate = predict.copy()
                    best_stump['id'] = j
                    best_stump['thresh'] = thresh
                    best_stump['inequal'] = inequal
    return best_stump, min_error, best_estimate


def adaBoostTrainDS(X, Y, max_iter=40):
    m, n = X.shape
    weak_classifiers = []
    D = np.ones(m) / m                                         #初始化样本权重
    agg_estimate = np.zeros(m)                                 #累计估计值向量
    err_rate = 1.0
    for i in range(max_iter):
        bestStump, error, classEst = buildStump(X, Y, D)       #构建单层决策树
        alpha = np.log((1.0 - error)/max(error, 1e-6))/2       #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha
        weak_classifiers.append(bestStump)                     #存储单层决策树
        expon = np.exp(-1 * alpha * Y * classEst)
        D = D * expon
        D = D / sum(D)
        agg_estimate += alpha * classEst                       #加法模型, 不同的若分类器给予不同的权重alpha
        agg_error = np.sign(agg_estimate) != Y
        err_rate = sum(agg_error) / m
        if err_rate == 0.0:
            break
    return weak_classifiers, np.sign(agg_estimate), err_rate


def ada_classify(x, classifiers):
    estimates = np.zeros(len(x))
    for classifier in classifiers:
        est = stumpClassify(x, classifier['id'], classifier['thresh'], classifier['inequal'])
        estimates += classifier['alpha'] * est
    return np.sign(estimates)


if __name__ == '__main__':
    train = np.loadtxt('horseColicTraining2.txt')
    # train = np.loadtxt('svm_kernel1.txt')
    x_tr = train[:, :-1]
    y_tr = train[:, -1]
    weak_classifier, agg_estimates, train_error = adaBoostTrainDS(x_tr, y_tr, 40)
    print('训练集的错误率:%.9f%%' %(train_error * 100))


    # predictions = ada_classify(x_tr, weak_classifier)
    # print('训练集的错误率:%.9f%%' % float(sum(predictions != y_tr) / len(x_tr) * 100))

    test = np.loadtxt('horseColicTest2.txt')
    x_te = test[:, :-1]
    y_te = test[:, -1]
    predictions = ada_classify(x_te, weak_classifier)
    print('测试集的错误率:%.9f%%' % float(sum(predictions != y_te) / len(x_te) * 100))
