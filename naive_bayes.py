import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def gaussian_distribution(x, u, std):
    return np.exp(-(x-u)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

def bayes(x_train, x_test, y_train, y_test):
    n_test = len(x_test)
    labels = list(set(y_train))
    x_train_split = []
    for label in labels:                                 # 根据label，将相同label的data放在一起
        x_tmp = x_train[np.where(y_train == label)]
        x_train_split.append(x_tmp)

    mean = [np.mean(p, axis=0)for p in x_train_split]   #计算各类下，各特征的均值
    std = [np.std(p, axis=0) for p in x_train_split]   #计算各类下，各特征的标准差
    right = 0

    for i in range(n_test):
        test = x_test[i]
        max_pro = -np.inf
        result = None
        for j in range(len(labels)):
            pro_list = gaussian_distribution(test, mean[j], std[j])   # 计算各特征的概率
            pro_log = sum(np.log(pro_list))                           # 为防止溢出，概率积转换为求log
            if pro_log > max_pro:                                     # 判断哪一类的概率最大
                max_pro = pro_log
                result = j
        if result == y_test[i]:
            right +=1
    print("test: %d, right: %d, accuracy: %.3f" %(n_test, right, right/n_test))

def sklearn_bayes(x_train, x_test, y_train, y_test):
    n_test = len(x_test)
    model = GaussianNB()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    right = sum(predict == y_test)
    print("test: %d, right: %d, accuracy: %.3f" %(n_test, right, right/n_test))


if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3)
    bayes(x_tr, x_te, y_tr, y_te)
    sklearn_bayes(x_tr, x_te, y_tr, y_te)
