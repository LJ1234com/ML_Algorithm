import numpy as np
from sklearn import model_selection
import sklearn.datasets as datasets


def createData(file, testPercent):
        data1 = datasets.load_iris()
        data = data1.data
        label = data1.target
        # print(data)
        # print(label)

        train, test, trainLabel, testLabel = model_selection.train_test_split(data, label, test_size=testPercent)
        # print(len(train))
        # print(len(trainLabel))
        return train, test, trainLabel, testLabel


def calDist(trainList, test):
        distList = []
        for i in range(len(trainList)):
                dist = np.sqrt(np.sum((trainList[i]-test) ** 2))
                distList.append(dist)
        return distList

def KNN(trainList, testList, trainLabel, testLabel, K):
        labelList = list(np.unique(trainLabel.tolist() + testLabel.tolist()))
        right = 0
        for m in range(len(testList)):
                maxCount = -np.inf
                predictLabel = ""
                distList = calDist(trainList, testList[m])
                distSort = np.argsort(distList)  # 由小到大排列，返回对应的索引值

                #下面这段可代替下面注释部分
                k_nearest = distSort[:K]
                for label in labelList:
                        label_count = [trainLabel[index] for index in k_nearest].count(label)
                        if label_count > maxCount:
                                maxCount = label_count
                                predictLabel = label
                '''
                countDict = {x: 0 for x in labelList}
                for i in range(K):                  #计算每一类出现的次数
                        for label in labelList:
                                if trainLabel[distSort[i]] == label:
                                        countDict[label] += 1
                # print(countDict)
                for label in countDict:      #找出次数出现最多的一类
                        if countDict[label] > maxCount:
                                maxCount = countDict[label]
                                predictLabel = label
                '''

                if predictLabel == testLabel[m]:
                        right += 1
        print("total: %d, right: %d, accuracy: %f" %(len(testList), right, right/len(testList)))

if __name__ == "__main__":
        for i in range(20):   #执行20次，看准确率有多少
                train, test, trainLabel, testLabel =createData("iris.txt", 0.3)
                # print(train)
                # print(trainLabel)
                KNN(train, test, trainLabel, testLabel, 3)
