import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    alldata = np.loadtxt(fileName)
    dataMat = alldata[:, 0:-1]  # 添加数据
    labelMat = alldata[:, -1]  # .astype(int).reshape(-1,1)  #添加标签
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []          #正样本
    data_minus = []         #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)      #转换为numpy矩阵
    data_minus_np = np.array(data_minus)    #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0],1)) #初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1  #如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1   #如果大于阈值,则赋值为-1
    return retArray




def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf') #最小误差初始化为正无穷大
    for i in range(n):  #遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max() #找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps #计算步长
        for j in range(-1, int(numSteps) + 1): #对每个步长

            for inequal in ['lt', 'gt']:# 有两种赋值法，一种是将小于阈值的样本赋-1， 一种是将大于阈值的样本赋-1
                threshVal = (rangeMin + float(j) * stepSize) #计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
                errArr = np.mat(np.ones((m,1))) #初始化误差矩阵为全一
                errArr[predictedVals == labelMat] = 0   #分类正确的,赋值为0
                weightedError = D.T * errArr     #计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:    #找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst





def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.ones((m, 1)) / m #初始化权重
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  #构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  #计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha                                          #存储弱学习算法权重
        weakClassArr.append(bestStump)                                      #存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)     #计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()                                                        #根据样本权重公式，更新样本权重
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.sign(aggClassEst) != np.mat(classLabels)     #计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break                                             #误差为0，退出循环
    return weakClassArr, aggClassEst




def adaClassify(datToClass,classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):                                        #遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet(r'horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 40)
    testArr, testLabelArr = loadDataSet(r'horseColicTest2.txt')

    predictions = adaClassify(dataArr, weakClassArr)

    errArr = np.mat(np.ones((len(dataArr), 1)))

    print('训练集的错误率:%.9f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))



    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.9f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))



