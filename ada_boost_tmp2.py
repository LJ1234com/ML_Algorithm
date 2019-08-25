from numpy import *

#获取数据集
def loadSimpData():
    dataMat=matrix([[1. ,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels


# 单层决策树的阈值过滤函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    # 对数据集每一列的各个特征进行阈值过滤
    retArray = ones((shape(dataMatrix)[0], 1))
    # 阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    # 将大于某一阈值的特征归类为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    # 将数据集和标签列表转为矩阵形式
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    # 最小错误率初始化为+∞
    minError = inf
    # 遍历每一列的特征值
    for i in range(n):
        # 找出列中特征值的最小值和最大值
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()
        # 求取步长大小或者说区间间隔
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历各个步长区间
        for j in range(-1, int(numSteps) + 1):
            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']:
                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal = (rangeMin + float(j) * stepSize)
                # 选定阈值后，调用阈值过滤函数分类预测
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化错误向量
                errArr = mat(ones((m, 1)))
                # 将错误向量中分类正确项置0
                errArr[predictedVals == labelMat] = 0
                # 计算"加权"的错误率
                weigthedError = D.T * errArr
                # 打印相关信息，可省略
                # print("split: dim %d, thresh %.2f,thresh inequal:\
                #    %s, the weighted error is %.3f",
                #    %(i,threshVal,inequal,weigthedError))
                # 如果当前错误率小于当前最小错误率，将当前错误率作为最小错误率
                # 存储相关信息
                if weigthedError < minError:
                    minError = weigthedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return bestStump, minError, bestClasEst




#adaBoost算法
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #弱分类器相关信息列表
    weakClassArr=[]
    #获取数据集行数
    m=shape(dataArr)[0]
    #初始化权重向量的每一项值相等
    D=mat(ones((m,1))/m)
    #累计估计值向量
    aggClassEst=mat(zeros((m,1)))
    #循环迭代次数
    for i in range(numIt):
        #根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #打印权重向量
        # print("D:",D.T)
        #求单层决策树的系数alpha
        alpha=float(0.5*log((1.0-error)/(max(error,1e-16))))
        #存储决策树的系数alpha到字典
        bestStump['alpha']=alpha
        #将该决策树存入列表
        weakClassArr.append(bestStump)
        #打印决策树的预测结果
        # print("classEst:",classEst.T)
        #预测正确为exp(-alpha),预测错误为exp(alpha)
        #即增大分类错误样本的权重，减少分类正确的数据点权重
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新权值向量
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #累加当前单层决策树的加权预测值
        aggClassEst+=alpha*classEst
        # print("aggClassEst",aggClassEst.T)
        #求出分类错的样本个数
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        #计算错误率
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        #错误率为0.0退出循环
        if errorRate==0.0:break
    #返回弱分类器的组合列表
    return weakClassArr, aggClassEst

#测试adaBoost，adaBoost分类函数
#@datToClass:测试数据点
#@classifierArr：构建好的最终分类器
def adaClassify(datToClass,classifierArr):
    #构建数据向量或矩阵
    dataMatrix=mat(datToClass)
    #获取矩阵行数
    m=shape(dataMatrix)[0]
    #初始化最终分类器
    aggClassEst=mat(zeros((m,1)))
    #遍历分类器列表中的每一个弱分类器
    for i in range(len(classifierArr[0])):
        #每一个弱分类器对测试数据进行预测分类
        classEst=stumpClassify(dataMatrix,classifierArr[0][i]['dim'],\
                                classifierArr[0][i]['thresh'],
                                classifierArr[0][i]['ineq'])
        #对各个分类器的预测结果进行加权累加
        aggClassEst+=classifierArr[0][i]['alpha']*classEst
        # print('aggClassEst',aggClassEst)
    #通过sign函数根据结果大于或小于0预测出+1或-1
    return sign(aggClassEst)


#自适应加载数据
def loadDataSet(filename):
    #创建数据集矩阵，标签向量
    dataMat=[];labelMat=[]
    #获取特征数目(包括最后一类标签)
    #readline():读取文件的一行
    #readlines:读取整个文件所有行
    numFeat=len(open(filename).readline().split('\t'))
    #打开文件
    fr=open(filename)
    #遍历文本每一行
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #数据矩阵
        dataMat.append(lineArr)
        #标签向量
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#训练和测试分类器
def classify():
    #利用训练集训练分类器
    datArr,labelArr=loadDataSet('horseColicTraining.txt')
    #得到训练好的分类器
    classifierArray=adaBoostTrainDS(datArr,labelArr,10)
    #利用测试集测试分类器的分类效果
    testArr,testLabelArr=loadDataSet('horseColicTest.txt')
    prediction=adaClassify(testArr,classifierArray)
    #输出错误率
    num=shape(mat(testLabelArr))[1]
    errArr=mat(ones((num,1)))
    error=errArr[prediction!=mat(testLabelArr).T].sum()
    print("the errorRate is: %.9f"  %(float(error)/float((num))))


classify()