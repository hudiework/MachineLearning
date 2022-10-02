# coding:UTF-8
from os import listdir

from matplotlib.font_manager import FontProperties
from numpy import *
import importlib
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    :param inX:是输入的测试样本，是一个[x, y]样式的
    :param dataSet:是训练样本集
    :param labels:是训练样本标签
    :param k: 选择距离最近的k个点 是top k最相近的
    :return:
    """
    dataSetSize = dataSet.shape[0]  # 获取数据集的行数 shape【1】获取列数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 把inX生成 （dataSize行,1列）的数组 并求与dataSize的差值
    sqDiffMat = diffMat ** 2  # 差值求平方
    sqDistances = sqDiffMat.sum(axis=1)  # 矩阵的每一行向量相加
    distances = sqDistances ** 0.5  # 向量和开方
    sortedDistIndicies = distances.argsort()  # argsort函数返回的是数组值从小到大的索引值 按照升序进行快速排序，返回的是原数组的下标

    # 存放最终的分类结果及相应的结果投票数
    classCount = {}
    # 投票过程，就是统计前k个最近的样本所属类别包含的样本个数
    for i in range(k):
        # index = sortedDistIndicies[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('A' or 'B')
        voteIlabel = labels[sortedDistIndicies[i]]
        # classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # classCount.iteritems()将classCount字典分解为元组列表，
    # operator.itemgetter(1)按照第二个元素的次序对元组进行排序，reverse=True是逆序，即按照从大到小的顺序排列
    return sortedClassCount[0][0]


def file2matrix(filename):
    """

    :param filename: 文件名
    :return:
            returnMat - 特征矩阵
            classLabelVector - 分类Label向量
    """
    fr = open(filename)  # 打开文件
    arrayOLines = fr.readlines()  # 读取所有文件
    numberOfLines = len(arrayOLines)  # 获取文件长度
    returnMat = zeros((numberOfLines, 3))  # 设定几行几类的0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        listFromLines = line.split('\t')
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        # 对于datingTestSet2.txt  最后的标签是已经经过处理的 标签已经改为了1, 2, 3
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLines[0: 3]
        if listFromLines[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLines[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLines[-1] == 'largeDoses':
            classLabelVector.append(3)
        # classLabelVector.append(int(listFromLines - 1))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.01
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                                     datingLabels[numTestVecs: m], 3)
        # print("分类器解析得：%d 正确为： %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("错误率 %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['一点也不喜欢', '一般般', '很感兴趣']
    percentTats = float(input( \
        "愿意花时间打游戏的时间占比是？"))
    ffMiles = float(input("每年飞行里程数"))
    iceCream = float(input("每周吃的冰淇淋的公升数"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你可能是获得哪种打分？", resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('trainingDigits/%s' % (fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 5)
        print("分类器执行的结果： %s ，正确答案是  %s " % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("错误的总数是 ： %d" % (errorCount))
    print("总的错误率是 %f" % (errorCount / float(mTest)))


# 执行截图


if __name__ == "__main__":
    group, labels = createDataSet()
    classify0([0, 0], group, labels, 3)
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    print(normDataSet)
    print(ranges)
    print(minVals)
    
    
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('TkAgg')  # 必须显式指明matplotlib的后端
    # fig = plt.figure()
    # 设置汉字格式
    fig, axs = plt.subplots(sharex=False, sharey=False, figsize=(13, 8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs.scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs_title_text = axs.set_title(u'Random Plot Picture')
    axs_xlabel_text = axs.set_xlabel(u'Miles')
    axs_ylabel_text = axs.set_ylabel(u'Play Video Game')
    plt.setp(axs_title_text, size=9, weight='bold', color='red')
    plt.setp(axs_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs_ylabel_text, size=7, weight='bold', color='black')

    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    plt.show()

# datingDataMat
# datingLabels[0:20]

# testVector = img2vector('testDigits/0_13.txt')
# handwritingClassTest()
# print(testVector[0, 0:31])
# print(testVector[0, 32:63])
# datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
# normDataSet, ranges, minVals = autoNorm(datingDataMat)
# datingClassTest()
# classifyPerson()

# matplotlib.use('TkAgg')  # 必须显式指明matplotlib的后端
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()
