# encoding:utf-8
import numpy as np
from matplotlib import pyplot as plt


# 加载数据
def dataLoad(dataPath):
    data = []
    with open(dataPath, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            d = line.strip(' ').split(' ')
            data.append(np.float64(d))
    file.close()
    return np.array(data)


# 拆分数据, 拆成4部分
def splitData(dataSet):
    dataBlock = [[] for i in range(4)]
    x_span = [np.min(dataSet[:, 0]), np.max(dataSet[:, 0])]
    y_span = [np.min(dataSet[:, 1]), np.max(dataSet[:, 1])]
    index = [[] for i in range(4)]
    for k in range(0, len(dataSet)):
        t = dataSet[k]
        if dataSet[k][0] <= (x_span[0]+x_span[1])/2:
            if dataSet[k][1] > (y_span[0]+y_span[1])/2:
                dataBlock[0].append(dataSet[k])
                index[0].append(k)
            elif dataSet[k][1] <= (y_span[0] + y_span[1]) / 2:
                dataBlock[1].append(dataSet[k])
                index[1].append(k)
        elif dataSet[k][0] > (x_span[0]+x_span[1])/2:
            if dataSet[k][1] > (y_span[0]+y_span[1])/2:
                dataBlock[2].append(dataSet[k])
                index[2].append(k)
            elif dataSet[k][1] <= (y_span[0] + y_span[1])/2:
                dataBlock[3].append(dataSet[k])
                index[3].append(k)
    return dataBlock,x_span, y_span


# 拆分数据, 拆成2部分
def splitData2(dataSet):
    dataBlock = [[] for i in range(2)]
    x_span = [np.min(dataSet[:, 0]), np.max(dataSet[:, 0])]
    y_span = [np.min(dataSet[:, 1]), np.max(dataSet[:, 1])]
    index = [[] for i in range(2)]
    for k in range(0, len(dataSet)):
        if dataSet[k][0] <= (x_span[0]+x_span[1])/2:
            dataBlock[0].append(dataSet[k])
            index[0].append(k)
        elif dataSet[k][0] > (x_span[0]+x_span[1])/2:
            dataBlock[1].append(dataSet[k])
            index[1].append(k)
    return dataBlock,x_span, y_span



def drawOutlier(data, outlier, of, path):
    """
    画出离群点及其离群因子
    :param data: 源数据
    :param outlier: 源数据
    :return:
    """
    if len(outlier):
        plt.subplot(1, 2, 1)
        plt.scatter(data[:, 0], data[:, 1], s=5, c='g', marker='*')
        plt.scatter(outlier[:, 0], outlier[:, 1], s=5, c='r', marker='*')
    plt.subplot(1, 2, 2)
    x = range(0, len(data))
    plt.plot(x, of[:, 1], c='b', label='LOF')
    plt.legend(loc='best')
    plt.savefig(path)
    plt.show()

# 定义阈值
def findThreshold(of_block, f):
    mean = np.mean(of_block[:, 1])
    std = np.std(of_block[:, 1])
    threshold = mean + f*std
    return threshold