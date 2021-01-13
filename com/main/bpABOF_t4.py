# encoding: utf-8
from sklearn.neighbors import NearestNeighbors, lof
from com.util.dataAbout import dataLoad
from com.util.thresholdAbout import findThreshold
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt

'''
计算knn
'''
def computeKnn(k, data):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    knnMatrix = [indices, distances]
    return knnMatrix

'''
计算ABOF
'''
def getABOF_appro(p, knnMatrix, dataSet, minPts):
    cos_block = []
    for pt_a in range(1, minPts):
        for pt_b in range(1, minPts):
            if pt_b != pt_a:
                pi = dataSet[knnMatrix[0][p][pt_a]] - dataSet[p]
                pj = dataSet[knnMatrix[0][p][pt_b]] - dataSet[p]
                cos = (np.dot(pi, pj)) / np.square(knnMatrix[1][p][pt_a] * knnMatrix[1][p][pt_b])
                cos_block.append(cos)
    v = np.var(cos_block)
    return v

'''
冒泡排序
'''
def sortBubble(data, flag):
    if flag == 'ascend':
        for i in range(0, len(data)):
            for j in range(0, len(data)-i-1):
                if data[j][1] > data[j+1][1]:
                    temp = data[j]
                    data[j] = data[j+1]
                    data[j+1] = temp
    elif flag == 'descend':
        for i in range(0, len(data)):
            for j in range(0, len(data)-i-1):
                if data[j][1] < data[j+1][1]:
                    temp = data[j]
                    data[j] = data[j+1]
                    data[j+1] = temp
    return data

'''
传入boundary的下标
'''
def checkOutlier(indexs, minPts, knnMatrix):
    rem = []
    for i in indexs:
        num = 0
        for pt in knnMatrix[0][i][:minPts]:
            if pt != i and pt in indexs:
                num = num + 1
        if num < minPts/5:
            rem.append(i)
    return list(set(indexs).difference(set(rem)))

'''
画图
'''
def drawOutlier(data, outlier):
    """
    画出离群点及其离群因子
    :param data: 源数据
    :param outlier: 源数据
    """
    if len(outlier):
        plt.scatter(data[:, 0], data[:, 1], s=5, c='g', marker='*')
        plt.scatter(outlier[:, 0], outlier[:, 1], s=5, c='r', marker='*')

'''
0.获取数据
'''
dataSet = dataLoad('../../data/t4_6142.dat')
outliers_fraction = 0.25  #异常样本比例
'''
1.拆分
'''
db = DBSCAN(eps=22, min_samples=5).fit(dataSet)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
gloable_outlier = dataSet[labels == -1]
if len(gloable_outlier) > 0:
    plt.scatter(gloable_outlier[:, 0], gloable_outlier[:, 1], s=5, c='r', marker='*')
mem = 650
'''

'''
for t in range(num_cluster):
    block = dataSet[labels == t]
    n = len(block)
    MinPtsLB = 5
    MinPtsUB = 6
    abof_block = [[i, 0] for i in range(0, len(block))]
    if len(block) < MinPtsUB:
        plt.scatter(block[:, 0], block[:, 1], s=5, c='r', marker='*')
    else:
        knnMatrix = computeKnn(MinPtsUB, block)
        for minPts in range(MinPtsLB, MinPtsUB):
            for i in range(0, len(block)):
                abof_i = getABOF_appro(i, knnMatrix, block, minPts)
                if abof_i < abof_block[i][1]:
                    abof_block[i][1] = abof_i
        abof_block = np.array(abof_block)
        # threshold = findThreshold(abof_block, -0.1)
        # print('---- lof阈值为：' + str(threshold) + '--------')
        #  indexs = [np.int32(abof_block[i][0]) for i in np.int32(abof_block[:, 0]) if abof_block[i][1] < threshold]
        # indexs = checkOutlier(indexs, 11, knnMatrix)
        thres = int(n * 0.25)
        indexs = [int(bk[0]) for bk in abof_block[0:thres]]
        outliers = np.array([block[index] for index in indexs])
        if len(outliers) > 0:
            drawOutlier(block, outliers)
            plt.show()
        else:
            print('此数据集不存在离群点')
plt.show()