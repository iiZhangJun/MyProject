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
计算某点P的局部可达密度
'''
def reachDensity(pIndex, knnMatrix, minPts):
    flag = 1
    sumReachDist = 0
    while flag < minPts:
        index_o = knnMatrix[0][pIndex][flag]
        dist_po = knnMatrix[1][pIndex][flag]
        minPts_dist_o = knnMatrix[1][index_o][-1]
        reach_Dist_p = max(dist_po, minPts_dist_o)
        sumReachDist = sumReachDist + reach_Dist_p
        flag = flag+1
    lrd = (minPts-1)/sumReachDist
    return lrd

'''
计算某点的离群因子LOF
'''
def computeLOF(pIndex,knnMatrix, minPts):
    lrd_p = reachDensity(pIndex, knnMatrix, minPts)
    flag = 1
    lrd_op_sum = 0
    while flag < minPts:
        index_o = knnMatrix[0][pIndex][flag]
        lrd_o = reachDensity(index_o, knnMatrix, minPts)
        lrd_op_sum = lrd_op_sum + lrd_o/lrd_p
        flag = flag+1
    lof_p = lrd_op_sum/(minPts-1)
    return lof_p

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
    indexs_new = []
    for i in indexs:
        num = 0 # 表示某个离群点KNN中的也被识别为离群点的数量，若该数量大于一半，则其K近邻都被认为是离群点，否则
        # knn_dist[i]
        for pt in knnMatrix[0][i]:
            if pt in indexs and pt != i:
                num = num + 1
        if num > minPts/2:
            for pt in knnMatrix[0][i]:
                if pt != i:
                    indexs_new.append(pt)
        elif num <= minPts/5:
            indexs.remove(i)
    return list(set(indexs) | set(indexs_new))

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
dataSet = dataLoad('../../data/t5_5614.dat')
outliers_fraction = 0.25  #异常样本比例
'''
1.拆分
'''
db = DBSCAN(eps=20, min_samples=5).fit(dataSet)
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
    MinPtsLB = 7
    MinPtsUB = 45
    lof_block = [[i, 0] for i in range(0, len(block))]
    if len(block) < MinPtsUB:
        plt.scatter(block[:, 0], block[:, 1], s=5, c='r', marker='*')
    else:
        knnMatrix = computeKnn(MinPtsUB, block)
        for minPts in range(MinPtsLB, MinPtsUB):
            for i in range(0, len(block)):
                lof_i = computeLOF(i, knnMatrix, minPts)
                if lof_i > lof_block[i][1]:
                    lof_block[i][1] = lof_i
        lof_block = np.array(lof_block)
        threshold = findThreshold(lof_block, -0.15)
        print('---- lof阈值为：' + str(threshold) + '--------')
        indexs = [np.int32(lof_block[i][0]) for i in np.int32(lof_block[:, 0]) if lof_block[i][1] > threshold]
        # indexs = checkOutlier(indexs, 10, knnMatrix)
        outliers = np.array([block[index] for index in indexs])
        if len(outliers) > 0:
            drawOutlier(block, outliers)
        else:
            print('此数据集不存在离群点')
plt.show()