# encoding:utf-8
import numpy as np
import sys
import math

# 计算邻接矩阵
def getAllIniEdges(dataSet):
    """
    得到第i个点和第j个点间的距离，这些距离权值保存在邻接矩阵中
    :return:
    """
    adjacentMatrix = np.full(shape=(len(dataSet), len(dataSet)), fill_value=sys.maxsize, dtype=np.float32)
    for i in range(0, len(dataSet)):
        for j in range(i+1, len(dataSet)):
            distance = np.sqrt(np.sum(np.square(np.subtract(dataSet[i], dataSet[j]))))
            adjacentMatrix[i][j] = adjacentMatrix[j][i] = distance
    return adjacentMatrix


def getABOF_exact(p, dataSet, n, edgeMatrix):
    cos_block = []
    for i in range(0, n):
        for j in range(0, n):
            if i == p:
                break
            elif i != j and j != p:
                pi = dataSet[i] - dataSet[p]
                pj = dataSet[j] - dataSet[p]
                cos = (np.dot(pi, pj)) / np.square(edgeMatrix[p][i] * edgeMatrix[p][j])
                cos_block.append(cos)
    v = np.var(cos_block)
    return v

# def getABOF_appro():


# 传入boundary的下标
def checkOutlier(indexs, minPts, knnMatrix):
    indexs_new = []
    print(len(indexs))
    for i in indexs:
        knnDist = knnMatrix[i]
        num = 0 # 表示某个离群点KNN中的也被识别为离群点的数量，若该数量大于一半，则其K近邻都被认为是离群点，否则
        # knn_dist[i]
        for pt in knnDist:
            if pt[0] in indexs:
                num = num + 1
        if num >= minPts/2:
            for pt in knnDist:
                if pt[0] != i:
                    indexs_new.append(pt[0])
        elif num <= minPts/2:
            indexs.remove(i)
    return list(set(indexs) | set(indexs_new))

# 冒泡排序
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




# 传入boundary的下标
def checkOutlier2(indexs, minPts, knnMatrix, dataSet):
    all = [i for i in range(len(dataSet))]
    notOutlier = list(set(all)-set(indexs))
    for i in notOutlier:
        knnDist = knnMatrix[i]
        num = 0 # 表示某个离群点KNN中的也被识别为离群点的数量，若该数量大于一半，则其K近邻都被认为是离群点，否则
        # knn_dist[i]
        for pt in knnDist:
            if pt[0] in indexs:
                num = num + 1
        if num > 0.7*minPts:
            indexs.append(i)
    return indexs