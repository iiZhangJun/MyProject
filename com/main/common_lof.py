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

def drawOutlier(data, outlier):
    """
    画出离群点及其离群因子
    :param data: 源数据
    :param outlier: 源数据
    """
    if len(outlier):
        plt.scatter(outlier[:, 0], outlier[:, 1], s=2, c='r', marker='*')
