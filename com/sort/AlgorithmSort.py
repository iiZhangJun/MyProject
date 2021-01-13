# encoding:utf-8
import numpy as np
np.set_printoptions(suppress=True)
import sys
import math
from matplotlib import pyplot as plt


# 加载数据
def dataLoad(dataPath):
    data = []
    with open(dataPath, mode='r', encoding='utf-8') as file:
        for line in file.readlines():
            d = line.strip(' ').split('\t')
            data.append(np.float64(d))
    file.close()
    return np.array(data)


# 计算邻接矩阵
def getAllIniEdges(dataSet):
    """
    得到第i个点和第j个点间的距离，这些距离权值保存在邻接矩阵中
    :return:
    """
    adjacentMatrix = np.full(shape=(len(dataSet), len(dataSet)), fill_value=sys.maxsize, dtype=np.float64)
    for i in range(0, len(dataSet)):
        for j in range(i+1, len(dataSet)):
            distance = np.sqrt(np.sum(np.square(np.subtract(dataSet[i], dataSet[j]))))
            adjacentMatrix[i][j] = adjacentMatrix[j][i] = distance
    return adjacentMatrix

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

# 计算每个点的MinPts-NN距离
def computeKnn(minPts, dataSet):
    dataSize = len(dataSet)
    knn_Matrix = []
    for i in range(0, dataSize):
        k = 0
        knn_distance = []
        for j in range(0, dataSize):
            if i != j:
                distance = np.sqrt(np.sum(np.square(dataSet[i] - dataSet[j])))
                if k < minPts:
                    knn_distance.append([j, distance])
                    k = k+1
                    if k == minPts:
                        sortBubble(knn_distance, 'ascend')
                else:
                    if distance < knn_distance[-1][1]:
                        knn_distance[-1] = [j, distance]
                        sortBubble(knn_distance, 'ascend')
        knn_Matrix.append(knn_distance)
    return knn_Matrix


# 计算某点P的局部可达密度
def reachDensity(pIndex, knnMatrix, minPts):
    flag = 0
    sumReachDist = 0
    while flag < minPts:
        o = knnMatrix[pIndex][flag]
        index_o = o[0]
        dist_po = o[1]
        minPts_dist_o = knnMatrix[index_o][-1][1]
        reach_Dist_p = max(dist_po, minPts_dist_o)
        sumReachDist = sumReachDist + reach_Dist_p
        flag = flag+1
    lrd = minPts/sumReachDist
    return lrd


# 计算某点的离群因子LOF
def computeLOF(pIndex,knnMatrix, minPts):
    lrd_p = reachDensity(pIndex, knnMatrix, minPts)
    flag = 0
    lrd_op_sum = 0
    while flag < minPts:
        o = knnMatrix[pIndex][flag]
        index_o = o[0]
        lrd_o = reachDensity(index_o, knnMatrix, minPts)
        lrd_op_sum = lrd_op_sum + lrd_o/lrd_p
        flag = flag+1
    lof_p = lrd_op_sum/minPts
    return lof_p


# 计算某点的基于角度的近似离群因子ABOF -- FastABOD
def computeABOF_k(pIndex, knnMatrix, data):

    knn_p = knnMatrix[pIndex]
    first_sum_u = 0
    sum_d = 0
    second_sum_u = 0
    flag = []
    for pt_a in knn_p:
        for pt_b in knn_p:
            if pt_b != pt_a:
                pa = data[pt_a[0]]-data[pIndex]
                pb = data[pt_b[0]]-data[pIndex]
                dot_prod = np.dot(pa, pb)
                sqr_prod = pt_a[1]*pt_b[1]
                square_prod = np.square(sqr_prod)
                common = (1 / sqr_prod) * (dot_prod / square_prod)
                second_sum_u += common
                first_sum_u += np.square(common)
                sum_d += (1/sqr_prod)
    abof_p = first_sum_u/sum_d - np.square(second_sum_u/sum_d)
    return abof_p


def getABOF_appro(p, knnMatrix, dataSet):
    cos_block = []
    knn_p = knnMatrix[p]
    for pt_a in knn_p:
        for pt_b in knn_p:
            if pt_b != pt_a:
                pi = dataSet[pt_a[0]] - dataSet[p]
                pj = dataSet[pt_b[0]] - dataSet[p]
                cos = (np.dot(pi, pj)) / np.square(pt_a[1] * pt_b[1])
                cos_block.append(cos)
    v = np.var(cos_block)
    return v












# 找阈值点
def findMaxGapIndex(xOFBlock, flag):
    """
    找到变化最剧烈的位置处作为分界
    :param xOFBlock:
    :param flag:
    :return:
    """
    gap_index = -1
    if len(xOFBlock):
        # 降序 适合LOF 或 DB-outlier
        if flag == 'descend':
            gap = [xOFBlock[i-1][1] - xOFBlock[i][1] for i in range(1, len(xOFBlock))]
            gap_index = np.where(gap == max(gap))[-1][-1]
        # 升序 适合ABOF-outlier
        elif flag == 'ascend':
            gap = [xOFBlock[i][1] - xOFBlock[i-1][1] for i in range(1, len(xOFBlock))]
            gap_index = np.where(gap == max(gap))[-1][-1]
    return gap_index


# 定义阈值
def findThreshold(of_block, f):
    of_block = np.array(of_block)
    mean2 = sum(of_block[:, 1])/len(of_block)
    mean = np.mean(of_block[:, 1], dtype=np.float64)
    std = np.std(of_block[:, 1], dtype=np.float64)
    threshold = mean + f*std
    return threshold


def drawOutlier(data, outlier):
    """
    画出离群点及其离群因子
    :param data: 源数据
    :param outlier: 源数据
    :return:
    """
    if len(outlier):
        plt.scatter(data[:, 0], data[:, 1], s=5, c='g', marker='*')
        plt.scatter(outlier[:, 0], outlier[:, 1], s=5, c='r', marker='*')


def splitData(dataSet, m, n):
    dataBlock = []
    x_span = np.linspace(np.min(dataSet[:, 0]), np.max(dataSet[:, 0])+1, n)
    y_span = np.linspace(np.min(dataSet[:, 1]), np.max(dataSet[:, 1])+1, m)
    for i in range(1, len(x_span)):
        for j in range(1, len(y_span)):
            index = []
            for k in range(0, len(dataSet)):
                if x_span[i-1] <= dataSet[k][0] < x_span[i] and y_span[j-1] <= dataSet[k][1] < y_span[j]:
                    dataBlock.append(dataSet[k])
                    index.append(k)
            if len(index) > 0:
                dataSet = np.delete(dataSet, index, 0)
    return np.array(dataBlock)

# 拆分数据, 拆成2部分
def splitData2(dataSet):
    dataBlock = [[] for i in range(2)]
    x_span = [np.min(dataSet[:, 0]), np.max(dataSet[:, 0])]
    ax = (x_span[0]+x_span[1])/2
    index = [[] for i in range(2)]
    for k in range(0, len(dataSet)):
        if dataSet[k][0] <= ax:
            dataBlock[0].append(dataSet[k])
            index[0].append(k)
        elif dataSet[k][0] > ax:
            dataBlock[1].append(dataSet[k])
            index[1].append(k)
    return dataBlock








# 计算每个点的MinPts-NN距离
def computeKnn2(minPts, dataSet):
    dataSetEdge = getAllIniEdges(dataSet)
    dataSize = len(dataSet)
    knn_Matrix = []
    for i in range(0, dataSize):
        k = 0
        knn_distance = []
        for j in range(0, dataSize):
            if i != j:
                distance = dataSetEdge[i][j]
                if k < minPts:
                    knn_distance.append([j, distance])
                    k = k+1
                    if k == minPts:
                        sortBubble(knn_distance, 'ascend')
                else:
                    if distance < knn_distance[-1][1]:
                        knn_distance[-1] = [j, distance]
                        sortBubble(knn_distance, 'ascend')
        knn_Matrix.append(knn_distance)
    return dataSetEdge, knn_Matrix


def sortBubble2(data, flag):
    if flag == 'ascend':
        for i in range(0, len(data)):
            for j in range(0, len(data)-i-1):
                if data[j] > data[j+1]:
                    temp = data[j]
                    data[j] = data[j+1]
                    data[j+1] = temp
    elif flag == 'descend':
        for i in range(0, len(data)):
            for j in range(0, len(data)-i-1):
                if data[j] < data[j+1]:
                    temp = data[j]
                    data[j] = data[j+1]
                    data[j+1] = temp
    return data


def angleVar(minPts, normalVec):
    cos_block = []
    for i in range(0, minPts):
        for j in range(i+1, minPts):
            #if i != j:
            tem = np.dot(normalVec[i], normalVec[j])
            cos = (np.dot(normalVec[i], normalVec[j]))/(math.sqrt(np.sum(np.square(normalVec[i])))*math.sqrt(np.sum(np.square(normalVec[j]))))
            cos_block.append(cos)
    v = np.var(cos_block)
    return v


def postiveAngNum(minPts, normalVec):
    t = 0
    vec_norm = np.mean(normalVec, axis=0)
    for i in range(0, minPts):
        cos = (np.dot(normalVec[i], vec_norm)) / (math.sqrt(np.sum(np.square(normalVec[i])))*math.sqrt(np.sum(np.square(vec_norm))))
        if cos >= 0:
            t = t + 1
    return t




