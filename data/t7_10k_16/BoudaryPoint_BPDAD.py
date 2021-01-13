import numpy as np
from com.sort.AlgorithmSort import computeKnn2, postiveAngNum, sortBubble2, dataLoad, splitData, angleVar
from matplotlib import pyplot as plt

for t in range(16):
    plt.subplot(4,4,t+1)
    '''
    0.获取数据
    '''
    dataSet = dataLoad('data'+str(t))
    '''
    1.设置参数
    '''
    n = len(dataSet)
    # minPts = math.floor(5*math.log10(n))
    minPts = 30
    dataSetEdge, knnMatrix = computeKnn2(minPts, dataSet)

    '''
    3.求角度ksi
    '''
    v_block = []
    t_block = []
    ksi_block = []
    for i in range(0, n):
        knn = knnMatrix[i]
        normalVec = []
        for j in range(0, minPts):
            x_ji = np.subtract(dataSet[i], dataSet[knn[j][0]])/knn[j][1]
            normalVec.append(x_ji)
        vi = angleVar(minPts, normalVec)
        v_block.append([i, vi])
        ti = postiveAngNum(minPts, normalVec)
        t_block.append([i, ti])
    '''
    5. 求结果边界点
    '''
    t_block.sort(key=lambda x: x[1], reverse=True)
    outliers = []
    thres = int(n * 0.6)
    indexs = [int(bk[0]) for bk in t_block[0:thres]]
    outliers = []
    for k in indexs:
        outliers.append(dataSet[k])
    outliers = np.array(outliers)
    file = "F:\\Python\\MyProject\\data\\t7_10k_4\\outliers_"+str(t)
    # np.savetxt(file, outliers, delimiter=" ")
    '''
    6.画出结果
    '''
    if len(outliers):
        plt.scatter(dataSet[:, 0], dataSet[:, 1], s=1, c='g', marker='*')
        plt.scatter(outliers[:, 0], outliers[:, 1], s=1, c='r', marker='*')
plt.show()