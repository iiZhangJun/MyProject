import numpy as np
from com.sort.AlgorithmSort import computeKnn2, postiveAngNum, sortBubble2, dataLoad, splitData, angleVar
from matplotlib import pyplot as plt
from com.boundary.detectAlgorithm import checkOutlier2
import math

'''
0.获取数据
'''
# dataSet = dataLoad('../../clusterBlock/t7_10k/16/data2')
dataSet = dataLoad('../../data/t4_6142.dat')
'''
1.设置参数
'''
n = len(dataSet)
minPts = math.floor(5*math.log10(n))
c = math.floor(n*0.02)
#  print(minPts)
# minPts = 30
dataSetEdge, knnMatrix = computeKnn2(minPts+c, dataSet)
dist_cutoff = np.mean(np.array(knnMatrix)[:, minPts+c-1, 1]) # sortBubble2(dataSetEdge[2], 'ascend')[c]

v_block = []
t_block = []
ksi_block = []
for i in range(0, n):
    knn = knnMatrix[i]
    normalVec = []
    for j in range(0, minPts):
        x_ji = np.subtract(dataSet[i], dataSet[knn[j][0]])/knn[j][1]
        normalVec.append(x_ji)
    # vi = angleVar(minPts, normalVec)
    # v_block.append([i, vi])
    ti = postiveAngNum(minPts, normalVec)
    t_block.append([i, ti])

'''
2.求密度
'''
pi_block = []
for i in range(0, n):
    pi = 1
    for j in range(0, n):
        if dataSetEdge[i][j] < dist_cutoff:
            pi += 1
    pi_block.append(pi)


for i in range(0, n):
    ri = t_block[i][1]/pi_block[i]
    t_block[i][1] = ri


t_block.sort(key=lambda x: x[1], reverse=True)
outliers = []
thres = int(n * 0.5)
indexs = [int(bk[0]) for bk in t_block[0:thres]]
# for i in range(2):
#     indexs = checkOutlier2(indexs, minPts, knnMatrix, dataSet)

for k in indexs:
    outliers.append(dataSet[k])
outliers = np.array(outliers)


if len(outliers):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], s=5, c='g', marker='*')
    plt.scatter(outliers[:, 0], outliers[:, 1], s=5, c='r', marker='*')
plt.show()