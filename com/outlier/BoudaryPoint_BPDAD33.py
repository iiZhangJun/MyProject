import numpy as np
from com.sort.AlgorithmSort import computeKnn2, postiveAngNum, sortBubble2, dataLoad, splitData, angleVar
from matplotlib import pyplot as plt
import math
'''
0.获取数据
'''
dataSet = dataLoad('../../data/t7_10k_32/data31')
'''
1.设置参数
'''
n = len(dataSet)
minPts = math.floor(10*math.log10(n))
# minPts = 20
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
    v_block.append(vi)
    ti = postiveAngNum(minPts, normalVec)
    t_block.append([i,ti])

v_max = max(v_block)
t_max = max(t_block)
for i in range(0, n):
    vi = v_block[i]/v_max
    ti = t_block[i]/t_max
    ksi = vi/ti
    ksi = v_block[i]/t_block[i]
    ksi_block.append([i,ksi])

ksi_max = np.max(np.array(ksi_block)[:, 1])
for i in range(0, n):
    ksi_block[i][1] = ksi_block[i][1]/ksi_max

'''
5. 求结果边界点
'''
# t_block.sort(key=lambda x: x[1], reverse=True)
ksi_block.sort(key=lambda x:x[1])
print(ksi_block)
outliers = []
thres = int(n * 0.5)
indexs = [int(bk[0]) for bk in t_block[0:thres]]
# indexs = [int(bk[0]) for bk in ksi_block[0:thres]]
outliers = []
for k in indexs:
    outliers.append(dataSet[k])
outliers = np.array(outliers)

'''
6.画出结果
'''
if len(outliers):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], s=1, c='g', marker='*')
    plt.scatter(outliers[:, 0], outliers[:, 1], s=1, c='r', marker='*')
plt.show()