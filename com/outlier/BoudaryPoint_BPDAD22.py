import numpy as np
from com.sort.AlgorithmSort import computeKnn2, postiveAngNum, sortBubble2, dataLoad, splitData, angleVar
from matplotlib import pyplot as plt
import math

'''
0.获取数据
'''
dataSet = dataLoad('../../data/t7_10k_32/data0')
'''
1.设置参数
'''
n = len(dataSet)
m = int(0.02*n)
minPts = math.floor(5*math.log10(n))
# minPts = 30
dataSetEdge, knnMatrix = computeKnn2(minPts, dataSet)
c = math.floor(n*0.02)
dist_cutoff = np.mean(np.array(knnMatrix)[:, m-1, 1])  # sortBubble2(dataSetEdge[2], 'ascend')[c]

print(m)
print(np.mean(np.array(knnMatrix)[:, m-1, 1]))

'''
2.求密度
'''
pi_block = []
for i in range(0, n):
    pi = 0
    for j in range(0, n):
        if dataSetEdge[i][j] < dist_cutoff:
            pi += 1
    pi_block.append([i,pi])

pi_block.sort(key=lambda x:x[1])
print(pi_block)


outliers = []
thres = int(n * 0.5)
# indexs = [int(bk[0]) for bk in t_block[0:thres]]
# indexs = [int(bk[0]) for bk in ksi_block[0:thres]]
indexs = [int(bk[0]) for bk in pi_block[0:thres]]

outliers = []
for k in indexs:
    outliers.append(dataSet[k])
outliers = np.array(outliers)


if len(outliers):
    plt.scatter(dataSet[:, 0], dataSet[:, 1], s=5, c='g', marker='*')
    plt.scatter(outliers[:, 0], outliers[:, 1], s=5, c='r', marker='*')
plt.show()