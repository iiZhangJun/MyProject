# encoding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from com.sort.AlgorithmSort import dataLoad, computeKnn, computeLOF, sortBubble, drawOutlier, splitData,findThreshold
from com.boundary.detectAlgorithm import checkOutlier
from sklearn.cluster import DBSCAN
'''
0.获取数据
'''
dataSet = dataLoad('../../data/t7_10k_8/data0')

'''
2.针对拆分后的每部分分别其中每个点的MinPts-NN,并计算对应MinPts的每个点的LOF
3.从temple_lof_block分别取每个点的最大的lof最为其LOF,将最终每个点的LOF按降序排序
temple_lof_block 存储MinPtsLB ~ MinPtsUB 范围所有点的LOF
'''

block = dataSet
n=len(block)
MinPtsLB = n/4
MinPtsUB = n/3
lof_block = [[i, 0] for i in range(0, len(block))]
if len(block) < MinPtsUB:
    MinPtsUB = MinPtsLB+1
knnMatrix = computeKnn(MinPtsUB, block)
for minPts in range(MinPtsLB, MinPtsUB):
    for i in range(0, len(block)):
        lof_i = computeLOF(i, knnMatrix, minPts)
        if lof_i > lof_block[i][1]:
            lof_block[i][1] = lof_i
lof_block.sort(key=lambda x: x[1], reverse=True)
thres = int(n * 0.5)
indexs = [int(bk[0]) for bk in lof_block[0:thres]]
outliers = []
for k in indexs:
    outliers.append(block[k])
outliers = np.array(outliers)
# indexs = checkOutlier(indexs, 10, knnMatrix)
outliers = np.array([block[index] for index in indexs])
if len(outliers) > 0:
    drawOutlier(block, outliers)
    plt.show()
else:
    print('此数据集不存在离群点')
plt.show()