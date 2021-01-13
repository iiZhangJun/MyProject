# encoding:utf-8
from matplotlib import pyplot as plt
from com.graph import Graph as gph
import math
import numpy as np
from com.sort.AlgorithmSort import computeKnn, splitData, sortBubble, dataLoad, findThreshold, drawOutlier
from sklearn.cluster import DBSCAN


'''
0.获取数据
'''
dataSet = dataLoad('../../data/t4_6142.dat')
db = DBSCAN(eps=22, min_samples=5).fit(dataSet)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
mem = 1000
'''
1.拆分
'''
for t in range(num_cluster):
    block = dataSet[labels == t]
    n = len(block)
    minPts = 5*math.log10(n)
    knnMatrix = computeKnn(minPts, block)
    # 求KnnMatrix,以每个点的K-distance作为其离群因子
    db_block = []
    for i in range(0, len(block)):
        # lof_i = knnMatrix[i][1][-1]
        lof_i = np.mean(knnMatrix[i][1][0:-1])
        db_block.append([i, lof_i])
    db_block = np.array(sortBubble(db_block, 'descend'))
    threshold = findThreshold(db_block, 0.5)
    print('---- lof阈值为：' + str(threshold) + '--------')
    outliers = np.array(
        [block[np.int32(db_block[i][0])] for i in np.int32(db_block[:, 0]) if db_block[i][1] > threshold])
    if len(outliers) > 0:
        flag = 'DB/' + str(t) + '_DB_'
        path = '../../data/t4_6142/' + str(flag) + '.jpg'
        drawOutlier(block, outliers, np.array(db_block), path)
    else:
        print('此数据集不存在离群点')


"""
DB用来检测离群点比较好一点，检测边界点状况不佳，可考虑结合多种方法的交并集
"""