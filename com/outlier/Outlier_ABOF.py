# encoding:utf-8
import numpy as np
np.set_printoptions(suppress=True)
from com.sort.AlgorithmSort import computeKnn, computeABOF_k, sortBubble, dataLoad, splitData2, getABOF_appro, findThreshold, drawOutlier
from matplotlib import pyplot as plt


dataSet = dataLoad('C:/Users/HP/Desktop/data/musk.txt')
# dataBlock = splitData2(dataSet)
# block = np.array(dataBlock[1])

abof_block = []
n = len(dataSet)
minPts = 80  # math.floor(30 * math.log10(n))
knnMatrix = computeKnn(minPts, dataSet)
for i in range(0, n):
    abof_i = getABOF_appro(i, knnMatrix, dataSet)
    # abof_i = computeABOF_k(i, knnMatrix, block)
    abof_block.append([i, abof_i])
abof_block = sortBubble(abof_block, 'ascend')
count = 0
num = 97
for i in range(len(abof_block)):
    f = 0
    if abof_block[i][0] < num:
        count = count + 1
        f = 1
    if count == int(0.1 * num) and f == 1:
        print(count, i)
    if count == int(0.2 * num) and f == 1:
        print(count, i)
    if count == int(0.3 * num) and f == 1:
        print(count, i)
    if count == int(0.4 * num) and f == 1:
        print(count, i)
    if count == int(0.5 * num) and f == 1:
        print(count, i)
    if count == int(0.6 * num) and f == 1:
        print(count, i)
    if count == int(0.7 * num) and f == 1:
        print(count, i)
    if count == int(0.8 * num) and f == 1:
        print(count, i)
    if count == int(0.9 * num) and f == 1:
        print(count, i)
    if count == num and f == 1:
        print(count, i)

# thres = int(n * 0.6)
# indexs = [int(bk[0]) for bk in abof_block[0:thres]]
# outliers = []
# for k in indexs:
#     outliers.append(block[k])
# outliers = np.array(outliers)
# if len(outliers) > 0:
#     drawOutlier(block, outliers)
# else:
#     print('此数据集不存在离群点')
# plt.show()

"""
    # threshold = findThreshold(abof_block, -2)
        # print('---- lof阈值为：' + str(threshold) + '--------')
        # outliers = np.array([block[np.int32(abof_block[i][0])] for i in np.int32(abof_block[:, 0]) if abof_block[i][1] < threshold])
        # if len(outliers) > 0:
        #     drawOutlier(block, outliers, np.array(abof_block))
        # else:
        #     print('此数据集不存在离群点')
    
        # bd = findMaxGapIndex(abof_block, 'ascend')
        # if bd != -1:
        #     print('---- lof分界为：' + str(bd) + '--------')
        #     outliers = np.array([block[np.int32(abof_block[i][0])] for i in range(0, bd+1)])
        #     drawOutlier(block, outliers, np.array(abof_block))
        # else:
        #     print('次数据集不存在离群点')
"""
