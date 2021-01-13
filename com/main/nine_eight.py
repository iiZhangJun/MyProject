from com.util.dataAbout import splitData, dataLoad, findThreshold,splitData2
from com.main.common_lof import computeKnn, computeLOF, drawOutlier
from matplotlib import pyplot as plt
from com.mst import SimplePrim2 as sppm
import numpy as np
import time


dataSet = dataLoad('../../data/t4_6142.dat')
dataBlock, x, y = splitData(dataSet)
color = ['g', 'b', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
# for i in range(len(dataBlock)):
#     block = np.array(dataBlock[i])
#     ax.scatter(block[:, 0], block[:, 1], color=color[i], s=1)
# plt.show()

'''
将对每一块数据集求边界点的方法封装，不同块拥有不同的阈值参数
'''
def subDetectFunc(block, minPtsLB, minPtsUB, thresParam):
    outliers = []
    MinPtsLB = minPtsLB
    MinPtsUB = minPtsUB
    lof_block = [[i, 0] for i in range(0, len(block))]
    if len(block) < MinPtsUB:
        ax.scatter(block[:, 0], block[:, 1], s=5, c='r', marker='*')
    else:
        knnMatrix = computeKnn(MinPtsUB, block)
        for minPts in range(MinPtsLB, MinPtsUB):
            for i in range(0, len(block)):
                lof_i = computeLOF(i, knnMatrix, minPts)
                if lof_i > lof_block[i][1]:
                    lof_block[i][1] = lof_i
        lof_block = np.array(lof_block)
        threshold = findThreshold(lof_block, thresParam)
        print('---- lof阈值为：' + str(threshold) + '--------')
        indexs = [np.int32(lof_block[i][0]) for i in np.int32(lof_block[:, 0]) if lof_block[i][1] > threshold]
        print(len(block), len(indexs))
        outliers = np.array([block[index] for index in indexs])
        if len(outliers) > 0:
            ax.scatter(outliers[:, 0], outliers[:, 1], s=2, c='r', marker='*')
        else:
            print('此数据集不存在离群点')
    return outliers

# 第一块

block = np.array(dataBlock[0])
ax.scatter(block[:, 0], block[:, 1], color=color[0], s=1)

time_start = time.time()
outliers_1 = subDetectFunc(block, 49, 50, -0.17)
time_end = time.time()
print('total:', time_end-time_start)


# 第二块
block = np.array(dataBlock[1])
ax.scatter(block[:, 0], block[:, 1], color=color[1], s=1)
time_start = time.time()
outliers_2 = subDetectFunc(block, 49, 50, -0.16)
time_end = time.time()
print('total:', time_end-time_start)


# 第三块
block = np.array(dataBlock[2])
ax.scatter(block[:, 0], block[:, 1], color=color[2], s=1)
outliers_3 = subDetectFunc(block, 49, 50, -0.18)
time_end = time.time()
print('total:', time_end-time_start)


# 第四块
block = np.array(dataBlock[3])
ax.scatter(block[:, 0], block[:, 1], color=color[3], s=1)
time_start = time.time()
outliers_4 = subDetectFunc(block, 49, 50, -0.1)
time_end = time.time()
print('total:', time_end-time_start)

plt.show()

outliers = np.vstack((np.vstack((np.vstack((outliers_1, outliers_2)), outliers_3)), outliers_4))


time_start = time.time()
prim = sppm.SimeplePrim2(outliers)
prim.createMST()
time_end = time.time()
print('total:', time_end-time_start)
for i in range(0, len(prim.vertex)):
    frontPoint = prim.edge_parent[i]
    if frontPoint != -1:
        plt.plot([prim.vertex[frontPoint][0], prim.vertex[i][0]], [prim.vertex[frontPoint][1], prim.vertex[i][1]], c='r')
plt.show()