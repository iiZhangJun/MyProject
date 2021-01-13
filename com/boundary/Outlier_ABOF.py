# encoding:utf-8
import numpy as np
from com.util.dataAbout import dataLoad, splitData, drawOutlier
from com.util.thresholdAbout import findThreshold
from com.boundary.detectAlgorithm import getABOF_exact, getAllIniEdges,sortBubble

dataBlock = dataLoad('../../data/data0')
n =len(dataBlock)
abof_block = []
edgeMatrix = getAllIniEdges(dataBlock)
for i in range(0, n):
    ang_i = getABOF_exact(i, dataBlock, n, edgeMatrix)
    abof_block.append([i, ang_i])
abof_block = np.array(sortBubble(abof_block, 'ascend'))

# threshold = findThreshold(abof_block, 0)
# outliers = np.array(
#     [dataBlock[np.int32(abof_block[i][0])] for i in np.int32(abof_block[:, 0]) if abof_block[i][1] < threshold])
thres = int(n * 0.6)
indexs = [int(bk[0]) for bk in abof_block[0:thres]]
outliers = []
for k in indexs:
    outliers.append(dataBlock[k])
outliers = np.array(outliers)
if len(outliers) > 0:
    flag = 'ABOF/_ABOF_'
    path = '../../data/t4_6142/' + str(flag) + '.png'
    drawOutlier(dataBlock, outliers, abof_block, path)
else:
    print('此数据集不存在离群点')