# encoding: utf-8
from sklearn.cluster import KMeans
from mst import Prim as pm
from mst import SimplePrim as sppm
from mst.Kruskal import Kruskal
from util.dataAbout import dataLoad, combineEdge
from util.border import getBorder,getAllIniEdges
import time
import numpy as np
from util.dataAbout import dataLoad
import matplotlib.pyplot as plt
#matplotlib inline


M=16    #  分区数
dataBlock = [[] for i in range(M)]
dataSet = dataLoad('../../../data/t7_10k.dat')
sumTime = 0
start = time.time()
y_pred = KMeans(n_clusters=M, random_state=9).fit_predict(dataSet)
for k in range(M):
    index = [i for i,v in enumerate(y_pred) if v==k]
    dataBlock[k] = np.array([dataSet[id] for id in index])
    np.savetxt('F:/Python/EMST/cluster/t7_10k/'+str(M)+'/data'+str(k), dataBlock[k], delimiter=" ")

end = time.time()
sumTime = sumTime + (end-start)
plt.scatter(dataSet[:, 0], dataSet[:, 1], c=y_pred, s=1)
plt.show()

time_start = time.time()
n = len(dataSet)
edgelist = []
time_end = time.time()
sumTime += (time_end - time_start)
outliers=[]

# 分块EMST生成
for k in range(M):
    plt.subplot(4,4,k+1)
    block = np.array(dataBlock[k])
    plt.scatter(block[:, 0], block[:, 1], color='b', s=1)
    time_start = time.time()
    adjacentMatrix = getAllIniEdges(block)
    prim = pm.Prim(block, adjacentMatrix)
    prim.createMST()
    edgelist = edgelist + combineEdge(block, dataSet, prim.edge_parent, prim.dist)
    time_end = time.time()
    sumTime += (time_end - time_start)
    print('第 ' + str(k) + ' 块数据的EMST生成+整合时间：-- ', time_end - time_start)

    time_start = time.time()
    outlier = getBorder(block,adjacentMatrix, 60, 0.5)
    if k == 0:
        outliers = outlier
    else:
        outliers = np.vstack((outliers, outlier))
    time_end = time.time()
    sumTime += time_end - time_start
    print('第' + str(k) + '块数据的边界点识别耗时：', time_end - time_start)
    plt.scatter(outlier[:, 0], outlier[:, 1], color='r', s=1)

plt.show()
time_start = time.time()
prim_out = sppm.SimeplePrim(outliers)
prim_out.createMST()
time_end = time.time()
sumTime += (time_end - time_start)
print('所有边界点EMST生成耗时:', time_end-time_start)

# 整合的时候先判断edge[f,l,dist]或[l,f,dist]是否在edgeList中，若不在加进去，若存在则不操作，
time_start = time.time()
edgelist = edgelist+combineEdge(outliers, dataSet, prim_out.edge_parent, prim_out.dist)
time_end = time.time()
sumTime += (time_end - time_start)
print('边界点EMST整合时间：', time_end - time_start)


print("稀疏图的边集数： " + str(len(edgelist)))
print('----------*** 最终的稀疏图EMST生成 ***-----------')
time_start = time.time()
kruskal = Kruskal(dataSet)
tree = kruskal.createMST(edgelist)
time_end = time.time()
sumTime += (time_end - time_start)
print('总用时：', sumTime)

tree.sort(key=lambda x:x[2])
print(sum(np.array(tree)[:,2]))