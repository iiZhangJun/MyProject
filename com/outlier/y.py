from sklearn.neighbors import NearestNeighbors
from numpy import *
from com.outlier import util2
import numpy as np
from matplotlib import pyplot as plt

def data_loadDataSet():
    fileName='../../data/flame'
    dataMat=[]
    with open(fileName) as data:
        lines=data.readlines()
        for line in lines:
            lineData=line.strip().split(' ')
            lineData=list(map(lambda x:float(x), lineData))
            dataMat.append(lineData)
    return(np.array(dataMat))

instances = data_loadDataSet()
k=15
nbrs=NearestNeighbors(n_neighbors=15,algorithm='brute').fit(instances)
distances,indices=nbrs.kneighbors(instances)
#distances最近的距离#indices最近邻的索引

zhixin = []
datazx = []
# for i in range(len(instances)):
#     for w in range(len(instances[1])):
#       zx = 0
#       for j in range(1,k):
#         zx = instances[indices[i][j]][w]+zx
#         print(instances[indices[i][j]][w])
#         datazx.append(zx/k)
#
#     zhixin.append(datazx)
def allzhixin(instances):
    allzhixin = np.mean(instances, axis=0)
    return allzhixin
for instancei in instances:
    (k_distance, kNN) = util2.k_nearest_neighbors(instances, instancei, k)
    zhixin.append(allzhixin(kNN))
    # print(zhixin)

nbrs=NearestNeighbors(n_neighbors=15,algorithm='brute').fit(zhixin)
distancesz,indicesz=nbrs.kneighbors(zhixin)

snn=[]
for i in range(len(instances)):
    c=0
    for j in range(1,k):
        for w in range(1,k):
            if(indices[i][j]==indicesz[i][w]):
                c=c+1
                break
    snn.append(c)

sumzx=[]
sump=[]
for i in range(len(instances)):
    s=0
    for j in range(1,k):
        s=s+distances[i][j]
    sz=0
    for j in range(1,k):
        sz=sz+distancesz[i][j]
    sumzx.append(sz)
    sump.append(s)
out=[]
for i in range(len(instances)):
    # a = (snn[i] * snn[i]) / (sump[i] + sumzx[i])
    a=snn[i]/(sump[i]+sumzx[i])
    out.append({"lof": a, "instance": instances[i]})

sort_factors = sorted(out, key=lambda keys: keys['lof'],reverse=False)[:120]
print(sort_factors)

plt.scatter(instances[:,0],instances[:,1],color='b')
for outlier in sort_factors:
    instance = outlier["instance"]
    plt.scatter(instance[0], instance[1], color="r")
plt.show()
