from sklearn.cluster import DBSCAN

from com.util.dataAbout import dataLoad
from matplotlib import pyplot as plt

dataSet = dataLoad('../../data/t5_5614.dat')
db = DBSCAN(eps=20, min_samples=5).fit(dataSet)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
print('分簇的数目: %d' % num_cluster)

for i in range(num_cluster):
    print('簇 ', i, '的所有样本:')
    one_cluster = dataSet[labels == i]
    print(len(one_cluster))
    plt.scatter(one_cluster[:, 0], one_cluster[:, 1], s=5, c='r', marker='o')
    plt.show()


