# encoding: utf-8
from com.util.dataAbout import dataLoad
from sklearn.cluster import DBSCAN
import numpy as np
from matplotlib import pyplot as plt


'''
0.获取数据
'''
dataSet = dataLoad('../../data/t8_6566.dat')
outliers_fraction = 0.25  #异常样本比例
'''
1.拆分
'''
db = DBSCAN(eps=20, min_samples=6).fit(dataSet)
labels = db.labels_
num_cluster = len(set(labels)) - (1 if -1 in labels else 0)
gloable_outlier = dataSet[labels == -1]
if len(gloable_outlier) > 0:
    plt.scatter(gloable_outlier[:, 0], gloable_outlier[:, 1], s=5, c='r', marker='*')
mem = 1000
gloable_outlier = dataSet[labels == -1]
if len(gloable_outlier) > 0:
    plt.scatter(gloable_outlier[:, 0], gloable_outlier[:, 1], s=5, c='r', marker='*')
    plt.show()
'''

'''
for t in range(num_cluster):
    block = dataSet[labels == t]
    print(len(block))
    plt.scatter(block[:, 0], block[:, 1], s=5, c='b', marker='*')
    plt.show()