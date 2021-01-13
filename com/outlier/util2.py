# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 01:03:13 2018

@author: sbml1
"""
import numpy as np
from scipy.stats import norm
from math import isnan

# 欧氏距离
def distance_euclidean(instance1, instance2):
    """Computes the distance between two instances. Instances should be arrays of equal length.
    Returns: Euclidean distance
    """
    # check if instances are of same length
    if len(instance1) != len(instance2):
        raise AttributeError("Instances have different number of arguments.")

    distance = np.linalg.norm(instance1 - instance2)
    return distance
def eucledian_dist(lat1,lon1,lat2, lon2):
    return (abs(float(lat2)-float(lat1))**2+abs(float(lon2)-float(lon1))**2)**0.5

# k个最近邻 KNN 和 距离
def k_nearest_neighbors(instances, instance, k):
    """Computes the k-distance of instance as defined in paper. It also gatheres the set of k-distance neighbours.
    Returns: (k-distance, k-distance neighbours)"""
    distances = {}
    for instance2 in instances:
        distance_value = distance_euclidean(instance, instance2)
        if distance_value in distances:
            distances[distance_value].append(instance2)
        else:
            distances[distance_value] = [instance2]
    distances = sorted(distances.items())
    neighbours = []
    k_distance_value = 0
    for i in range(1, len(distances)):
        neighbours.extend(distances[i][1])
        # extend()在列表末尾一次性追加另一个序列中的多个值
        if len(neighbours) == k:
            k_distance_value = distances[i][0]
            break
        elif len(neighbours) > k:
            k_distance_value = distances[i - 1][0]
            break
    return k_distance_value, neighbours


def kNN_dic(instances, k):
    """Computes the kNN for each instance
    Retrun: dic{instanceNo,kNNNos}
    instanceNo means the sort instance in instances
    kNNNos means the numbers of kNN
    """
    dic = {}
    N = len(instances)
    for i in range(N):
        p = instances[i]
        distances = {}
        for j in range(N):
            q = instances[j]
            distance_value = distance_euclidean(p, q)
            if distance_value in distances:
                distances[distance_value].append(j)
            else:
                distances[distance_value] = [j]
        distances = sorted(distances.items())
        neighbours = []
        for m in range(1, len(distances)):
            neighbours.extend(distances[m][1])
            if len(neighbours) >= k:
                break
        dic[i] = neighbours
    return dic
