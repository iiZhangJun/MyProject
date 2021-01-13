# encoding:utf-8
import numpy as np
import math


# 定义阈值
def findThreshold(of_block, f):
    mean = np.mean(of_block[:, 1])
    std = np.std(of_block[:, 1])
    threshold = mean + f*std
    return threshold