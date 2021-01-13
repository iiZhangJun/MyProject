# encoding:utf-8
import numpy as np
import sys


class Graph:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.vertex = self.dataLoad()
        # self.adjacentMatrix = self.getAllIniEdges()

    def dataLoad(self):
        data = []
        with open(self.dataPath, mode='r', encoding='utf-8') as file:
            for line in file.readlines():
                d = line.strip(' ').split(' ')
                data.append(np.float64(d))
        file.close()
        return np.array(data)

    def getAllIniEdges(self):
        """
        得到第i个点和第j个点间的距离，这些距离权值保存在邻接矩阵中
        :return:
        """
        adjacentMatrix = np.full(shape=(len(self.vertex), len(self.vertex)), fill_value=sys.maxsize, dtype=np.float64)
        for i in range(0, len(self.vertex)):
            for j in range(i+1, len(self.vertex)):
                distance = np.sqrt(np.sum(np.square(np.subtract(self.vertex[i], self.vertex[j]))))
                adjacentMatrix[i][j] = adjacentMatrix[j][i] = distance
        return adjacentMatrix