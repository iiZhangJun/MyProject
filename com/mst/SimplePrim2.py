# encoding:utf-8
import numpy as np
import sys


class SimeplePrim2:

    def __init__(self, dataSet):
        self.vertex = dataSet
        self.vertex_inTree = [0]
        self.vertex_notInTree = [i for i in range(1, len(self.vertex))]
        self.dist = [sys.maxsize]*len(self.vertex)
        self.edge_parent = [-1]*len(self.vertex)      # 存放父亲节点

    def createMST(self):
        """
        1.Update 2.Scan 3.Add
        :return:
        """
        # 当图中所有的点都并到 vertexInTree时，终止循环
        while self.vertex_notInTree:
            # 1.Update 更新dist和parent列表
            i = self.vertex_inTree[-1]
            lastVtIntree = self.vertex[i]
            for j in self.vertex_notInTree:
                ptNotIntree = self.vertex[j]
                edge_ptTovt = np.sqrt(np.sum(np.square(np.subtract(ptNotIntree, lastVtIntree))))
                if edge_ptTovt < self.dist[j]:
                    self.dist[j] = edge_ptTovt
                    self.edge_parent[j] = i
            # 2. Scan扫描dist，找到权值最短的且顶点不在tree中的点加入到tree中
            minEdge = sys.maxsize
            addVtIndex = -1
            for j in self.vertex_notInTree:
                EdgeToTree = self.dist[j]
                if EdgeToTree < minEdge:
                    minEdge = EdgeToTree
                    addVtIndex = j
            # 3.Add 将dist最小的索引对应的点加入到Intree中，并从NotIntree中移除
            self.vertex_inTree.append(addVtIndex)
            # print('第' + str(addVtIndex) + '个点加入，' + '权值为：' + str(minEdge) + '--' + str(self.dist[addVtIndex]))
            self.vertex_notInTree.remove(addVtIndex)



