# encoding: utf-8
from matplotlib import pyplot as plt
from com.mst import SimplePrim as sppm
from com.graph import Graph as gph
import time


pathList = ['../../data/t4_6142.dat', '../../data/t5_5614.dat', '../../data/t7_7655.dat', '../../data/t8_6566.dat']
graph = gph.Graph(pathList[0])
plt.scatter(graph.vertex[:, 0], graph.vertex[:, 1], s=10, c='b', marker='o')

time_start = time.time()
prim = sppm.SimeplePrim(graph)
prim.createMST()
time_end = time.time()
print('total:', time_end-time_start)
for i in range(0, len(graph.vertex)):
    frontPoint = prim.edge_parent[i]
    if frontPoint != -1:
        plt.plot([graph.vertex[frontPoint][0], graph.vertex[i][0]], [graph.vertex[frontPoint][1], graph.vertex[i][1]], c='r')
plt.show()


