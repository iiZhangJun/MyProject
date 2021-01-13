from sklearn.cluster import KMeans
import numpy as np
from com.util.dataAbout import dataLoad
import matplotlib.pyplot as plt
#matplotlib inline

X = dataLoad('../../data/t7_10k.dat')
y_pred = KMeans(n_clusters=32, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=1)
plt.show()



