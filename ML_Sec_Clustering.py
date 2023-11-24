# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:29:26 2023

@author: Bora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Kmeans memory leak durumundan kaçınmak için
# import os
# os.environ["OMP_NUM_THREADS"] = '1'
##--------------------------------------------


veriler = pd.read_csv("musteriler.csv")

x_values = veriler.iloc[:,3:].values

# Kmeans

from sklearn.cluster import KMeans

kmns = KMeans(n_clusters=4, init="k-means++")
kmns.fit(x_values)

print(kmns.cluster_centers_)
results = []

for i in range(1,11):
    kmns = KMeans(n_clusters=i, init="k-means++", random_state=15)
    kmns.fit(x_values)
    results.append(kmns.inertia_)

plt.plot(range(1,11),results)
plt.show()

kmns = KMeans(n_clusters=4, init="k-means++", random_state=123)
y_tahmin = kmns.fit_predict(x_values)
print(y_tahmin)

plt.scatter(x_values[y_tahmin==0,0], x_values[y_tahmin==0,1], s=100, c='red')
plt.scatter(x_values[y_tahmin==1,0], x_values[y_tahmin==1,1], s=100, c='blue')
plt.scatter(x_values[y_tahmin==2,0], x_values[y_tahmin==2,1], s=100, c='green')
plt.scatter(x_values[y_tahmin==3,0], x_values[y_tahmin==3,1], s=100, c='yellow')
plt.title("Kmeans")
plt.show()

# Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

y_tahmin = ac.fit_predict(x_values)
print(y_tahmin)

plt.scatter(x_values[y_tahmin==0,0], x_values[y_tahmin==0,1], s=100, c='red')
plt.scatter(x_values[y_tahmin==1,0], x_values[y_tahmin==1,1], s=100, c='blue')
plt.scatter(x_values[y_tahmin==2,0], x_values[y_tahmin==2,1], s=100, c='green')
plt.scatter(x_values[y_tahmin==3,0], x_values[y_tahmin==3,1], s=100, c='yellow')
plt.title("Agglomerated")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x_values, method='ward'))
plt.show()