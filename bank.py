import numpy as np
import sklearn.cluster as clust

f = open("/Users/blindfish/bank/dataset.tsv")

data = []

for i in f.read().split("\n"):
    data.append(i.split("\t"))

cdata = []
for i in data:
    cdata.append(i[0:8])

trainData = np.array()

hclust = clust.AgglomerativeClustering(n_clusters=4)
#kclust = clust.k_means(n_clusters = 4, init='k-means++', precompute_distances= 'auto')
kclust = clust.KMeans(n_clusters = 4, random_state=0).fit(cdata)


