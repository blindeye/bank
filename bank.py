import numpy as np
import sklearn.cluster as clust
import sklearn.preprocessing as prepro

f = open("/Users/blindfish/bank/dataset.tsv")

data = []

for i in f.read().split("\n"):
    data.append(i.split("\t"))

data = data[:-1]
cdata = []
for i in data:
    cdata.append(i[0:8])

def category2vec(various):
    result = np.array([])
    various_set = list(set(various))
    for i in various:
        rows = np.zeros(len(various_set))
        for j in range(len(various_set)):
            if i == various_set[j]:
                rows[j]=1
        result = np.append(result,rows)
    return result.reshape(len(various),len(various_set))

def get_ncol(data, col):
    result = []
    for i in data:
        result.append(i[col])
    return np.array(result)

def get_ncol_float(data, col):
    result = []
    for i in data:
        result.append(float(i[col]))
    return np.array(result)

min_max_scaler = prepro.MinMaxScaler()
v1 = min_max_scaler.fit_transform(get_ncol_float(cdata,0))
#print v1
#print len(v1)
v2 = category2vec(get_ncol(cdata,1))
v3 = category2vec(get_ncol(cdata,2))
v4 = category2vec(get_ncol(cdata,3))
v5 = category2vec(get_ncol(cdata,4))
v6 = min_max_scaler.fit_transform(get_ncol_float(cdata,5))
v7 = category2vec(get_ncol(cdata,6))
v8 = category2vec(get_ncol(cdata,7))

trainData = []
for i in range(len(cdata)):
    trainpart=[]
    trainpart.append(v1[i])
    trainpart = trainpart+v2[i].tolist()
    trainpart = trainpart+v3[i].tolist()
    trainpart = trainpart+v4[i].tolist()
    trainpart = trainpart+v5[i].tolist()
    trainpart.append(v6[i])
    trainpart = trainpart+v7[i].tolist()
    trainpart = trainpart+v8[i].tolist()
    trainData.append(trainpart)
    #trainData.append([np.array(v1[i]),v2[i],v3[i],v4[i],v5[i],np.array(v6[i]),v7[i],v8[i]])
trainData = np.array(trainData)
print trainData[0]

#hclust = clust.AgglomerativeClustering(n_clusters=4)
#kclust = clust.k_means(n_clusters = 4, init='k-means++', precompute_distances= 'auto')
#kclust = clust.KMeans(n_clusters = 4, random_state=0).fit(cdata)


