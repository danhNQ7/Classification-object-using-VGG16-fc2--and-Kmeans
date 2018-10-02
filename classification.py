import numpy as np 
from sklearn.cluster import KMeans
import pickle
import os
#load data
NCLUSTERS = 5
PATH = 'Dataset Caltech-256/database/train/train70.txt'
filename = 'ModelKmeans.sav'

with open(PATH,'r') as f:
    pathImages = f.read().strip('\n').split('\n')
dataX = []
limit = 0
trainY=[]
for path in pathImages:
    temp = path.rfind('/')
    tag = path[path[:temp].rfind('/')+1:temp]
    trainY.append(tag)
if not os.path.exists(filename):
    for path in pathImages:
        temp = path.rfind('/')
        pathFeature = 'Dataset Caltech-256/features/{}.npy'.format(path[path[0:temp].rfind('/')+1:path.rfind('.')])
        dataX.append(np.load(pathFeature)[0])
    print('Prepare Data: Done!')
    clf = KMeans(n_clusters=NCLUSTERS,random_state=0)
    clf.fit(dataX)
    pickle.dump(clf, open(filename, 'wb'))
    # np.save('logLabel',clf.labels_)
    labels = clf.labels_
    print(type(clf.labels_))
    print(clf.cluster_centers_)
else:
    
    with open(filename, 'rb') as file:  
        clf = pickle.load(file)

    labels = clf.labels_
result ={}

for i,val in enumerate(labels):
    if val not in result.keys():
        result[val]=[trainY[i]]
    else:
        result[val].append(trainY[i])
for key in result.keys():
    tmp = result[key]
    result[key] = max(set(tmp),key=tmp.count)
print(result)
with open('label2folder' + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)



