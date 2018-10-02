import numpy as np 
from sklearn.cluster import KMeans
import pickle
import os

from sklearn.metrics import accuracy_score
PATH = 'Dataset Caltech-256/database/train/test30.txt'
filename = 'ModelKmeans.sav'
SAVEFOLDER = 'Dataset Caltech-256/features'
testX = []
TargetTestY =[]
#prepare data:
with open(PATH,'r') as f:
    pathImages = f.read().strip('\n').split('\n')
for path in pathImages:
    temp = path.rfind('/')
    pathFeature = '{}/{}.npy'.format(SAVEFOLDER,path[path[0:temp].rfind('/')+1:path.rfind('.')])
    testX.append(np.load(pathFeature)[0])
    nameFolder = path[path[:temp].rfind('/')+1: temp]
    TargetTestY.append(nameFolder)
# print(TargetTestY)
with open('label2folder' + '.pkl', 'rb') as f:
        tag = pickle.load(f)
#Load model

with open(filename, 'rb') as file:  
    clf = pickle.load(file)
trainY = list(clf.predict(np.array(testX)))
# print(type(trainY))

for index, val in enumerate(trainY):
    # print(val)
    trainY[index] = tag[val]
print('Accurancy: {}'.format(accuracy_score(TargetTestY,trainY)))