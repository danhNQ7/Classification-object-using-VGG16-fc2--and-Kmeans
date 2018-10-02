from glob import glob
import random
listFolder = sorted(glob('Dataset Caltech-256/Images/*'))
print(len(listFolder))
fileTrain = open('Dataset Caltech-256/database/train/train70.txt','w')
fileTest = open('Dataset Caltech-256/database/train/test30.txt','w')
for pathFolder in listFolder:
    listImage = glob(pathFolder+'/*.jpg')
    lenn = len(listImage)
    print(pathFolder)
    listTrain = random.sample(listImage,int(0.7*lenn))
    fileTrain.writelines('\n'.join(listTrain)+'\n')
    listTest = list(set(listImage)-set(listTrain))
    fileTest.writelines('\n'.join(listTest)+'\n')
    print(lenn,len(listTrain),len(listTest))
    # print(listImage)
    # input()
fileTrain.close()
fileTest.close()