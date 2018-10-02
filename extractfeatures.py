
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
from PIL import ImageFile
import os
import time

PATH = 'Dataset Caltech-256/database/train/train70.txt'
SAVEFOLDER = 'Dataset Caltech-256/features'

begin = time.time()
model = VGG16(weights='imagenet', include_top=True)
model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
with open(PATH,'r') as f:
    pathImages = f.read().strip('\n').split('\n')
for path in pathImages:
    temp = path.rfind('/')
    dirname = path[path[:temp].rfind('/')+1:temp]
    dirfolder = '{}/{}'.format(SAVEFOLDER,dirname)
    if not os.path.exists(dirfolder):
        os.mkdir(dirfolder)
        print(dirfolder)
    if os.path.exists('{}/{}.npy'.format(dirfolder,path[temp+1:path.rfind('.')])):
        # print('co r')
    else :
        img_path = path
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        np.save('{}/{}'.format(dirfolder,path[temp+1:path.rfind('.')]),features)
    print('Time : {}'.format(time.time()-begin))

# print(features)
# print(type(features))