import cv2
import glob
import numpy as np
from PIL import Image
#Train data
train = []
files = glob.glob ("./skins/*.jpg") # your image path
i=0
for myFile in files:
    
    image = cv2.imread (myFile,cv2.IMREAD_UNCHANGED)
    # image=Image.open(myFile)
    if image is None:
        print(i,"Error reading the image")
    if image is not None and image.shape==(64,64,4):
        
        data = np.array( image, dtype='uint8' )
        train.append(data)
    i+=1
train = np.array(train) 

print("shape before reshaping",train.shape)
train = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
print("shape after reshaping",train.shape)
np.save('./dataset/data',train)