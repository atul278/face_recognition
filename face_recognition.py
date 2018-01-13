import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
fac=cv2.CascadeClassifier("/home/atul/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
import keras
import os

def load_data(target):
    names=os.listdir(target)
    x=[]
    y=[]
    for i in names:
        path_target=os.path.join(target,i)
        path_image=os.listdir(path_target)
        for j in path_image:
            path_final=os.path.join(path_target,j)
            img=cv2.imread(path_final)
            x.append(img)
            y.append(i)
    return x,y

def process_image(data_in):
    out=[]
    for i in data_in:
        img=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        key=fac.detectMultiScale(img)
        for (a,b,w,h) in key:
#        new=cv2.rectangle(img,(a,b),(a+w,b+h),(255,0,0),2)
            new=img[b:b+h,a:a+w]
        new=cv2.resize(new,(128,128))
        out.append(new)
    return out

x,y=load_data("data")
x_final=process_image(x)

x=np.array(x)
x_final=np.array(x_final)
print (x.shape,x_final.shape)

q,w,e=x_final.shape
x_final=x_final.reshape(q,w,e,1)

y=pd.DataFrame(y,columns=["label"])
y_f=pd.get_dummies(y["label"])
y_final=y_f.values

x_final=x_final.astype("float32")
y_final=y_final.astype("float32")
x_final/=255


from keras.layers import Dense,Convolution2D,MaxPool2D,Dropout,Activation,Flatten
from keras.models import Sequential

model=Sequential()
model.add(Convolution2D(64,(3,3),input_shape=(128,128,1),activation="relu"))
model.add(MaxPool2D((3,3),strides=(2, 2)))
model.add(Convolution2D(128,(3,3),activation="relu"))
model.add(MaxPool2D((3,3),strides=(2, 2)))
model.add(Convolution2D(256,(3,3),activation="relu"))
model.add(MaxPool2D((3,3),strides=(2, 2)))
model.add(Convolution2D(512,(3,3),activation="relu"))
model.add(MaxPool2D((3,3),strides=(2, 2)))
model.add(Flatten())

model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=y_final.shape[1],activation="softmax"))

model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_final,y_final,epochs=50)
model.save("pre_face.h5")
print("dane")
