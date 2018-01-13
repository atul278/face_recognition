import cv2
import numpy as np
import pandas as pd
fac=cv2.CascadeClassifier("/home/atul/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
from keras.models import load_model
model=load_model("pre_face.h5")

def out(image):
    name=""
    new=cv2.resize(image,(128,128))
    new=np.reshape(new,(1,128,128,1)).astype("float32")
    new/=255
    a=model.predict(new)
    #print(a)
    predict=pd.DataFrame(a,columns=["atul","himalay","khan"])
    predict=(predict>0.5).astype(int)
#    print(predict)
    if (predict.loc[0,"atul"]==1):
        name="atul"
#    if (predict.loc[0,"anshuman"]==1):
#        name="anshuman"
    if (predict.loc[0,"khan"]==1):
        name="khan"
    if (predict.loc[0,"himalay"]==1):
        name="himalay"
    return name

while(1):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    key=fac.detectMultiScale(gray)
    for (a,b,w,h) in key:
       frame=cv2.rectangle(frame,(a,b),(a+w,b+h),(255,0,0),2)
       img=gray[b:b+h,a:a+w]
       nam=out(img)
       font = cv2.FONT_HERSHEY_SIMPLEX
       cv2.putText(frame,nam,(a,b), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
