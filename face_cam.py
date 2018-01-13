import cv2
#from matplotlib import pyplot as plt
fac=cv2.CascadeClassifier("/home/atul//opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while(1):
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    key=fac.detectMultiScale(gray)
    for (a,b,w,h) in key:
       frame=cv2.rectangle(frame,(a,b),(a+w,b+h),(255,0,0),2)
       font = cv2.FONT_HERSHEY_SIMPLEX
       cv2.putText(frame,'ATUL',(a,b), font, 1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
