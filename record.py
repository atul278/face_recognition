import cv2
cap=cv2.VideoCapture(0)
import time
count=0
while(1):
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        name="new/image%d.jpg"%count
#        print (name)
        cv2.imshow("image",frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.imwrite(name,frame)
            count+=1
            print (count)
        if k == 27:
            break


cap.release()
cv2.destroyAllWindows()
