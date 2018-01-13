import cv2
from matplotlib import pyplot as plt
fac=cv2.CascadeClassifier("/home/atul/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

img= cv2.imread("im1.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.destroyAllWindows()
face=fac.detectMultiScale(gray, 1.3, 5)
print (face)
for (x,y,w,h) in face:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
