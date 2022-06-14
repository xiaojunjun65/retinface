print("1111")
import cv2
import numpy as np
import encoding
from retinaface import Retinaface

retinaface = Retinaface()
def predict(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_image,name = retinaface.detect_image(image)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
    return  r_image,name

# img = cv2.imread("sty_1.jpg")
# src,name = predict(img)
# print(name)
# cv2.imshow("aa",src)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.resize(frame, (200,200))
    src, name = predict(gray)
    print(name)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()