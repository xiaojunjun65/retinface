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

img = cv2.imread("zgj_1.jpg")
src,name = predict(img)
print(name)
cv2.imshow("aa",src)
cv2.waitKey(0)