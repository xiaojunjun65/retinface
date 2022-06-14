

import cv2
import numpy as np
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='openpose inference parameters')
    parser.add_argument('--model', type=str, default='./models/face_int18.pth',
                        help='path of inference model')
    parser.add_argument('--image', type=str, default='test1',
                        help='path of inference image')
    parser.add_argument('--save', type=str, default='images_out/out_online.png', 
                        help='path of inference image')
    parser.add_argument('--mlu', type=bool, default=True, help='True/False for online inference')
    parser.add_argument('--jit', type=bool, default=True, help='True/False fusion of online')
    args = parser.parse_args()
    return args
from retinaface import *
from retinaface import Retinaface
if __name__ == "__main__":
    args = parse_args()
    model_path = args.model
    pose_estimator = Retinaface( encoding=0,model_path, args.mlu, args.jit, args.save, online=True)   # 初始化openpose对象
    show_preview = True
    start = time.time()
    for i in range(100):
        img = cv2.imread(args.image)  
        objects, annot_image = pose_estimator.detect_image(img, return_annotated_image=show_preview)
        if annot_image is not None:
            cv2.imwrite(args.save, annot_image)
    end = time.time()
    print("total time: %ss" %(end-start))
    print("average : ", 100/(end-start), "fps")
