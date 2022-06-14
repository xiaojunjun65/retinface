import os

from retinaface import Retinaface

'''
在更换facenet网络后一定要重新进行人脸编码，运行encoding.py。
'''


def genxin():

    retinaface = Retinaface(1)

    list_dir = os.listdir("./face_dataset")

    image_paths = []
    names = []

    for name in list_dir:
        #print(os.path.join("E:/facenet-retinaface-pytorch-main/face_dataset/"+name))
        for i in os.listdir(os.path.join("./face_dataset/"+name)):
            image_paths.append("./face_dataset/"+name+"/"+i)
            #print("..","E:/facenet-retinaface-pytorch-main/face_dataset/"+name+"/"+i)
            names.append(name.split("_")[0])
    retinaface.encode_face_dataset(image_paths,names)



genxin()