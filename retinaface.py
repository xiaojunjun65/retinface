import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from nets.facenet import Facenet
from nets_retinaface.retinaface import RetinaFace
from anchors import Anchors
from box_utils import decode, decode_landm, non_max_suppression
from config import cfg_mnet, cfg_re50
from utils import (Alignment_1, compare_faces, letterbox_image,
                   retinaface_correct_boxes)


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # 设置字体
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label,'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)
    
def preprocess_input(image):
    image -= np.array((104, 117, 123),np.float32)
    return image

#--------------------------------------#
#   一定注意backbone和model_path的对应。
#   在更换facenet_model后，
#   一定要注意重新编码人脸。
#--------------------------------------#
class Retinaface(object):
    _defaults = {
        "retinaface_model_path" : './model_data/Retinaface_mobilenet0.25.pth',
        #-----------------------------------#
        #   可选retinaface_backbone有
        #   mobilenet和resnet50
        #-----------------------------------#
        "retinaface_backbone"   : "mobilenet",
        "confidence"            : 0.5,
        "iou"                   : 0.3,
        #----------------------------------------------------------------------#
        #   是否需要进行图像大小限制。
        #   输入图像大小会大幅度地影响FPS，想加快检测速度可以减少input_shape。
        #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
        #   会导致检测结果偏差，主干为resnet50不存在此问题。
        #   可根据输入图像的大小自行调整input_shape，注意为32的倍数，如[640, 640, 3]
        #----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        #-----------------------------------#
        #   是否需要进行图像大小限制
        #-----------------------------------#
        "letterbox_image"       : True,
        
        "facenet_model_path"    : './model_data/facenet_mobilenet.pth',
        #-----------------------------------#
        #   可选facenet_backbone有
        #   mobilenet和inception_resnetv1
        #-----------------------------------#
        "facenet_backbone"      : "mobilenet",
        "facenet_input_shape"   : [160,160,3],
        "facenet_threhold"      : 0.7,

        "cuda"                  : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Retinaface
    #---------------------------------------------------#
    def __init__(self, encoding=0, model_path, mlu=False, jit=False, save=False, online=False):

        self.__dict__.update(self._defaults)

        if self.retinaface_backbone == "mobilenet":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50

        self.generate()
        self.anchors = Anchors(self.cfg, image_size=(self.retinaface_input_shape[0], self.retinaface_input_shape[1])).get_anchors()
        if online:

            self.cn_engine = retinaf(model_path, mlu, jit)

        try:
            self.known_face_encodings = np.load("./model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone))
            self.known_face_names     = np.load("./model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone))

        except:
            if not encoding:
                print("载入已有人脸特征失败，请检查model_data下面是否生成了相关的人脸特征文件。")
            pass
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):

        self.net        = RetinaFace(cfg=self.cfg, phase='eval', pre_train=False).eval()
        self.facenet    = Facenet(backbone=self.facenet_backbone, mode="predict").eval()

        print('Loading weights into state dict...')
        state_dict = torch.load(self.retinaface_model_path,map_location='cpu')

        self.net.load_state_dict(state_dict)

        state_dict = torch.load(self.facenet_model_path)
        self.facenet.load_state_dict(state_dict, strict=False)

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

            self.facenet = nn.DataParallel(self.facenet)
            self.facenet = self.facenet.cuda()
        print('Finished!')

    def encode_face_dataset(self, image_paths, names):

        face_encodings = []
        for index, path in enumerate(tqdm(image_paths)):
            image = Image.open(path)
            image = np.array(image, np.float32)
            old_image = image.copy()
            
            im_height, im_width, _ = np.shape(image)

            #---------------------------------------------------#
            #   它的作用是将归一化后的框坐标转换成原图的大小
            #---------------------------------------------------#
            scale = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]])
            scale_for_landmarks = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                                np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                                np.shape(image)[1], np.shape(image)[0]])
            
            if self.letterbox_image:
                image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
                anchors = self.anchors
            else:
                anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()

            #---------------------------------------------------#
            #   图片预处理，归一化
            #---------------------------------------------------
            image = preprocess_input(image).transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0).type(torch.FloatTensor)
            
            if self.cuda:
                scale               = scale.cuda()
                scale_for_landmarks = scale_for_landmarks.cuda()
                image               = image.cuda()
                anchors             = anchors.cuda()

            #---------------------------------------------------#
            #   将处理完的图片传入Retinaface网络当中进行预测
            #---------------------------------------------------#
            with torch.no_grad():
                loc, conf, landms = self.net(image)
                    
                #---------------------------------------------------#
                #   Retinaface网络的解码，最终我们会获得预测框
                #   将预测结果进行解码和非极大抑制
                #---------------------------------------------------#
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                boxes = boxes * scale
                boxes = boxes.cpu().numpy()

                conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
                
                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
                landms = landms * scale_for_landmarks
                landms = landms.cpu().numpy()

                boxes_conf_landms = np.concatenate([boxes,conf,landms],-1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)

            if len(boxes_conf_landms)<=0:
                #print(names[index], "：未检测到人脸")
                continue

            results = np.array(boxes_conf_landms)
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
            
            #---------------------------------------------------#
            #   选取最大的人脸框。
            #---------------------------------------------------#
            best_face_location = None
            biggest_area = 0
            for result in results:
                left, top, right, bottom = result[0:4]

                w = right - left
                h = bottom - top
                if w*h > biggest_area:
                    biggest_area = w*h
                    best_face_location = result

            #---------------------------------------------------#
            #   截取图像
            #---------------------------------------------------#
            crop_img = old_image[int(best_face_location[1]):int(best_face_location[3]), int(best_face_location[0]):int(best_face_location[2])]
            landmark = np.reshape(best_face_location[5:],(5,2)) - np.array([int(best_face_location[0]),int(best_face_location[1])])
            crop_img,_ = Alignment_1(crop_img,landmark)

            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = crop_img.transpose(2, 0, 1)
            crop_img = np.expand_dims(crop_img,0)
            #---------------------------------------------------#
            #   利用图像算取长度为128的特征向量
            #---------------------------------------------------#
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)

        np.save("./model_data/{backbone}_face_encoding.npy".format(backbone=self.facenet_backbone),face_encodings)

        np.save("./model_data/{backbone}_names.npy".format(backbone=self.facenet_backbone),names)


    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        # 绘制人脸框
        image = np.array(image, np.float32)

        old_image = np.array(image.copy(), np.uint8)

        #---------------------------------------------------#
        #   Retinaface检测部分-开始
        #---------------------------------------------------#
        im_height, im_width, _ = np.shape(image)

        # 它的作用是将归一化后的框坐标转换成原图的大小
        scale = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]])
        scale_for_landmarks = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]])

        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        image = preprocess_input(image).transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).type(torch.FloatTensor)

        if self.cuda:
            scale               = scale.cuda()
            scale_for_landmarks = scale_for_landmarks.cuda()
            image               = image.cuda()
            anchors             = anchors.cuda()

        #---------------------------------------------------#
        #   将处理完的图片传入Retinaface网络当中进行预测
        #---------------------------------------------------#
        with torch.no_grad():
            loc, conf, landms = self.net(image)  # forward pass
            
            #---------------------------------------------------#
            #   Retinaface网络的解码，最终我们会获得预测框
            #   将预测结果进行解码和非极大抑制
            #---------------------------------------------------#
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
            
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            landms = landms * scale_for_landmarks
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes,conf,landms],-1)
            
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
        
        #---------------------------------------------------#
        #   如果没有预测框则返回原图
        #---------------------------------------------------#
        if len(boxes_conf_landms)<=0:
            return old_image

        boxes_conf_landms = np.array(boxes_conf_landms)
        if self.letterbox_image:
            boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
            
        #---------------------------------------------------#
        #   Retinaface检测部分-结束
        #---------------------------------------------------#
        
        #-----------------------------------------------#
        #   Facenet编码部分-开始
        #-----------------------------------------------#
        face_encodings = []
        for boxes_conf_landm in boxes_conf_landms:
            #----------------------#
            #   图像截取，人脸矫正
            #----------------------#
            boxes_conf_landm    = np.maximum(boxes_conf_landm, 0)
            crop_img            = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
            landmark            = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
            crop_img, _         = Alignment_1(crop_img,landmark)

            #----------------------#
            #   人脸编码
            #----------------------#
            crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
            crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
            with torch.no_grad():
                crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                if self.cuda:
                    crop_img = crop_img.cuda()

                # 利用facenet_model计算长度为128特征向量
                face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                face_encodings.append(face_encoding)
        #-----------------------------------------------#
        #   Facenet编码部分-结束
        #-----------------------------------------------#

        #-----------------------------------------------#
        #   人脸特征比对-开始
        #-----------------------------------------------#
        face_names = []
        for face_encoding in face_encodings:
            # 取出一张脸并与数据库中所有的人脸进行对比，计算得分
            matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
            name = ""
            # 取出这个最近人脸的评分
            # 取出当前输入进来的人脸，最接近的已知人脸的序号
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]: 
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        #-----------------------------------------------#
        #   人脸特征比对-结束
        #-----------------------------------------------#
        
        for i, b in enumerate(boxes_conf_landms):
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(old_image, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)
            
            name = face_names[i]
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2) 
            #--------------------------------------------------------------#
            #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
            #   如果不是必须，可以换成cv2只显示英文。
            #--------------------------------------------------------------#
            old_image = cv2ImgAddText(old_image, name, b[0]+5 , b[3] - 25)
        return old_image,name

    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        old_image = np.array(image.copy(), np.uint8)

        im_height, im_width, _ = np.shape(image)

        scale = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]])
        scale_for_landmarks = torch.Tensor([np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]])

        if self.letterbox_image:
            image = letterbox_image(image, [self.retinaface_input_shape[1], self.retinaface_input_shape[0]])
            anchors = self.anchors
        else:
            anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
            
        image = preprocess_input(image).transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).type(torch.FloatTensor)

        if self.cuda:
            scale               = scale.cuda()
            scale_for_landmarks = scale_for_landmarks.cuda()
            image               = image.cuda()
            anchors             = anchors.cuda()

        with torch.no_grad():
            loc, conf, landms = self.net(image) 
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()

            conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()
            
            landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
            landms = landms * scale_for_landmarks
            landms = landms.cpu().numpy()

            boxes_conf_landms = np.concatenate([boxes,conf,landms],-1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
        
        if len(boxes_conf_landms)>0:
            boxes_conf_landms = np.array(boxes_conf_landms)
            if self.letterbox_image:
                boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
                
            face_encodings = []
            for boxes_conf_landm in boxes_conf_landms:
                boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                crop_img    = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                landmark    = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
                crop_img, _ = Alignment_1(crop_img,landmark)

                crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
                with torch.no_grad():
                    crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                    if self.cuda:
                        crop_img = crop_img.cuda()

                    face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                    face_encodings.append(face_encoding)

            face_names = []
            for face_encoding in face_encodings:
                matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                name = "Unknown"
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]: 
                    name = self.known_face_names[best_match_index]
                face_names.append(name)
        
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                loc, conf, landms = self.net(image) 
                boxes = decode(loc.data.squeeze(0), anchors, self.cfg['variance'])
                boxes = boxes * scale
                boxes = boxes.cpu().numpy()

                conf = conf.data.squeeze(0)[:,1:2].cpu().numpy()

                landms = decode_landm(landms.data.squeeze(0), anchors, self.cfg['variance'])
                landms = landms * scale_for_landmarks
                landms = landms.cpu().numpy()

                boxes_conf_landms = np.concatenate([boxes,conf,landms],-1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            
            if len(boxes_conf_landms)>0:
                boxes_conf_landms = np.array(boxes_conf_landms)
                if self.letterbox_image:
                    boxes_conf_landms = retinaface_correct_boxes(boxes_conf_landms, np.array((self.retinaface_input_shape[0], self.retinaface_input_shape[1])), np.array([im_height, im_width]))
                    
                face_encodings = []
                for boxes_conf_landm in boxes_conf_landms:
                    boxes_conf_landm = np.maximum(boxes_conf_landm, 0)
                    crop_img    = np.array(old_image)[int(boxes_conf_landm[1]):int(boxes_conf_landm[3]), int(boxes_conf_landm[0]):int(boxes_conf_landm[2])]
                    landmark    = np.reshape(boxes_conf_landm[5:],(5,2)) - np.array([int(boxes_conf_landm[0]),int(boxes_conf_landm[1])])
                    crop_img, _ = Alignment_1(crop_img,landmark)

                    crop_img = np.array(letterbox_image(np.uint8(crop_img),(self.facenet_input_shape[1],self.facenet_input_shape[0])))/255
                    crop_img = np.expand_dims(crop_img.transpose(2, 0, 1),0)
                    with torch.no_grad():
                        crop_img = torch.from_numpy(crop_img).type(torch.FloatTensor)
                        if self.cuda:
                            crop_img = crop_img.cuda()

                        face_encoding = self.facenet(crop_img)[0].cpu().numpy()
                        face_encodings.append(face_encoding)

                face_names = []
                for face_encoding in face_encodings:
                    matches, face_distances = compare_faces(self.known_face_encodings, face_encoding, tolerance = self.facenet_threhold)
                    name = "Unknown"
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]: 
                        name = self.known_face_names[best_match_index]
                    face_names.append(name)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
class retinaf():
    def __init__(self,model_path ,mlu=False, jit=False):
        self.mlu = mlu
        self.jit = jit
        net = RetinaFace(cfg=cfg_mnet, phase='eval', pre_train=False).eval()
        MODEL_WEIGHTS = model_path
        if self.mlu:
            import torch_mlu
            import torch_mlu.core.mlu_model as ct
            import torch_mlu.core.mlu_quantize as mlu_quantize
            print('------------- MLU Online Begin! ---------------------')
            self.device = ct.mlu_device()
            print(self.device)
            if self.jit:
                print('-----------Online fusion----------------')
                torch.set_grad_enabled(False)
                ct.set_core_number(4)
                trace_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
                trace_input = trace_input.to(self.device)
            self._model_trt = mlu_quantize.quantize_dynamic_mlu(net)
            self._model_trt.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
            self._model_trt = self._model_trt.to(self.device)
            if self.jit:
                self._model_trt = torch.jit.trace(self._model_trt, trace_input, check_trace=False)
        else:
            print('--------------- CPU Begin! -------------------')
            self._model_trt = net
            self._model_trt.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
        self._model_trt = self._model_trt.eval()
    def generate_offline_model(self, save_path, core_version="MLU220"):
        import torch_mlu.core.mlu_model as ct
        self.device = ct.mlu_device()
        print(self.device)
        print('-----------Online fusion----------------')
        print('Save Offline Model To:{}'.format(save_path))
        ct.save_as_cambricon(save_path)
        torch.set_grad_enabled(False)
        ct.set_core_number(4)
        ct.set_core_version(core_version)
        trace_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        trace_input = trace_input.to(self.device)
        if not self.jit:
            self._model_trt = torch.jit.trace(self._model_trt, trace_input, check_trace=False)
        self._model_trt(trace_input)
        ct.save_as_cambricon(save_path)

    def preprocess(self, image):
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.float()
        if not self.mlu:
            mean = torch.Tensor([0.485, 0.456, 0.406])
            std = torch.Tensor([0.229, 0.224, 0.225])
            # 转成tensor
            image = transforms.functional.to_tensor(image)
            # 减均值 除方差
            image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def infer(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.preprocess(image)
        # MLU在线推理
        if self.mlu:
            cmap, paf = self._model_trt(data.to(self.device))
            cmap, paf = cmap.cpu(), paf.cpu()
            # print('------------- MLU Online END! -----------------')
        # CPU推理
        else:
            cmap, paf = self._model_trt(data)
            # print('--------------- CPU END! ------------------')
        return cmap, paf
