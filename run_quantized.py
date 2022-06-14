#
# import warnings
# import torch.nn as nn
# import torch
# from nets_retinaface.retinaface import RetinaFace
# from config import cfg_mnet, cfg_re50
# warnings.filterwarnings('ignore')
# import argparse
# # from ssd import SSD
# from PIL import Image
# import os
# import torch
# from retinaface import Retinaface
#
#
#
# def run_quantized_model():
#     cfg = cfg_mnet
#     net= RetinaFace(cfg=cfg, phase='eval', pre_train=False).eval()
#     state_dict = torch.load("./model_data/Retinaface_mobilenet0.25.pth",map_location='cpu')
#     net.load_state_dict(state_dict)
#
#     mean = [0.0, 0.0, 0.0]
#     std = [1.0, 1.0, 1.0]
#     qconfig = {'iteration': opt.image_number, 'use_avg': False, 'data_scale': 1.0, 'mean': mean, 'std': std,
#                'per_channel': True, 'firstconv': False}
#
#     quantized_model = mlu_quantize.quantize_dynamic_mlu(model, qconfig,
#                                                         dtype='int8' if opt.quantized_mode == 1 else 'int16',
#                                                         gen_quant=True)
#
#     stride = int(model.stride.max())  # model stride
#     imgsz = check_img_size(opt.img_size, s=stride)  # check img_size
#
#     dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
#
#     count = 0
#     t0 = time.time()
#     for path, img, im0s, vid_cap in dataset:
#         count += 1
#         img = torch.from_numpy(img)
#         img = img.float()  # uint8 to fp32
#         img /= 255.0  # 0 - 255 to 0.0 - 1.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#         img.to(device)
#         # Inference
#         t1 = time_synchronized()
#
#         pred = quantized_model(img, augment=opt.augment)
#
#         # Apply NMS
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
#                                    max_det=opt.max_det)
#         t2 = time_synchronized()
#
#         # Process detections
#         for i, det in enumerate(pred):  # detections per image
#
#             p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
#
#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # img.jpg
#             s += '%gx%g ' % img.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#
#                 # Print results
#                 for c in det[:, -1].unique():
#                     n = (det[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (int(n) > 1)}, "  # add to string
#
#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
#                     c = int(cls)  # integer class
#                     label = f'{names[c]} {conf:.2f}'
#                     plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)
#
#             # Print time (inference + NMS)
#             print(f'{s}Done. ({t2 - t1:.3f}s)')
#
#             # Save results (image with detections)
#             cv2.imwrite(save_path, im0)
#
#         if count == opt.image_number:
#             break
#     print(f"Results saved to {opt.save_dir}")
#     model_name = opt.weights.split('.')[0].split('/')[-1]
#     checkpoint = quantized_model.state_dict()
#     if opt.quantized_mode == 1:
#         if not os.path.exists(opt.quantized_model_path):
#             os.mkdir(opt.quantized_model_path)
#         torch.save(checkpoint, '{}/{}-int8.pth'.format(opt.quantized_model_path, model_name))
#     else:
#         opt.quantized_model_path = './weights_int16'
#         if not os.path.exists(opt.quantized_model_path):
#             os.mkdir(opt.quantized_model_path)
#         torch.save(checkpoint, '{}/{}-int16.pth'.format(opt.quantized_model_path, model_name))
#
#     print(f'Done. ({time.time() - t0:.3f}s)')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='model.pt path(s)')
#     parser.add_argument('--cfg', type=str, default='models/yolov3.yaml', help='model.yaml path')
#     parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
#     parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
#     parser.add_argument('--save_dir', type=str, default='output', help='the path to save results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#
#     parser.add_argument('--image_number', type=int, default=2, help='test image number')
#     parser.add_argument("--quantized_mode", type=int, default=1, help="1-int8 2-int16")
#     parser.add_argument('--quantized_model_path', default='./weights_int8', type=str, help='Quantized model path')
#     parser.add_argument('--use_tiny', action='store_true', help='use yolo tiny')
#     opt = parser.parse_args()
#     print(opt)
#     run_quantized_model()

import torch
import torch.nn as nn
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision
import torchvision.transforms as transforms
import os
import time
import cv2
import numpy as np
import json
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='openpose inference parameters')
    parser.add_argument('--model', type=str, default='model_data/Retinaface_mobilenet0.25.pth',
                        help='origin model path')
    parser.add_argument('--image', type=str, default='test1.jpg',
                        help='image used to quantize')
    parser.add_argument('--save', type=str, default='models/face_int18.pth',
                        help='path to save quantized model')
    args = parser.parse_args()
    return args


def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    # 转成tensor
    image = transforms.functional.to_tensor(image)
    # 减均值 除方差
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


if __name__ == "__main__":
    args = parse_args()
    from nets_retinaface.retinaface import RetinaFace
    from config import cfg_mnet, cfg_re50

    cfg = cfg_mnet
    net= RetinaFace(cfg=cfg, phase='eval', pre_train=False).eval()
    state_dict = torch.load("./model_data/Retinaface_mobilenet0.25.pth",map_location='cpu')
    net.load_state_dict(state_dict)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 调用量化接口
    qconfig = {'iteration': 1, 'use_avg': False,
               'mean': mean, 'std': std, 'data_scale': 1.0, 'firstconv': True,
               'per_channel': False
               }
    quantized_net = mlu_quantize.quantize_dynamic_mlu(net.eval(), qconfig_spec=qconfig, dtype='int8', gen_quant=True)
    quantized_net = quantized_net.eval().float()

    image = cv2.imread(args.image)
    image = cv2.resize(image, (224, 224))
    img = preprocess(image)
    img = img.float()
    # 进行cpu推理
    cmap, paf = quantized_net(img)
    cmap = cmap.detach().numpy().reshape(-1)
    # 保存量化权重
    torch.save(quantized_net.state_dict(), args.save)

