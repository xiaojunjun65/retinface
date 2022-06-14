import torch.nn as nn
import torch
from nets_retinaface.retinaface import RetinaFace
from config import cfg_mnet, cfg_re50
cfg = cfg_mnet
net= RetinaFace(cfg=cfg, phase='eval', pre_train=False).eval()
state_dict = torch.load("./model_data/Retinaface_mobilenet0.25.pth",map_location='cpu')
net.load_state_dict(state_dict)
temp = torch.Tensor(1,3,224,224)
out = net(temp)
print(out)
