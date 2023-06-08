import json
import importlib
import torch
from option import get_option
import glob
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def renorm(x):
    xf=x.flatten(1)
    xa=xf.min(dim=1,keepdim=True)[0].unsqueeze(-1)
    xb=xf.max(dim=1,keepdim=True)[0].unsqueeze(-1)
    return  (x-xa)/(xb-xa)

class selfreformer:
    def __init__(self,device,ckpt='/mnt/home/SelfReformer/best_DUTS-TE.pt'):
        opt = dict(model='network',transformer = [[2, 1, 512, 3, 49],
                           [2, 1, 320, 3, 196],
                           [2, 1, 128, 3, 784],
                           [2, 1, 64, 3, 3136]])
        self.module = importlib.import_module("model.{}".format(opt['model'].lower()))
        self.dev = device
        self.net = self.module.Net(opt).eval().to(self.dev)
        state_dict = torch.load(ckpt, map_location='cpu')
        self.net.load_state_dict(state_dict)
        self.norm = transforms.Normalize(mean=(0.485, 0.458, 0.407),std=(0.229, 0.224, 0.225))
    
    def forward(self,img,**kwargs):
        img=self.norm(img).to(self.dev)
        b, c, h, w = img.shape
        img=F.interpolate(img, (224,224), mode='bilinear', align_corners=False)
        pred = self.net(img)
        pred_sal = F.pixel_shuffle(pred[-1], 4)
        pred_sal = F.interpolate(pred_sal, (h,w), mode='bilinear', align_corners=False)
        pred_sal = torch.sigmoid(pred_sal).squeeze()
        pred_sal=renorm(pred_sal)
        pred_sal = (pred_sal * 255.).detach().cpu().numpy().astype('uint8')
        return pred_sal

