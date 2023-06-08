import os
#from turtle import forward
from PIL import Image
import torch
from torchvision import transforms
from model_image import build_model
import numpy as np
import torch.nn.functional as F

def renorm(x):
    xf=x.flatten(1)
    xa=xf.min(dim=1,keepdim=True)[0].unsqueeze(-1)
    xb=xf.max(dim=1,keepdim=True)[0].unsqueeze(-1)
    return  (x-xa)/(xb-xa)

class UFO:
    def __init__(self,device,ckpt='/mnt/home/UFO/image_best.pth') -> None:
        self.dev=device
        self.net = build_model(device).to(device).eval()
        ckpt=torch.load(ckpt, map_location=self.dev)
        self.net.load_state_dict({i[7:]:ckpt[i] for i in ckpt})
        self.img_size=224
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self,img,**kwags):
        img=self.img_transform(img).to(self.dev)
        b,c,h,w=img.shape
        img=F.interpolate(img,(self.img_size,self.img_size),mode='bilinear')
        _, pred_mask = self.net(img)
        pred_mask=F.interpolate(pred_mask.unsqueeze(1), (h,w), mode='bilinear', align_corners=False).squeeze(1)
        pred_mask=renorm(pred_mask)
        return  (pred_mask * 255.).cpu().numpy().astype('uint8')





        
