from model import U2NET 
import torch
from torchvision import transforms
import torch.nn.functional as F

def renorm(x):
    xf=x.flatten(1)
    xa=xf.min(dim=1,keepdim=True)[0].unsqueeze(-1)
    xb=xf.max(dim=1,keepdim=True)[0].unsqueeze(-1)
    return  (x-xa)/(xb-xa)

class u2net:
    def __init__(self,device) -> None:
        self.dev = device
        self.net=U2NET(3,1).eval().to(self.dev)
        self.net.load_state_dict(torch.load('/mnt/home/syn4det/U-2-Net/saved_models/u2net.pth',map_location=self.dev))
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self,img,**kwargs):
        img_input = self.transform(img)
        H,W=img_input.shape[2:]
        d1,d2,d3,d4,d5,d6,d7 = self.net(F.interpolate(img_input.to(self.dev),(256, 256),mode='bilinear'))
        pred = d1[:,:1,:,:]
        pred=F.interpolate(pred,(H,W),mode='bilinear').squeeze(1)
        pred=renorm(pred)
        return (pred.cpu().numpy()*255).astype('uint8')