from PIL import Image
from IPython.display import display
import torch as th
import json
import sys,os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import tqdm
import clip
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import lmdb
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
th.set_grad_enabled(False)


def init(args,rank=None):
    global clip_model,preprocess,seg_model
    if rank is None:
        rank=th.multiprocessing.current_process()._identity[0]-1
    print("init process GPU:",rank)
    device = th.device('cuda:%s'%rank)
    th.cuda.set_device(device)
    th.cuda.empty_cache()
    clip_model, preprocess = clip.load("ViT-L/14", device=rank)
    n_px=224
    preprocess=transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    if args.seg_method=='clipseg':
        sys.path.insert(0,os.path.join(os.getcwd(),'clipseg'))
        from export import clipseg_matting
        seg_model=clipseg_matting(device)
    elif args.seg_method=='UFO':
        sys.path.insert(0,os.path.join(os.getcwd(),'UFO'))
        from export import UFO
        seg_model=UFO(device)
    elif args.seg_method=='selfreformer':
        sys.path.insert(0,os.path.join(os.getcwd(),'SelfReformer'))
        from export import selfreformer
        seg_model=selfreformer(device)
    elif args.seg_method=='U2Net':
        sys.path.insert(0,os.path.join(os.getcwd(),'U-2-Net'))
        from export import u2net 
        seg_model=u2net(device)
    elif args.seg_method=='full':
        class fullmask:
            def __init__(self):
                pass
            
            def forward(self,x,prompt):
                B,C,H,W=x.shape
                return 255*np.ones([B,H,W],dtype='uint8')
        seg_model=fullmask()
    




def seg_image(classname,path,total,batch_size,output_path):
    text='a photo of ' + ' '.join(classname.split('_')) 
    clips=[]
    areas=[]
    path=os.path.join(path,classname)
    files=sorted(os.listdir(path))
    output_path=os.path.join(output_path,classname)
    os.makedirs(output_path,exist_ok=True)
    for i in range(total//batch_size):
        try:
            imgs=torch.stack([transforms.ToTensor()(Image.open(os.path.join(path,fn)).convert('RGB')) for fn in files[i*batch_size:(i+1)*batch_size]],0)
            mask=seg_model.forward(imgs,prompt=text)
            mask_im=torch.from_numpy(mask>128).float().unsqueeze(1)
            mask_area=torch.sum(mask_im,dim=[1,2,3])/mask_im.shape[2]/mask_im.shape[3]
            wb_im=imgs*mask_im+torch.ones_like(imgs)*(1-mask_im)
            text_feature = clip.tokenize(text).cuda()
            _, logits_per_text = clip_model(preprocess(wb_im).cuda(), text_feature)
            logits_per_text=logits_per_text.view(-1).cpu().tolist()
            mask_area=mask_area.cpu().tolist()
            for j in range(mask.shape[0]):
                cv2.imwrite(os.path.join(output_path,files[i*batch_size+j]),mask[j])
        except Exception as e:
            print(e)
            logits_per_text=[-1]*batch_size
            mask_area=[0]*batch_size
        clips+=logits_per_text
        areas+=mask_area
    return clips,areas

        
        
    


from collections import defaultdict
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
import argparse
import os
d1=defaultdict(list)
d2=defaultdict(list)
if __name__=="__main__":
    _ = clip.load("ViT-L/14")
    del _
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--seg_method', type=str)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    output_path=os.path.join(args.output_dir,args.seg_method)
    mp.set_start_method('spawn',force=True)
    with open(os.path.join(args.input_dir,'results.json')) as f:
        t_classes=json.load(f)
    classes=[i['name'] for i in t_classes]
    num_gpus=th.cuda.device_count()
    if num_gpus>1:
        pool = mp.Pool(processes=num_gpus,initializer=init,initargs=(args,))
    else:
        init(args,rank=0)
    if num_gpus>1:
        results = pool.starmap(seg_image, [(i,args.input_dir,args.samples,args.batch_size,output_path) for i in classes],1)
    else:
        results = [seg_image(i,args.input_dir,args.samples,args.batch_size,output_path) for i in classes]
    r_classes=deepcopy(t_classes)
    for i,j in zip(r_classes,results):
        i['clip_scores']=j[0]
        i['areas']=j[1]
    with open(os.path.join(output_path,"results.json"),'w') as f:
        json.dump(r_classes,f)
