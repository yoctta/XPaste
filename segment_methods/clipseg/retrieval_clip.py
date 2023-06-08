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
from models.clipseg import CLIPDensePredT
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
    global clip_model,preprocess
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

def gen_image(classname,path,batch_size,limit):
    text=('an' if classname[0] in 'aeiou' else 'a') + ' '+ classname 
    clips=[]
    filenames=os.listdir(path)[:limit]
    input_files=[os.path.join(path,i) for i in filenames]
    for i in range(int(np.ceil(len(filenames)/batch_size))):
        batch_files=input_files[i*batch_size:(i+1)*batch_size]
        try:
            in_imgs=[transforms.ToTensor()(Image.open(fn).convert('RGBA')).unsqueeze(0) for fn in batch_files]
            imgs=torch.cat(in_imgs,0)
            #mask_im=imgs[:,-1:]
            imgs=imgs[:,:-1]
            #wb_im=imgs*mask_im+torch.ones_like(imgs)*(1-mask_im)
            wb_im=imgs
            text_feature = clip.tokenize(text).cuda()
            _, logits_per_text = clip_model(preprocess(wb_im).cuda(), text_feature)
            logits_per_text=logits_per_text.view(-1).cpu().tolist()
        except Exception as e:
            print(str(e))
            logits_per_text=[-1]*len(batch_files)
        clips+=logits_per_text
    return {i:j for i,j in zip(filenames,clips)}

        
        
    


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
    sub_dirs=os.listdir('/mnt/data/LVIS_retrieval/masks/U2Net')
    PATH=[os.path.join('/mnt/data/LVIS_retrieval/masks/U2Net',i)  for i in sub_dirs]
    mp.set_start_method('spawn',force=True)
    num_gpus=th.cuda.device_count()
    if num_gpus>1:
        pool = mp.Pool(processes=num_gpus,initializer=init,initargs=(None,))
    else:
        init(None,rank=0)
    if num_gpus>1:
        results = pool.starmap(gen_image, [(sub_dirs[i],PATH[i],20,100) for i in range(len(sub_dirs))],1)
    else:
        results = [gen_image(sub_dirs[i],PATH[i],20,100) for i in range(len(sub_dirs))]
    results={i:j for i,j in zip(sub_dirs,results)}
    with open('/mnt/data/LVIS_retrieval/results_100.json','w') as f:
        json.dump(results,f)
