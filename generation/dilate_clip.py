import clip
import torch
import glob
import cv2
import json
from tqdm.auto import tqdm
import random
import numpy as np
import argparse
from PIL import Image
with open('Pool_100_stable_b21.json') as f:
    a=json.load(f)

t=[]
for i in a:
    t+=a[i]

t=random.choices(t,k=1000)

clip_model, preprocess = clip.load("ViT-L/14", device='cuda')

def read_im(file):
    file=file.split('|')
    cat=' '.join(file[0][:-4].split('/')[-1].split('_')[:-1])
    img=cv2.imread(file[0],cv2.IMREAD_UNCHANGED)
    rgb=cv2.cvtColor(img[:,:,:3],cv2.COLOR_BGR2RGB)
    mask=img[:,:,-1]
    if len(file)==2:
        mask=cv2.imread(file[1],cv2.IMREAD_UNCHANGED)
    bin_mask=mask>128
    return rgb,bin_mask.astype('uint8'),cat

def dilate(bin_mask, area=1):
    if area==1:
        return bin_mask
    u,v=np.where(bin_mask)
    H=np.max(u)-np.min(u)+1
    W=np.max(v)-np.min(v)+1
    L=min(H,W)
    if area>1:
        dilate_rate=int((area-1)*L)
        return cv2.dilate(bin_mask,np.ones([dilate_rate,dilate_rate]),iterations=1)
    else:
        erode_rate=int((1-area)*L)
        return cv2.erode(bin_mask,np.ones([erode_rate,erode_rate]),iterations=1)

def get_clip_score(rgb,bin_mask,cat,diltations=[0.8,0.9,1,1.1,1.2]):
    img=torch.stack([preprocess(Image.fromarray(rgb*dilate(bin_mask,i)[:,:,None])) for i in diltations]).cuda()
    text_feature = clip.tokenize(cat).cuda()
    _, logits_per_text = clip_model(img,text_feature)
    return logits_per_text.flatten().cpu().tolist()


diltations=[0.8,0.9,1,1.1,1.2]
clips=[]
for file in tqdm(t):
    try:
        clips.append(get_clip_score(*read_im(file),diltations))
    except:
        pass

with open('dilate_clips.json','w') as f:
    json.dump(dict(diltations=diltations,scores=clips),f)

print(np.mean(clips,dim=0))

    
    


