import json
import os
import argparse
import random
import numpy as np
from collections import defaultdict
import cv2
import multiprocessing as mp
from PIL import Image,ImageFile

def filter_none(x):
    return [i for i in x if i is not None]

def get_largest_connect_component(img): 
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) >= 1:
        max_idx = np.argmax(area)
        img2=np.zeros_like(img)
        cv2.fillPoly(img2, [contours[max_idx]], 1)
        return img2
    else:
        return img

def subwork(img_path,output):
    mask_path=None
    if '|' in img_path:
        mask_path=img_path.split('|')[1]
        img_path=img_path.split('|')[0]
    try:
        img_RGBA=np.array(Image.open(img_path).convert('RGBA'))
    except:
        return None
    if mask_path is not None:
        try:
            img_RGBA[:,:,-1]=np.array(Image.open(mask_path))
        except:
            return None
    alpha=img_RGBA[...,3:]
    seg_mask=(alpha>128).astype('uint8')
    seg_mask=get_largest_connect_component(seg_mask)
    seg_mask_ = np.where(seg_mask)
    y_min,y_max,x_min,x_max = np.min(seg_mask_[0]), np.max(seg_mask_[0]), np.min(seg_mask_[1]), np.max(seg_mask_[1])
    if y_max<=y_min or x_max<=x_min:
        return None
    img_RGBA[:,:,3:]*=seg_mask
    img_RGBA=img_RGBA[y_min:y_max+1,x_min:x_max+1]
    pil_image=Image.fromarray(img_RGBA)
    pil_image.save(output)
    return '*'+output

    
def work(part):
    output_path=part['output']
    del part['output']
    for i in part:
        print(i)
        os.makedirs(os.path.join(output_path,'images',str(i)),exist_ok=True)
    res= {i:filter_none([subwork(j,os.path.join(output_path,'images',str(i),'{}.png'.format(c))) for c,j in enumerate(part[i])]) for i in part}
    print('done',{i:len(res[i]) for i in res})
    return res



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--min_clip', type=float,default=25)
    parser.add_argument('--min_area', type=float,default=0.0)
    parser.add_argument('--max_area', type=float,default=1.0)
    parser.add_argument('--tolerance', type=float,default=1)
    args = parser.parse_args()
    seg_methods=os.listdir(args.input_dir)
    results=[json.load(open(os.path.join(args.input_dir,seg_method,'results.json'))) for seg_method in seg_methods] 
    count=0
    datadict=defaultdict(list)
    for c in zip(*results):
        npc=np.stack([np.array(j['clip_scores']) for j in c],0)
        areas=np.stack([np.array(j['areas']) for j in c],0)
        name=c[0]['name']
        cid=c[0]['id']-1
        npx=np.argmax(npc,0)
        this_bar=min(args.min_clip,np.max(npc)-args.tolerance)
        for k in range(len(npx)):
            if npc[npx[k],k]<this_bar or areas[npx[k],k]<args.min_area or areas[npx[k],k]>args.max_area:
                continue
            seg_method=seg_methods[npx[k]]
            datadict[cid].append('|'.join([os.path.join(args.image_dir,name,f"{k:04d}.png"),os.path.join(args.input_dir,seg_method,name,f"{k:04d}.png")]))
            count+=1
    
    output_path=os.path.dirname(args.output_file)
    mp.set_start_method('spawn',force=True)
    num_threads=128
    pool = mp.Pool(processes=num_threads)
    parts=[{i:datadict[i],'output':output_path} for i in datadict]
    results = pool.map(work,parts,1)
    result={}
    for i in results:
        result.update(i)
    with open(args.output_file,'w') as f:
        json.dump(result,f)
                