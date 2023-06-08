import base64
import cv2
import numpy as np
import glob
import os
import json
import multiprocessing as mp
from PIL import Image,ImageFile
import argparse

mask_threshold=128
instance_filter_min=0.05
instance_filter_max=0.95 

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
        if img_path.startswith('/mnt/data/LVIS_retrieval/masks/'):
            img_RGBA=cv2.cvtColor(img_RGBA,cv2.COLOR_BGRA2RGBA)
    except:
        return None
    if mask_path is not None:
        try:
            img_RGBA[:,:,-1]=np.array(Image.open(mask_path))
        except:
            return None
    alpha=img_RGBA[...,3:]
    seg_mask=(alpha>mask_threshold).astype('uint8')
    seg_mask=get_largest_connect_component(seg_mask)
    seg_mask_ = np.where(seg_mask)
    instance_area=len(seg_mask_[0])
    instance_area_percent=instance_area/(seg_mask.shape[0]*seg_mask.shape[1])
    if instance_area_percent<=instance_filter_min or instance_area_percent>=instance_filter_max:
        return None
    y_min,y_max,x_min,x_max = np.min(seg_mask_[0]), np.max(seg_mask_[0]), np.min(seg_mask_[1]), np.max(seg_mask_[1])
    if y_max<=y_min or x_max<=x_min:
        return None
    instance_H=y_max+1-y_min
    instance_W=x_max+1-x_min
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
        os.makedirs(os.path.join(output_path,'images',i),exist_ok=True)
    res= {i:filter_none([subwork(j,os.path.join(output_path,'images',i,'{}_{}.png'.format(c,('gen' if 'LVIS_gen_FG' in j else 'retr')))) for c,j in enumerate(part[i])]) for i in part}
    print('done',{i:len(res[i]) for i in res})
    return res



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--pool_json', type=str, default='/mnt/home/syn4det/instance_pools/b21_retr1000000_stable1000000.json')
    parser.add_argument('--output_path', type=str, default='/mnt/home/syn4det/cleaned_pool_2m/')
    args = parser.parse_args()
    instance_pool=json.load(open(args.pool_json))
    output_path=args.output_path
    part_keys=sorted(list(instance_pool.keys()))[args.rank::args.world_size]
    mp.set_start_method('spawn',force=True)
    num_threads=128
    pool = mp.Pool(processes=num_threads)
    parts=[{i:instance_pool[i],'output':output_path} for i in part_keys]
    results = pool.map(work,parts,1)
    result={}
    for i in results:
        result.update(i)
    with open(os.path.join(output_path,"result_{}.json".format(args.rank)),'w') as f:
        json.dump(result,f)
                
            
            
