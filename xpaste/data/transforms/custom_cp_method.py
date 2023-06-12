import numpy as np
from xpaste.data.transforms.possion_blending import poisson_edit
import random
import cv2
def blend_image(dst_img,src_img,composed_mask,cp_method):
    cp_method=random.sample(cp_method,1)[0]
    if cp_method=='basic':
        src_img=src_img[:3]
        return dst_img*(1-composed_mask)+src_img*composed_mask
    if cp_method=='alpha':
        assert src_img.shape[0]==4
        alpha=src_img[3:]/255
        src_img=src_img[:3]
        return dst_img*(1-alpha)+src_img*alpha
    if cp_method=='gaussian':
        src_img=src_img[:3]
        composed_mask=cv2.blur(composed_mask.astype('float32'),(5,5))
        return dst_img*(1-composed_mask)+src_img*composed_mask
    if cp_method=='possion':
        src_img=src_img[:3].transpose(1,2,0)
        dst_img=dst_img.transpose(1,2,0)
        return poisson_edit(src_img,dst_img,composed_mask).transpose(2,0,1)