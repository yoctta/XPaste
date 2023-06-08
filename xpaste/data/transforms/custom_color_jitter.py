# Copyright (c) Facebook, Inc. and its affiliates.
import torchvision
import numpy as np
# import random
from numpy import random
import cv2
import torch
from fvcore.transforms.transform import Transform

def convert_color_factory(src: str, dst: str):
    
    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    return convert_color

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')

class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 cid_to_freq_dict,
                 freq_color_filter,
                 use_torchvision = False,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.freq_color_filter = freq_color_filter
        self.cid_to_freq_dict = cid_to_freq_dict
        self.use_torchvision = use_torchvision
        if use_torchvision :
            self.color_jitter_aug = torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)

    def __call__(self, results):
        if self.use_torchvision :
            img_origin = results['image']
            img_trans = self.color_jitter_aug(img_origin)
        else :
            origin_dtype = results['image'].dtype
            img = results['image'].numpy().transpose(1,2,0)
            # cvt rgb to bgr
            img = img[...,::-1]
            # img_ori = img
            # cv2.imwrite('aa-pre.jpg',img)
            img = self.apply_img(img)
            # cv2.imwrite('aa-post.jpg',img)
            # img_show = np.concatenate([img_ori, img], axis=1)
            # cv2.imwrite('aa-show.jpg',img_show)
            img = img[...,::-1]
            img_origin = results['image']
            img_trans = torch.tensor(img.transpose(2,0,1).copy(), dtype=origin_dtype)

        inst = results['instances']
        freq_filter = torch.tensor([self.cid_to_freq_dict[x] in self.freq_color_filter for x in inst.gt_classes.tolist()])
        if len(freq_filter) :
            inst = inst[freq_filter]
        else :
            inst = []
        if len(inst):
            gt_masks = inst.get('gt_masks').tensor
            composed_mask = gt_masks.max(0)[0]
            # composed_mask = np.where(np.any(gt_masks, axis=0), 1, 0).astype(img.dtype)[...,None]
            def cv_show(img, name):
                cv2.imwrite(name, img.numpy().transpose(1,2,0)[...,::-1])
            img = img_trans * composed_mask + img_origin * (~composed_mask)
            # cv_show(img_trans, 'aa-img-trans.jpg')
            # cv_show(img_origin, 'aa-img-ori.jpg')
            # cv_show(img, 'aa-img.jpg')
            results['image'] = img

        return results


    def apply_img(self, img):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img = img.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img

    # def __repr__(self):
    #     repr_str = self.__class__.__name__
    #     repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
    #     repr_str += 'contrast_range='
    #     repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
    #     repr_str += 'saturation_range='
    #     repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
    #     repr_str += f'hue_delta={self.hue_delta})'
    #     return repr_str
