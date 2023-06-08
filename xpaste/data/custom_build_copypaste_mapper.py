from .transforms.custom_augmentation_impl import EfficientDetResizeCrop
from .transforms.custom_copypaste import CopyPaste
from .transforms.custom_color_jitter import PhotoMetricDistortion
from detectron2.utils.visualizer import Visualizer
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import detectron2.utils.comm as comm
from detectron2.structures import BitMasks, Boxes, Instances
from detic.data.transforms.custom_cp_method import blend_image
import numpy as np
import os
import torch
import json
import copy
import cv2
from collections import defaultdict
from PIL import Image,ImageFile
import albumentations 
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

def pad_to_hw(data, h, w, y_start, x_start):
    M=np.float32([[1,0,x_start],[0,1,y_start]])
    pad=cv2.warpAffine(data,M,(w,h))
    if len(pad.shape)==2:
        pad=pad[:,:,None]
    return pad

def random_start_xy(data_dict,train_size):
    #print(data_dict['gt_masks'].shape,data_dict['image'].shape)
    h,w = data_dict['image'].shape[:2]
    if data_dict['gt_masks'].shape[:2]!=data_dict['image'].shape[:2]:
        data_dict['gt_masks']= np.ascontiguousarray(data_dict['gt_masks'].transpose(1, 2, 0))
    assert data_dict['gt_masks'].shape[:2]==(h,w)
    h_train, w_train = train_size
    x_range = w_train
    y_range = h_train
    x_mid=data_dict['gt_bboxes'][:,0::2].mean()
    y_mid=data_dict['gt_bboxes'][:,1::2].mean()
    x_start, y_start = np.random.randint(-x_mid, x_range-x_mid), np.random.randint(-y_mid, y_range-y_mid)
    return start_xy(data_dict,[x_start, y_start],train_size)

def start_xy(data_dict,bb,train_size):
    h,w = data_dict['image'].shape[:2]
    h_train, w_train = train_size
    x_start, y_start = bb[:2]
    for k in ['image', 'gt_masks']:
        data_dict[k] = np.ascontiguousarray(pad_to_hw(data_dict[k], h_train, w_train, y_start, x_start).transpose(2, 0, 1))
    data_dict['gt_bboxes']=get_bboxes(data_dict['gt_masks'])
    return data_dict

def convert_instance_to_dict(x):
    inst = x['instances']
    return {'image' : x['image'].numpy(), 'file_name':x['file_name'],
        'gt_bboxes': inst.get('gt_boxes').tensor.numpy(), 'gt_labels': inst.get('gt_classes').numpy(), 'gt_masks':inst.get('gt_masks').tensor.numpy()}

def get_updated_masks(masks, composed_mask):
    assert masks.shape[-2:] == composed_mask.shape[-2:], \
        'Cannot compare two arrays of different size {} {}'.format(masks.shape, composed_mask.shape)
    masks = np.where(composed_mask, 0, masks)
    return masks

def get_bboxes(masks):
    num_masks = len(masks)
    boxes = np.zeros((num_masks, 4), dtype=np.float32)
    x_any = masks.any(axis=1)
    y_any = masks.any(axis=2)
    for idx in range(num_masks):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            # use +1 for x_max and y_max so that the right and bottom
            # boundary of instance masks are fully included by the box
            boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                        dtype=np.float32)
    return boxes

class InstPool:
    def __init__(self, json_file, image_root, train_size, max_samples=20, image_format="BGR",use_largest_part=True,random_rotate=False,cp_method=['basic'],color_aug=False) -> None:
        if image_format=='RGBA':
            with open(json_file) as f:
                self.per_cat_pool=json.load(f)
            self.per_cat_pool={int(i):self.per_cat_pool[i] for i in self.per_cat_pool}
            with open('datasets/lvis/area_mean_std2.json') as f:
                self.HWms=json.load(f)
            self.cats= list(self.per_cat_pool.keys())
            self.data_to_cat={}
            self.dataset=[]
            per_cat_pool={}
            for i in self.per_cat_pool:
                per_cat_pool[i]=list(range(len(self.dataset),len(self.dataset)+len(self.per_cat_pool[i])))
                self.dataset+=self.per_cat_pool[i]
                for p in self.per_cat_pool[i]:
                    self.data_to_cat[p]=i
            self.per_cat_pool=per_cat_pool
        else:
            self.dataset = load_coco_json(json_file, image_root)
            self._get_per_cat_pool()
        #augmentations = [EfficientDetResizeCrop(size=-1, scale=(0.1, 2.)), T.RandomFlip()]
        augmentations=[T.RandomFlip()]
        if random_rotate:
            augmentations.append(T.RandomRotation([-30,30]))
        self.augmentations = T.AugmentationList(augmentations)
        self.instance_mask_format = 'bitmask'
        self.bbox_occluded_thr = 10
        self.mask_occluded_thr = 300
        self.scale_min=10
        self.scale_max=0.5
        self.instance_filter_min=0.01
        self.instance_filter_max=1.0
        self.mask_threshold=128
        self.use_largest_part=use_largest_part
        self.shape_jitter=0.2
        self.cp_method=cp_method
        self.train_size = train_size
        self.image_format = image_format
        self.max_samples = max_samples
        self.var_factor=1
        self.cumstom_augmentations=[]
        if color_aug:
            self.cumstom_augmentations.append(albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,p=0.5))
        self.cumstom_augmentations=albumentations.Compose(self.cumstom_augmentations)
        

    def _get_per_cat_pool(self, ):
        self.per_cat_pool = defaultdict(list)
        for i, data in enumerate(self.dataset) :
            assert len(data['annotations']) == 1
            anno = data['annotations'][0]
            self.per_cat_pool[anno['category_id']].append(i)
        self.cats = list(self.per_cat_pool.keys())


    def get_mix_result(self, sample_type='random', cids=None,data_dict=None,reference=None):
        results_origin=data_dict
        data_dict=convert_instance_to_dict(copy.deepcopy(data_dict))
        data_dict['instance_source']=np.zeros_like(data_dict['gt_labels'])
        if reference is not None:
            results=self._cat_a_new_image_with_ref( data_dict,reference)
        else:
            num_sample = np.random.randint(0, self.max_samples)
            if sample_type == 'random':
                ids = self._get_random_samples(num_sample)
            elif sample_type == 'cas_random':
                ids = self._get_cls_balanced_random_samples(num_sample)
            elif sample_type == 'cats_random':
                # assert cids is not None and len(cids)
                if len(cids) ==0:
                    cids = None
                ids = self._get_cls_balanced_random_samples(num_sample, cids=cids)
            else :
                raise NotImplementedError
            results = self._cat_a_new_image(ids, data_dict)
        if results is not None: 
            h, w = results['image'].shape[-2:]
            results_origin['image'] = torch.from_numpy(results['image'])
            results_origin['instances'] = Instances((h,w))
            results_origin['instances'].gt_boxes = Boxes(results['gt_bboxes'])
            results_origin['instances'].gt_classes = torch.tensor(results['gt_labels'], dtype=torch.int64)
            results_origin['instances'].gt_masks = BitMasks(results['gt_masks'])
            results_origin['height'], results_origin['width'] = h, w
        results_origin['instances'].instance_source=torch.tensor(data_dict['instance_source'], dtype=torch.int64)
        return results_origin

    def _get_cls_balanced_random_samples(self, nums, cids = None):
        if cids is None :
            cats_pool = [self.per_cat_pool[x] for x in self.per_cat_pool if len(self.per_cat_pool[x])>0]
        else :
            cats_pool = [self.per_cat_pool[x] for x in cids if len(self.per_cat_pool[x])>0]
        idxs = []
        if len(cats_pool)==0:
            return []
        for _ in range(nums):
            cid = np.random.randint(0, len(cats_pool))
            id_in_cat = np.random.randint(0, len(cats_pool[cid]))
            id = cats_pool[cid][id_in_cat]
            idxs.append(id)
        return idxs

    def _get_random_samples(self, nums):
        idxs = []
        for _ in range(nums):
            idxs.append(np.random.randint(0, len(self.dataset)))
        return idxs

    def _get_sample_from_cids(self, cats, nums):
        '''
        cats : [cid0, cid1, ...]
        nums : [n0, n1, ...]

        '''
        assert len(cats) == len(nums)
        idxs = []
        for cat, n in zip(cats, nums):
            if len(self.per_cat_pool[cat]) == 0:
                continue
            inner_ids = [np.random.randint(0, len(self.per_cat_pool[cat])) for _ in range(n)]
            idxs.extend([self.per_cat_pool[x] for x in inner_ids])
        return idxs

    def _load_RGBA(self,img_path,train_size,target_WH=None):
        if isinstance(train_size, int):
            train_size = (train_size, train_size)
        image_H,image_W=train_size
        label=self.data_to_cat[img_path]
        mask_path=None
        if img_path[0]=='*':
            img_path=img_path[1:]
            img_RGBA=np.array(Image.open(img_path).convert('RGBA'))
            preprocessed=True
        else:
            preprocessed=False
            if '|' in img_path:
                mask_path=img_path.split('|')[1]
                img_path=img_path.split('|')[0]
            try:
                img_RGBA=np.array(Image.open(img_path).convert('RGBA'))
            except:
                print("-----------image is None----------",img_path)
                return None
            if mask_path is not None:
                try:
                    img_RGBA[:,:,-1]=np.array(Image.open(mask_path))
                except:
                    print("-----------mask is None----------",mask_path)
        area_mean,area_std=self.HWms[str(label+1)][:2]
        if type(self.scale_min)==int:
            scale_min=self.scale_min/image_H
        else:
            scale_min=self.scale_min
        if type(self.scale_max)==int:
            scale_max=self.scale_max/image_H
        else:
            scale_max=self.scale_max
        area=np.clip(area_mean+np.random.randn()*area_std,scale_min,scale_max)
        classes = np.array([label], dtype=np.int64)

        if not preprocessed:
            alpha=img_RGBA[...,3:]
            seg_mask=(alpha>self.mask_threshold).astype('uint8')
            if self.use_largest_part:
                seg_mask=get_largest_connect_component(seg_mask)
            seg_mask_ = np.where(seg_mask)
            instance_area=len(seg_mask_[0])
            instance_area_percent=instance_area/(seg_mask.shape[0]*seg_mask.shape[1])
            if instance_area_percent<=self.instance_filter_min or instance_area_percent>=self.instance_filter_max:
                print("---------------mask area---------------",instance_area_percent)
                return None
            y_min,y_max,x_min,x_max = np.min(seg_mask_[0]), np.max(seg_mask_[0]), np.min(seg_mask_[1]), np.max(seg_mask_[1])
            if y_max<=y_min or x_max<=x_min:
                print("---------------mask boundary---------------",y_min,y_max,x_min,x_max)
                return None
            instance_H=y_max+1-y_min
            instance_W=x_max+1-x_min
            img_RGBA[:,:,3:]*=seg_mask
            img_RGBA=img_RGBA[y_min:y_max+1,x_min:x_max+1]
        if target_WH is None:
            target_WH=[area**2*image_H*image_W]
        if len(target_WH)==1:
            scale=target_WH[0]
            ratio=img_RGBA.shape[1]/img_RGBA.shape[0]*np.random.uniform(1-self.shape_jitter,1+self.shape_jitter)
            target_W=np.sqrt(ratio*scale)
            target_H=target_W/ratio
            target_W,target_H=int(target_W),int(target_H)
        else:
            target_W,target_H=target_WH
        if target_W<5 or target_W>=image_W or target_H<5 or target_H>=image_H:
            print("---------------target shape---------------",target_H,target_W,train_size)
            return None
        img_RGBA=cv2.resize(img_RGBA,(target_W,target_H))
        segment=(img_RGBA[:,:,-1]>0).astype(np.uint8)
        #### debug
        #print('target shape:',target_H,target_W)
        aug_input = T.AugInput(img_RGBA, sem_seg=segment,boxes=np.array([[0,0,target_W,target_H]]))
        transforms = self.augmentations(aug_input)
        img_RGBA=aug_input.image
        segment=aug_input.sem_seg
        gt_bboxes=aug_input.boxes
        res={'image' : img_RGBA, 'file_name':img_path,'gt_bboxes': gt_bboxes, 'gt_labels': classes, 'gt_masks':np.expand_dims(segment,-1)}
        return res

    def _load_dict(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        # TODO
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # dataset_dict["image"] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)).copy())
        dataset_dict['image'] = image
        if "annotations" in dataset_dict:

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    def _cat_a_new_image(self, ids, data_dict=None):
        train_size=data_dict['image'].shape[-2:]
        if self.image_format=='RGBA':
            datas=[self._load_RGBA(self.dataset[x],train_size) for x in ids]
        else:
            datas = [convert_instance_to_dict(self._load_dict(self.dataset[x])) for x in ids]
        datas=[x for x in  datas if x is not None]
        for x in datas:
            x['image'][:,:,:3]=self.cumstom_augmentations(image=x['image'][:,:,:3])['image']
        datas = [random_start_xy(x,train_size) for x in datas]
        datas = [x for x in datas if x is not None]
        if len(datas) == 0 :
            return None
        dst_results = data_dict
        #dst_results['origin_image']=dst_results['image']
        for x in datas :
            dst_results = self._copy_paste(dst_results, x)
        ##### debug
        #print(ids)
        return dst_results


    def _copy_paste(self, dst_results, src_results, ret_valid_idx=False):
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['image']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_source = dst_results['instance_source']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['image']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks, axis=0), 1, 0)
        updated_dst_masks = get_updated_masks(dst_masks, composed_mask)
        # updated_dst_bboxes = updated_dst_masks.get_bboxes()
        updated_dst_bboxes = get_bboxes(updated_dst_masks)
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = blend_image(dst_img,src_img,composed_mask,self.cp_method).astype(dst_img.dtype)
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        dst_source = np.concatenate([dst_source[valid_inds], [1]])
        masks = np.concatenate(
            [updated_dst_masks[valid_inds], src_masks])

        dst_results['image'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        dst_results['gt_masks'] = masks
        dst_results['instance_source'] = dst_source
        # dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                            #   masks.shape[2])

        if ret_valid_idx :
            return dst_results, valid_inds
        return dst_results

    def _cat_a_new_image_with_ref(self, data_dict=None,reference=None):
        train_size=data_dict['image'].shape[-2:]
        h_ref, w_ref = reference['image'].shape[1:]
        reference=reference['instances']
        reference_labels=reference.get('gt_classes').numpy()
        reference_bboxes=reference.get('gt_boxes').tensor.numpy()
        if len(reference_bboxes)==0: return None
        h_train, w_train = train_size
        scale = min(h_train/h_ref, w_train/w_ref)
        scale = np.random.uniform(0.8*scale, 1.0*scale)
        reference_bboxes=(scale*reference_bboxes).astype(np.int32)
        x_max=np.max(reference_bboxes[:,2])
        y_max=np.max(reference_bboxes[:,3])
        x_offset=np.random.randint(0,w_train-x_max+1)
        y_offset=np.random.randint(0,h_train-y_max+1)
        reference_bboxes[:,0::2] += x_offset
        reference_bboxes[:,1::2] += y_offset
        datas=[self._load_RGBA(self.dataset[x],train_size,target_WH=[(bb[3]-bb[1])*(bb[2]-bb[0])]) for x,bb in zip(reference_labels,reference_bboxes)]
        datas = [start_xy(x,bb,train_size) for x,bb in zip(datas,reference_bboxes) if x is not None]
        datas = [x for x in datas if x is not None]
        if len(datas) == 0 :
            return None
        dst_results =data_dict
        for x in datas:
            dst_results = self._copy_paste(dst_results, x)

        return dst_results

class InstaBoost:
    def __init__(self,
                 action_candidate=('normal', 'horizontal', 'skip'),
                 action_prob=(1, 0, 0),
                 scale=(0.8, 1.2),
                 dx=15,
                 dy=15,
                 theta=(-1, 1),
                 color_prob=0.5,
                 hflag=False,
                 aug_ratio=0.5,
                 cid_to_freq = None,
                 apply_freq = ['r', 'c', 'f']):
        try:
            import instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')
        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio
        self.cid_to_freq = cid_to_freq
        self.apply_freq = apply_freq
        self.count  = 0

    def __call__(self, dataset_dict):
        # img = results['image'].numpy()
        # ori_type = img.dtype

        anns = copy.deepcopy(dataset_dict['annotations'])
        boost_anns = []
        nboost_anns = []
        for ann in anns :
            if self.cid_to_freq[ann['category_id']] in self.apply_freq :
                boost_anns.append(ann)
            else :
                nboost_anns.append(ann)
        img = cv2.imread(dataset_dict['file_name'])

        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]) and len(boost_anns):
            try:
                import instaboostfast as instaboost
            except ImportError:
                raise ImportError('Please run "pip install instaboostfast" '
                                  'to install instaboostfast first.')
            try :
                # cv2.imwrite('inst_show/{}_ori.jpg'.format(self.count), img)
                boost_anns, img = instaboost.get_new_data(boost_anns, img, self.cfg, background=None)
                # cv2.imwrite('inst_show/{}_post.jpg'.format(self.count), img)
                # self.count +=1
                anns = boost_anns + nboost_anns
                new_anns = []
                for ann in anns :
                    x1, y1, w, h = ann['bbox']
                    if w <= 0 or h <= 0:
                        continue
                    new_anns.append(ann)
                anns = new_anns
                assert all(['segmentation' in ann for ann in anns]), 'instaboost error'
                dataset_dict['annotations'] = anns
                dataset_dict['image_new'] = img[...,::-1]
            except :
                print('failed at instaboost')
        return dataset_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cfg={self.cfg}, aug_ratio={self.aug_ratio})'
        return repr_str


class CopyPasteMapper:
    def __init__(self, mapper, cfg, dataset=None):
        self.mapper = mapper
        if dataset is None :
            repeat_probs = None
        else :    
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset, 0.001)
            repeat_probs = repeat_factors / repeat_factors.sum(-1, keepdim=True)

        self.dataset = dataset
        self.rfs_version = cfg.INPUT.RFS_VERSION
        self.vis_result = cfg.INPUT.VIS_RESULT
        self.use_scp = cfg.INPUT.USE_SCP
        self.remove_bg_prob = cfg.INPUT.RM_BG_PROB
        self.num_scr_image = cfg.INPUT.SCP_NUM_SRC
        self.rfs_choice = cfg.INPUT.SCP_RFS
        self.raw_dataset = None
        self.scp_select_cls = cfg.INPUT.SCP_SELECT_CATS_LIST
        self.limit_src_lsj = cfg.INPUT.LIMIT_SRC_LSJ
        self.use_copy_method= cfg.INPUT.USE_COPY_METHOD
        with open(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH) as f :
            freq_dict = json.load(f)
            self.cid_to_freq = dict()
            # convert 1-ind to 0-ind
            for x in freq_dict :
                self.cid_to_freq[x['id']-1] = x['frequency']

        self.instaboost_src = False
        self.instaboost_dst = False
        if cfg.INPUT.USE_INSTABOOST :
            self.instaboost_mapper = InstaBoost(cid_to_freq=self.cid_to_freq, apply_freq=cfg.INPUT.INSTABOOST_FREQ)
            insta_apply_type = cfg.INPUT.INSTABOOST_APPLY_TYPE
            assert insta_apply_type in ['both', 'src', 'dst']
            if insta_apply_type in ['both', 'src'] :
                self.instaboost_src = True
            if insta_apply_type in ['both', 'dst'] :
                self.instaboost_dst = True

        if cfg.INPUT.USE_COLOR_JITTER :
            self.color_aug = PhotoMetricDistortion(self.cid_to_freq, freq_color_filter=cfg.INPUT.COLOR_JITTER_FREQ_FILTER,
                                    use_torchvision=cfg.INPUT.COLOR_JITTER_USE_TORCHVISION)
            apply_type = cfg.INPUT.COLOR_JITTER_APPLY_TYPE
            if apply_type == 'both':
                self.src_color = True
                self.dst_color = True
            elif apply_type == 'src':
                self.src_color = True
                self.dst_color = False
            elif apply_type == 'dst':
                self.src_color = False
                self.dst_color = True
            else :
                raise NotImplementedError
        else :
            self.color_aug = None
            self.src_color = False
            self.dst_color = False
        if cfg.INPUT.INST_POOL :
            self.inst_pool = InstPool(cfg.INPUT.INST_POOL_PATH, cfg.INPUT.INST_POOL_ROOT, cfg.INPUT.TRAIN_SIZE, image_format=cfg.INPUT.INST_POOL_FORMAT, max_samples=cfg.INPUT.INST_POOL_MAX_SAMPLES,
                                      use_largest_part=cfg.USE_LARGEST_PART,random_rotate=cfg.INPUT.RANDOM_ROTATE,cp_method=cfg.INPUT.CP_METHOD,color_aug=self.src_color)
            self.inst_pool_sample_type = cfg.INPUT.INST_POOL_SAMPLE_TYPE
        else :
            self.inst_pool = None
        self.scp_type = cfg.INPUT.SCP_TYPE
        if self.scp_type == 'rc_only':
            self.freq_select = ['r', 'c']
        elif self.scp_type == 'f_only':
            self.freq_select = ['f']
        elif self.scp_type in ('in_domain', 'cas', 'the_cls', 'the_cls_img'):
            self.freq_select = None
        elif self.scp_type != '':
            raise NotImplementedError

        cid_filter = []
        for k, v in self.cid_to_freq.items():
            if v == 'r':
                cid_filter.append(k)
        cid_filter = set(cid_filter)
        self.scp_aug = CopyPaste(repeat_probs=repeat_probs, 
                                    selected=cfg.INPUT.SCP_SRC_OBJ_SELECT and self.scp_type not in ('in_domain', 'cas'),
                                    blank_ratio=cfg.INPUT.BLANK_RATIO,
                                    rotate_ang=cfg.INPUT.INP_ROTATE_ANG,
                                    cid_filter=cid_filter,
                                    limit_inp_trans=cfg.INPUT.INP_ROTATE_LIMIT,
                                    rotate_src = cfg.INPUT.ROTATE_SRC,cp_method=['basic'])
        # self.logger = open(os.path.join(cfg.OUTPUT_DIR, 'scp_log_{}.json'.format(comm.get_rank())), 'a+')
        if cfg.INPUT.LOG_SCP_PARAM :
            self.logger = os.path.join(cfg.OUTPUT_DIR, 'scp_log_{}.json'.format(comm.get_rank()))
            self.img_save_dir = os.path.join(cfg.OUTPUT_DIR, 'train_imgs')
            if not os.path.exists(self.img_save_dir):
                os.makedirs(self.img_save_dir,exist_ok=True)
        else :
            self.logger = None
            self.img_save_dir = None
        self.counter = 0
    
    # def _filter_in_specific_cls(self, dataset_dict, num_src=3, cas=False):
    def _filter_in_specific_cls(self, dataset_dict, num_src=3, cas=False, specific_cls=False, filter_cls_inst=True):
        if specific_cls :
            cls_list = np.random.choice(self.scp_select_cls, num_src, replace=False)
            cid_pool_list = [self.per_cat_map[x] for x in cls_list]
        elif cas :
            cls_list = np.random.choice(list(self.per_cat_map.keys()), num_src, replace=False)
            cid_pool_list = [self.per_cat_map[x] for x in cls_list]
        else :
            cls_list = dataset_dict['instances'].gt_classes
            cls_list = list(set(cls_list.tolist()))
            cats_pool = [self.per_cat_map[x] for x in cls_list]
            if len(cats_pool) == 0:
                return []
            cid_pool_list = []
            for _ in range(num_src):
                cid = np.random.randint(0, len(cats_pool))
                cid_pool_list.append(cats_pool[cid])

        mix_results = []
        # for _ in range(num_src):
        #     cid = np.random.randint(0, len(cats_pool))
        for cid_pool in cid_pool_list :
            id_in_cat = np.random.randint(0, len(cid_pool))
            idx = cid_pool[id_in_cat]
            src_dataset_dict = copy.deepcopy(self.dataset[idx])
            if filter_cls_inst :
                cls_filter = [(x['category_id'] in cls_list) for x in src_dataset_dict['annotations']]
                src_dataset_dict['annotations'] = [x for i, x in enumerate(src_dataset_dict['annotations']) if cls_filter[i]]
            src_dataset_dict = self.mapper(src_dataset_dict)

            mix_results.append(src_dataset_dict)

        return mix_results

    def set_dataset(self, dataset):
        rfs_choice = self.rfs_choice
        self.dataset = dataset
        if self.scp_type in ['rc_only', 'f_only']:
            self.dataset = []
            for i, data in enumerate(dataset):
                data = copy.deepcopy(data)
                anno = data['annotations']
                anno = [x for x in anno if self.cid_to_freq[x['category_id']] in self.freq_select]
                data['annotations'] = anno
                if len(anno):
                    self.dataset.append(data)
        if self.scp_type in ('in_domain', 'cas', 'the_cls', 'the_cls_img'):
            self.per_cat_map = defaultdict(list)
            for i, data in enumerate(dataset):
                cat_ids = set([x['category_id'] for x in data['annotations']])
                # for anno in data['annotations']:
                for cid in cat_ids :
                    self.per_cat_map[cid].append(i) 
        if rfs_choice :
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(dataset, 0.001)
            if self.rfs_version == 0 :
                repeat_probs = repeat_factors / repeat_factors.sum(-1, keepdim=True)
                self.scp_aug.repeat_probs = repeat_probs.numpy()
            elif self.rfs_version == 1:
                if self.raw_dataset is None :
                    self.raw_dataset = self.dataset
                _int_part = torch.trunc(repeat_factors)
                _frac_part = repeat_factors - _int_part
                rands = torch.rand(len(_frac_part))
                rep_factors = _int_part + (rands < _frac_part).float()
                indices = []
                for dataset_index, rep_factor in enumerate(rep_factors):
                    indices.extend([dataset_index] * int(rep_factor.item()))
                self.dataset = [self.raw_dataset[x] for x in indices]
                self.scp_aug.repeat_probs = None
            else :
                raise NotImplementedError

    def __call__(self, dataset_dict, cp_idx=None):
        assert self.dataset is not None , 'dataset cant be None in CopyPasteMapper'
        if self.instaboost_dst :
            dataset_dict = self.instaboost_mapper(dataset_dict)

        result = self.mapper(dataset_dict)
        if not 'instances' in result or not result['instances'].has('gt_masks'):
            print('no instance found ',result['file_name'])
            return result
        
        if self.color_aug is not None and self.dst_color :
            result = self.color_aug(result)

        if self.remove_bg_prob > 0 :
            assert self.remove_bg_prob <= 1
            if np.random.uniform(0.0, 1.0) <= self.remove_bg_prob :
                result = self.scp_aug.remove_background(result)
        if self.use_scp :
            if cp_idx is not None :
                idxs = [cp_idx]
            else :
                idxs = [self.scp_aug.get_indexes(self.dataset) for _ in range(self.num_scr_image)]
                # idx = self.scp_aug.get_indexes(self.dataset)
                self.counter += self.num_scr_image
                if self.counter > len(self.dataset) and self.rfs_version == 1:
                    self.counter = 0
                    self.set_dataset(self.raw_dataset, rfs_choice=self.rfs_choice)
            mix_results = []
            if self.use_copy_method=='both':
                copy_paste_method=['self_copy','syn_copy']
            elif self.use_copy_method in ['self_copy','syn_copy']:
                copy_paste_method=self.use_copy_method
            elif self.use_copy_method.startswith('p:'):
                copy_paste_method='self_copy' if np.random.rand()<float(self.use_copy_method[2:]) else 'syn_copy'
            else: copy_paste_method=[]
            if 'self_copy' in copy_paste_method :
                if self.scp_type == '' :
                    for idx in idxs :
                        src_dataset_dict = self.dataset[idx]
                        if self.instaboost_src :
                            src_dataset_dict = self.instaboost_mapper(src_dataset_dict)
                        if self.limit_src_lsj :
                            h_dst, w_dst = result['image'].shape[-2:]
                            h_src, w_src = src_dataset_dict['height'], src_dataset_dict['width']
                            scale = min(h_dst/h_src, w_dst/w_src)
                            scale = (0.8*scale, 1.2*scale)
                            augmentations = [EfficientDetResizeCrop(size=-1, scale=scale), T.RandomFlip()]
                        else :
                            augmentations = None
                        src_dataset_dict = self.mapper(src_dataset_dict, augmentations=augmentations)
                        mix_results.append(src_dataset_dict)
                elif self.scp_type == 'in_domain':
                    mix_results = self._filter_in_specific_cls(result, self.num_scr_image)
                elif self.scp_type == 'cas':
                    mix_results = self._filter_in_specific_cls(result, self.num_scr_image, cas=True)
                elif self.scp_type == 'the_cls':
                    mix_results = self._filter_in_specific_cls(result, self.num_scr_image, specific_cls=True)
                elif self.scp_type == 'the_cls_img':
                    mix_results = self._filter_in_specific_cls(result, self.num_scr_image, specific_cls=True, filter_cls_inst=False)
                else :
                    raise NotImplementedError
            if self.inst_pool is not None and 'syn_copy' in copy_paste_method: ##have inst pool
                cids = None
                src_dataset_dict=None
                if self.inst_pool_sample_type == 'cats_random':
                    cids = result['instances'].gt_classes.tolist()
                    cids = list(set(cids))
                if self.inst_pool_sample_type == 'reference':
                    src_dataset_dict =self.mapper( self.dataset[self.scp_aug.get_indexes(self.dataset)], augmentations=None)
                    #print(src_dataset_dict['image'].shape)
                    #print(src_dataset_dict['instances'])
                if self.vis_result:
                    origin_instances=len(result['instances'].gt_classes)
                result=self.inst_pool.get_mix_result(self.inst_pool_sample_type, cids,data_dict=result,reference=src_dataset_dict)
                # if mix_res is not None:
                #     mix_results.append(mix_res)
            if self.color_aug is not None and self.src_color :
                mix_results = [self.color_aug(x) for x in mix_results]
            result['mix_results'] = mix_results
            # self.scp_aug.remove_background(result)
            result = self.scp_aug(result, self.logger, self.img_save_dir)

        if self.vis_result :
            img = result['image']
            inst_pred = result['instances']
            inst_pred.pred_boxes = inst_pred.gt_boxes
            inst_pred.pred_classes = inst_pred.gt_classes
            inst_pred.pred_masks = inst_pred.gt_masks
            visualizer = Visualizer(img.permute(1,2,0), metadata=None)
            img_id=result['file_name'].split('/')[-1][:-4]
            #os.system('cp {} show_train/'.format(result['file_name']))
            visualizer.output.save(os.path.join(self.img_save_dir,'%s_origin.jpg'%img_id))
            vis = visualizer.overlay_instances(
                #boxes=inst_pred.gt_boxes,
                labels=inst_pred.gt_classes.tolist(),
                #assigned_colors=[(0,0,0)]*origin_instances+[(0,0,1)]*(len(result['instances'].gt_classes)-origin_instances))
                masks=inst_pred.gt_masks,)
            vis.save(os.path.join(self.img_save_dir,'%s_anno.jpg'%img_id))
            # import pdb
            # pdb.set_trace()
            # import cv2
            # cv2.imwrite('testcp.jpg', result['image'].numpy().transpose(1,2,0))
        return result
