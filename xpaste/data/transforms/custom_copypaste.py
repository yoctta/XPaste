import torch
import numpy as np
from numpy import random
import copy
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation_impl import RandomRotation
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, BoxMode
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import detectron2.utils.comm as comm
from xpaste.data.transforms.custom_cp_method import blend_image
import math
import json
import cv2
import os

def convert_instance_to_dict(x):
    if 'instances' not in  x :
        return x
    inst = x['instances']
    try :
        result = {'img' : x['image'].numpy(), 'file_name':x['file_name'],
            'gt_bboxes': inst.get('gt_boxes').tensor.numpy(), 'gt_labels': inst.get('gt_classes').numpy(), 'gt_masks':inst.get('gt_masks').tensor.numpy()}
        return result
    except :
        print('error in convert')
        return None

class CopyPaste:
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:
    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.
    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """

    def __init__(
        self,
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
        dataset = None,
        repeat_probs = None,
        blank_ratio = -1,
        rotate_ang = 30,
        cid_filter = [],
        limit_inp_trans = False,
        rotate_src = False,
        cp_method='basic'
    ):
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected
        self.dataset = dataset
        self.repeat_probs = repeat_probs
        self.blank_ratio = blank_ratio
        self.cid_filter = cid_filter
        self.cp_method= cp_method
        self.rotate_aug = RandomRotation([-rotate_ang,rotate_ang])
        self.count = 0
        self.limit_inp_trans = limit_inp_trans
        self.rotate_src = rotate_src

    def get_indexes(self, dataset):
        """Call function to collect indexes.s.
        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        if self.repeat_probs is not None :
            assert len(self.repeat_probs) == len(dataset)
            # return random.choices(list(range(dataset)), weight=self.repeat_probs)
            return random.choice(list(range(len(dataset))), p=self.repeat_probs)
        return random.randint(0, len(dataset))

    def remove_background(self, results):
        img = results['image']
        gt_masks = results['instances'].gt_masks.tensor
        # np.where(np.any(src_masks, axis=0), 1, 0)
        # compose_mask = torch.where(torch.any(gt_masks, dim=0), 1, 0, dtype=img.dtype)
        compose_mask = torch.any(gt_masks, dim=0).type(img.dtype)
        img_fg = img * compose_mask[None]
        results['image'] = img_fg
        return results

    def _inp_rotate(self, results, inp=True):
        # img_inp = results['inp_image'].numpy().transpose(1,2,0) # np (h,w,3)
        # img_inp = img_inp.numpy().transpose(1,2,0)
        image = results['image'].numpy().transpose(1,2,0)

        inst = results['instances']
        if len(inst) == 0 :
            return results
        gt_cls = inst.get('gt_classes').tolist()
        cls_filter = torch.tensor([(x in self.cid_filter) for x in gt_cls])
        inst_filter = inst[cls_filter]
        if len(inst_filter) == 0 :
            return results
        masks = inst_filter.get('gt_masks').tensor.numpy()
        masks_origin = masks
        bboxes = self.get_bboxes(masks).astype(int)

        def crop_from_img(bboxes, img, masks):
            img_list = []
            mask_list = []
            for i, bbox in enumerate(bboxes) :
                img_list.append(img[bbox[1]:bbox[3],bbox[0]:bbox[2]])
                mask_list.append(masks[i][bbox[1]:bbox[3],bbox[0]:bbox[2]])
            return img_list, mask_list

        def create_canvas(img, bboxes):
            h, w = img.shape[:2]
            max_h = bboxes[...,1::2].max()
            max_w = bboxes[...,0::2].max()
            h, w = max(h, max_h), max(w, max_w)
            canvas = np.zeros((h, w, 3), dtype=img.dtype)
            return canvas

        def copy_on_img(dst_bbox, img):
            dst_bbox_center = (dst_bbox[:2] + dst_bbox[2:]) //2
            src_bbox = np.array([0,0,img.shape[1],img.shape[0]])
            src_bbox_center = (src_bbox[:2] + src_bbox[2:]) //2
            shift = dst_bbox_center - src_bbox_center
            src_bbox_new = src_bbox + np.concatenate([shift, shift]) #shift
            # h, w = dst_img.shape[:2]
            # dst_img[src_bbox_new[1]:src_bbox_new[3], src_bbox_new[0]:src_bbox_new[2]][mask] = img[mask]
            return src_bbox_new

        
        img_list, mask_list = crop_from_img(bboxes, image, masks)
        # image_cp = img_inp.copy()
        mask_new_list = []
        bbox_new_list = []
        img_r_list = []
        mask_r_list = []
        for bbox, mask, img in zip(bboxes, mask_list, img_list):

            aug_input = T.AugInput(img, sem_seg=None)
            transform = self.rotate_aug(aug_input)
            img_r = aug_input.image
            mask_r = transform.apply_image(mask.astype(np.uint8))
            mask_r = mask_r.astype(bool)

            bbox_new = copy_on_img(bbox, img_r)
            bbox_new_list.append(bbox_new)
            img_r_list.append(img_r)
            mask_r_list.append(mask_r)
            # mask_new_list.append(mask_new)

        canvas = create_canvas(image, np.stack(bbox_new_list))
        h_c, w_c = canvas.shape[:2]
        for bbox_new, mask_r, img_r in zip(bbox_new_list, mask_r_list, img_r_list):
            bbox_new = bbox_new.clip(min=0)
            w, h = bbox_new[-2:] - bbox_new[:2]
            mask_r = mask_r[-h:,-w:]
            img_r = img_r[-h:, -w:]
            canvas[bbox_new[1]:bbox_new[3], bbox_new[0]:bbox_new[2]][mask_r] = img_r[mask_r]
            mask_new = np.zeros((h_c,w_c), dtype=bool)
            mask_new[bbox_new[1]:bbox_new[3], bbox_new[0]:bbox_new[2]] = mask_r
            mask_new_list.append(mask_new)

        if self.limit_inp_trans:
            h_o, w_o = image.shape[:2]
            canvas = canvas[:h_o, :w_o]
            for i, mask in enumerate(mask_new_list):
                mask_new_list[i] = mask[:h_o, :w_o] & masks_origin[i][:h_o, :w_o]

        # bboxes_new = np.stack(bbox_new_list)
        mask_new = np.stack(mask_new_list)
        bboxes_new = self.get_bboxes(mask_new)

        results_origin = copy.deepcopy(results)
        file_name = results['file_name']
        results_origin['instances'] = results_origin['instances'][~cls_filter]
        if inp and 'inp_image' in results:
            results_origin['image'] = results.pop('inp_image')
        else :
            results_origin['image'] = results['image']

        origin_dict = convert_instance_to_dict(results_origin)
        # import cv2
        # cv2.imwrite('inp_show/{}_inp-origin.jpg'.format(self.count), results['image'].numpy().transpose(1,2,0)[...,::-1])
        # cv2.imwrite('inp_show/{}_inp-src.jpg'.format(self.count), origin_dict['img'].transpose(1,2,0)[...,::-1])
        paste_dict = {'img' : canvas.transpose(2,0,1), 'gt_bboxes': bboxes_new, 'gt_labels': inst_filter.get('gt_classes').numpy(), 'gt_masks':mask_new}
        results, valid_idx, scale = self._scp_src_to_dst(origin_dict, paste_dict, True)
        # cv2.imwrite('inp_show/{}_inp-rotate_raw.jpg'.format(self.count), paste_dict['img'].transpose(1,2,0)[...,::-1])
        # cv2.imwrite('inp_show/{}_inp-rotate.jpg'.format(self.count), results['img'].transpose(1,2,0)[...,::-1])
        # cv2.imwrite('inp_show/{}_np-rotate_mask.jpg'.format(self.count), mask_new.max(axis=0).astype(np.uint8)[...,None]* 255)
        results_origin = {}
        h, w = results['img'].shape[-2:]
        results_origin['file_name'] = file_name
        results_origin['image'] = torch.from_numpy(results['img'])
        results_origin['instances'] = Instances((h,w))
        results_origin['instances'].gt_boxes = Boxes(results['gt_bboxes'])
        results_origin['instances'].gt_classes = torch.tensor(results['gt_labels'], dtype=torch.int64)
        results_origin['instances'].gt_masks = BitMasks(results['gt_masks'])
        results_origin['height'], results_origin['width'] = h, w

        if 0 :
            result = results_origin
            from detectron2.utils.visualizer import Visualizer
            img = result['image']
            inst_pred = result['instances']
            inst_pred.pred_boxes = inst_pred.gt_boxes
            inst_pred.pred_classes = inst_pred.gt_classes
            inst_pred.pred_masks = inst_pred.gt_masks
            visualizer = Visualizer(img.permute(1,2,0), metadata=None)
            vis = visualizer.overlay_instances(
                boxes=inst_pred.gt_boxes,
                labels=inst_pred.gt_classes.tolist(),
                masks=inst_pred.gt_masks,)
            vis.save('inp_show/{}_show.jpg'.format(self.count))
            self.count += 1 

        return results_origin

    def __call__(self, results, logger=None, save_img_dir=None):
        """Call function to make a copy-paste of image.
        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        if 'inp_image' in results :
            if np.random.randint(0, 3) :
                return self._inp_rotate(results)
        results_origin = copy.deepcopy(results)
        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        # when mix results is empty, jump scp
        if num_images == 0 :
            results.pop('mix_results')
            return results
        # assert num_images == 1, \
        #     f'CopyPaste only supports processing 2 images, got {num_images}'
        
        def update_log_dict(x):
            return {'file_name':x['file_name'], 'labels':x['gt_labels'].tolist(), 'boxes':x['gt_bboxes'].tolist()}

        results = convert_instance_to_dict(results)
        if results is None :
            return results_origin
        if logger is not None :
            scp_log_dict = dict()
            scp_log_dict['dst'] = update_log_dict(results)
            scp_log_dict['src'] = []
            # scp_log_dict.update(src_image=)
        src_results = None
        for i in range(num_images) :
            mix_results = results_origin['mix_results'][i]
            if self.rotate_src and np.random.randint(0, 3) :
                mix_results = self._inp_rotate(mix_results, False)
            selected_results = convert_instance_to_dict(mix_results)
            if selected_results is None :
                continue
            if self.selected:
                selected_results = self._select_object(selected_results)

            if len(selected_results['gt_bboxes']) == 0 :
                continue
            if logger is not None :
                scp_log_dict['src'].append(update_log_dict(selected_results))

            if src_results is None :
                src_results = selected_results
                continue

            src_results = self._scp_src_to_dst(src_results, selected_results, is_tmp_dst=True)

        if src_results is not None :
            results, valid_idx, scale = self._scp_src_to_dst(results, src_results, True)
            if logger is not None :
                scp_log_dict['dst']['valid_obj'] = valid_idx.tolist()
                scp_log_dict['scale'] = scale
        if logger is not None :
            with open(logger, 'a+')  as f:
                f.write(json.dumps(scp_log_dict) + '\n')
        if save_img_dir is not None :
            image_save = results['img']
            save_name = '{}_{}_{}'.format(comm.get_rank(), results_origin['image_id'], results_origin['mix_results'][0]['image_id'])
            cv2.imwrite(os.path.join(save_img_dir, '{}.jpg'.format(save_name)), image_save.transpose(1,2,0)[...,::-1])
            with open(os.path.join(save_img_dir, '{}.json'.format(save_name)), 'w') as f:
                f.write(json.dumps(scp_log_dict))
            new_image_id = results_origin['image_id'] * 100000000000 + results_origin['mix_results'][0]['image_id']
        h, w = results['img'].shape[-2:]
        results_origin['image'] = torch.from_numpy(results['img'])
        results_origin.pop('mix_results')
        results_origin['instances'] = Instances((h,w))
        results_origin['instances'].gt_boxes = Boxes(results['gt_bboxes'])
        results_origin['instances'].gt_classes = torch.tensor(results['gt_labels'], dtype=torch.int64)
        results_origin['instances'].gt_masks = BitMasks(results['gt_masks'])
        results_origin['height'], results_origin['width'] = h, w
        if save_img_dir is not None :

            insta_save = copy.deepcopy(results_origin['instances'])
            insta_save.gt_classes +=1 # convert 0-index to 1-index

            boxes = insta_save.gt_boxes.tensor.numpy()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            boxes = boxes.tolist()
            classes = insta_save.gt_classes.tolist()
            results = []
            for k in range(len(insta_save)):
                result = {
                    "image_id": new_image_id,
                    "category_id": classes[k],
                    "bbox": boxes[k],
                    "file_name": '{}.jpg'.format(save_name)
                }
                results.append(result)
            # results = dict(annotations=results``)
            with open(os.path.join(save_img_dir, '{}_gt.json'.format(save_name)), 'w') as f:
                json.dump(results, f)

        return results_origin

    def _scp_src_to_dst(self, dst_results, src_results, ret_valid_idx=False, is_tmp_dst=False):
        if is_tmp_dst :
            h1, w1 = dst_results['gt_bboxes'][...,3].max(), dst_results['gt_bboxes'][...,2].max()
            h1, w1 = math.ceil(h1), math.ceil(w1)
        else :
            h1, w1 = dst_results['img'].shape[-2:]
        # TODO : whether thrunk the hw
        # h2, w2 = selected_results['img'].shape[-2:]
        h2, w2 = src_results['gt_bboxes'][...,3].max(), src_results['gt_bboxes'][...,2].max()
        h2, w2 = math.ceil(h2), math.ceil(w2)
        h, w = max(h1,h2), max(w1,w2)

        scale = 1
        if not is_tmp_dst and self.blank_ratio > 0 :
            composed_mask = np.where(np.any(src_results['gt_masks'], axis=0), 1, 0)
            ratio = (h2 * w2 - composed_mask.sum() - h1 * w1) / (h * w)
            if ratio > self.blank_ratio :
                h2_new = np.random.randint(int(0.5*h1), int(1.1*h1))
                w2_new = np.random.randint(int(0.5*w1), int(1.1*w1))

                scale = min(h2_new / h2, w2_new / w2)
                new_hw = [int(x*scale) for x in src_results['img'].shape[-2:]]
                def resize(x, size):
                    wh_size = size[::-1]
                    result =  cv2.resize(x.transpose(1,2,0), wh_size)
                    if len(result.shape) == 2:
                        result = result[...,None]
                    return result.transpose(2,0,1)
                src_results['img'] = resize(src_results['img'], new_hw)
                if len(src_results['gt_masks']) :
                    src_results['gt_masks'] = resize(src_results['gt_masks'].astype(src_results['img'].dtype), new_hw).astype(bool)
                src_results['gt_bboxes'] = src_results['gt_bboxes'] * scale
                h2, w2 = h2_new, w2_new
                h, w = max(h1,h2), max(w1,w2)
        def pad_to_hw(data, h, w):
            new_data = np.zeros((data.shape[0], h, w), dtype=data.dtype)
            d_h, d_w = min(h, data.shape[1]), min(w, data.shape[2])
            new_data[:,:d_h,:d_w] = data[:,:d_h,:d_w]
            # new_data[:,:data.shape[1],:data.shape[2]] = data
            return new_data
        for k in ['img', 'gt_masks']:
            dst_results[k] = pad_to_hw(dst_results[k], h, w)
            src_results[k] = pad_to_hw(src_results[k], h, w)
        if ret_valid_idx :
            results, valid_idx = self._copy_paste(dst_results, src_results, ret_valid_idx)
            return results, valid_idx, scale
        return self._copy_paste(dst_results, src_results, ret_valid_idx)
        # results, valid_idx = self._copy_paste(dst_results, src_results)
        # return results

    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        # print('num paste', max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        return results
    
    def get_bboxes(self, masks):
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

    def _copy_paste(self, dst_results, src_results, ret_valid_idx=False):
        """CopyPaste transform function.
        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        # print('src shape', src_bboxes.shape)
        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        # updated_dst_bboxes = updated_dst_masks.get_bboxes()
        updated_dst_bboxes = self.get_bboxes(updated_dst_masks)
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
        # elif self.cp_method=='basic':
        #     img = (dst_img * (1 - composed_mask
        #                     ) + src_img * composed_mask).astype(dst_img.dtype)
        # elif self.cp_method=='possion':
        #     src_img=src_img.transpose(1,2,0)
        #     dst_img=dst_img.transpose(1,2,0)
        #     img=poisson_edit(src_img,dst_img,composed_mask)
        #     img_base=(dst_img * (1 - composed_mask[:,:,None]) + src_img * composed_mask[:,:,None]).astype(dst_img.dtype)
        #     import os
        #     img_id=np.random.random_integers(0,999999)
        #     os.makedirs('visualizations',exist_ok=True)
        #     cv2.imwrite('visualizations/%06d_origin.png'%img_id,dst_img[:,:,::-1])
        #     cv2.imwrite('visualizations/%06d_paste.png'%img_id,img_base[:,:,::-1])
        #     cv2.imwrite('visualizations/%06d_possion.png'%img_id,img[:,:,::-1])
        #     #img=cv2.seamlessClone(src_img.astype('uint8'), dst_img.astype('uint8') ,composed_mask.astype('uint8'), (src_img.shape[0]//2,src_img.shape[1]//2), cv2.MIXED_CLONE)
        #     src_img=src_img.transpose(2,0,1)
        #     dst_img=dst_img.transpose(2,0,1)
        #     img=img.transpose(2,0,1)
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks[valid_inds], src_masks])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        dst_results['gt_masks'] = masks
        # dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                            #   masks.shape[2])

        if ret_valid_idx :
            return dst_results, valid_inds
        return dst_results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size {} {}'.format(masks.shape, composed_mask.shape)
        masks = np.where(composed_mask, 0, masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}, '
        return repr_str
