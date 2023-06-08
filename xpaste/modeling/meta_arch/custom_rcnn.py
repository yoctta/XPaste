# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import json
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes, ROIMasks
import detectron2.utils.comm as comm

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data.detection_utils import convert_image_to_rgb

from torch.cuda.amp import autocast
from ..text.text_encoder import build_text_encoder
from ..utils import load_class_freq, get_fed_loss_inds

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Add image labels
    '''
    @configurable
    def __init__(
        self, 
        with_image_labels = False,
        dataset_loss_weight = [],
        fp16 = False,
        sync_caption_batch = False,
        roi_head_name = '',
        cap_batch_ratio = 4,
        with_caption = False,
        dynamic_classifier = False,
        **kwargs):
        """
        """
        self.with_image_labels = with_image_labels
        self.dataset_loss_weight = dataset_loss_weight
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        self.return_proposal = False
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret.update({
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'dataset_loss_weight': cfg.MODEL.DATASET_LOSS_WEIGHT,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS
        return ret


    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if 'proposals' in  batched_inputs[0]:
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        else :
            proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            results, inds =  self._postprocess(
                results, batched_inputs, images.image_sizes)
            if hasattr(self.roi_heads, 'save_bbox_features') :
                self.roi_heads.save_bbox_features = self.roi_heads.save_bbox_features[inds]
            return results
        else:
            return results


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Add ann_type
        Ignore proposal loss when training with image labels
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        ann_type = 'box'
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.with_image_labels:
            for inst, x in zip(gt_instances, batched_inputs):
                inst._ann_type = x['ann_type']
                inst._pos_category_ids = x['pos_category_ids']
            ann_types = [x['ann_type'] for x in batched_inputs]
            assert len(set(ann_types)) == 1
            ann_type = ann_types[0]
            if ann_type in ['prop', 'proptag']:
                for t in gt_instances:
                    t.gt_classes *= 0
        
        if self.fp16: # TODO (zhouxy): improve
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        cls_features, cls_inds, caption_features = None, None, None

        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()
        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))
        
        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type) # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[
                0].cls_score.zs_weight[:, ind_with_bg].permute(1, 0).contiguous()

        classifier_info = cls_features, cls_inds, caption_features
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)

        if self.roi_head_name in ['StandardROIHeads', 'CascadeROIHeads']:
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances)
        else:
            sem_seg_gt = None
            if "sem_seg" in batched_inputs[0]:
                sem_seg_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
                sem_seg_gt = ImageList.from_tensors(
                    sem_seg_gt, self.backbone.size_divisibility, -1
                ).tensor
            proposals, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances,
                ann_type=ann_type, classifier_info=classifier_info, sem_seg_gt=sem_seg_gt)
        
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        if self.with_image_labels:
            if ann_type in ['box', 'prop', 'proptag']:
                losses.update(proposal_losses)
            else: # ignore proposal loss for non-bbox data
                losses.update({k: v * 0 for k, v in proposal_losses.items()})
        else:
            losses.update(proposal_losses)
        if len(self.dataset_loss_weight) > 0:
            dataset_sources = [x['dataset_source'] for x in batched_inputs]
            assert len(set(dataset_sources)) == 1
            dataset_source = dataset_sources[0]
            for k in losses:
                losses[k] *= self.dataset_loss_weight[dataset_source]
        
        if self.return_proposal:
            return proposals, losses
        else:
            return losses


    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32, 
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
                if has_caption_feature else None # (NB) x (D + 1)
        return caption_features


    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids, 
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C, 
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map

    def _postprocess(self, instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r, inds = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
            # if hasattr(self, 'roi_heads.')
        return processed_results, inds

def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    inds = output_boxes.nonempty()
    results = results[inds]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results, inds
