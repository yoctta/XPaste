from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.config import configurable

from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, mask_rcnn_inference
from torchvision.ops import roi_align
from detectron2.layers import ROIAlign, cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures.masks import polygons_to_bitmask

# from torch.nn.functional import binary_cross_entropy

def get_gt_mask(pred_mask_logits, instances,):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)
    return gt_masks

def generate_block_target(mask_target, boundary_width=3):
    mask_target = mask_target.float()

    # boundary region
    kernel_size = 2 * boundary_width + 1
    laplacian_kernel = - torch.ones(1, 1, kernel_size, kernel_size).to(
        dtype=torch.float32, device=mask_target.device).requires_grad_(False)
    laplacian_kernel[0, 0, boundary_width, boundary_width] = kernel_size ** 2 - 1

    pad_target = F.pad(mask_target.unsqueeze(1), (boundary_width, boundary_width, boundary_width, boundary_width), "constant", 0)

    # pos_boundary
    pos_boundary_targets = F.conv2d(pad_target, laplacian_kernel, padding=0)
    pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    pos_boundary_targets[pos_boundary_targets > 0.1] = 1
    pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
    pos_boundary_targets = pos_boundary_targets.squeeze(1)

    # neg_boundary
    neg_boundary_targets = F.conv2d(1 - pad_target, laplacian_kernel, padding=0)
    neg_boundary_targets = neg_boundary_targets.clamp(min=0) / float(kernel_size ** 2)
    neg_boundary_targets[neg_boundary_targets > 0.1] = 1
    neg_boundary_targets[neg_boundary_targets <= 0.1] = 0
    neg_boundary_targets = neg_boundary_targets.squeeze(1)

    # generate block target
    block_target = torch.zeros_like(mask_target).long().requires_grad_(False)
    boundary_inds = (pos_boundary_targets + neg_boundary_targets) > 0
    foreground_inds = (mask_target - pos_boundary_targets) > 0
    block_target[boundary_inds] = 1
    block_target[foreground_inds] = 2
    return block_target


class RefineCrossEntropyLoss(nn.Module):

    def __init__(self,
                 stage_instance_loss_weight=[1.0, 1.0, 1.0, 1.0],
                 semantic_loss_weight=1.0,
                 boundary_width=2,
                 start_stage=1):
        super(RefineCrossEntropyLoss, self).__init__()

        self.stage_instance_loss_weight = stage_instance_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.boundary_width = boundary_width
        self.start_stage = start_stage

    def forward(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):
        loss_mask_set = []
        for idx in range(len(stage_instance_preds)):
            instance_pred, instance_target = stage_instance_preds[idx].squeeze(1), stage_instance_targets[idx]
            if len(instance_pred) == 0 :
                zero_loss = torch.zeros((1, ), device=instance_pred.device, dtype=torch.float32)[0]
                loss_mask_set.append(zero_loss)
                continue
            if idx <= self.start_stage:
                loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target)
                loss_mask_set.append(loss_mask)
                pre_pred = instance_pred.sigmoid() >= 0.5

            else:
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=self.boundary_width) == 1
                boundary_region = pre_boundary.unsqueeze(1)

                target_boundary = generate_block_target(
                    stage_instance_targets[idx - 1].float(), boundary_width=self.boundary_width) == 1
                boundary_region = boundary_region | target_boundary.unsqueeze(1)

                boundary_region = F.interpolate(
                    boundary_region.float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)
                boundary_region = (boundary_region >= 0.5).squeeze(1)

                loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target, reduction='none')
                loss_mask = loss_mask[boundary_region].sum() / boundary_region.sum().clamp(min=1).float()
                # loss_mask = F.binary_cross_entropy_with_logits(instance_pred, instance_target)
                loss_mask_set.append(loss_mask)

                # generate real mask pred, set boundary width as 1, same as inference
                pre_boundary = generate_block_target(pre_pred.float(), boundary_width=1) == 1

                pre_boundary = F.interpolate(
                    pre_boundary.unsqueeze(1).float(),
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True) >= 0.5

                pre_pred = F.interpolate(
                    stage_instance_preds[idx - 1],
                    instance_pred.shape[-2:], mode='bilinear', align_corners=True)

                pre_pred[pre_boundary] = stage_instance_preds[idx][pre_boundary]
                pre_pred = pre_pred.squeeze(1).sigmoid() >= 0.5

        assert len(self.stage_instance_loss_weight) == len(loss_mask_set)
        loss_instance = sum([weight * loss for weight, loss in zip(self.stage_instance_loss_weight, loss_mask_set)])
        loss_semantic = self.semantic_loss_weight * \
            F.binary_cross_entropy_with_logits(semantic_pred.squeeze(1), semantic_target)

        return loss_instance, loss_semantic


class ConvModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation
        )
        self.activate = nn.ReLU()
        self.init_weights()
    
    def _kaiming_init(self, module,
                    a=0,
                    mode='fan_out',
                    nonlinearity='relu',
                    bias=0,
                    distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.kaiming_uniform_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                nn.init.kaiming_normal_(
                    module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        self._kaiming_init(self.conv, nonlinearity='relu')

    def forward(self, x, activate=True):
        x = self.conv(x)
        x = self.activate(x)
        return x


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', ConvModule(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        # out_feat = (feat_1 + feat_2 + feat_3)
        # import pdb
        # pdb.set_trace()
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 ):
        super(SFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        # TODO

        self.semantic_roi_extractor = ROIPooler(
            output_size=out_size,
            scales= [1./self.semantic_out_stride],
            sampling_ratio=0,
            pooler_type='ROIAlignV2'
        )

        # self.semantic_roi_extractor = build_roi_extractor(dict(
        #         type='SingleRoIExtractor',
        #         roi_layer=dict(type='RoIAlign', output_size=out_size, sampling_ratio=0),
        #         out_channels=semantic_out_channel,
        #         featmap_strides=[semantic_out_stride, ]))
        self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            MultiBranchFusion(instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 2, 1)
        # self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.semantic_transform_out, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        # ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat], rois)
        ins_semantic_feats = self.relu(self.semantic_transform_out(ins_semantic_feats))
        concat_tensors.append(ins_semantic_feats)

        # instance masks
        instance_preds = self.instance_logits(instance_feats)[torch.arange(len(roi_labels)), roi_labels][:, None]
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2], mode='bilinear', align_corners=True)
        concat_tensors.append(instance_masks)

        # instance-wise semantic masks
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred
        ins_semantic_masks = self.semantic_roi_extractor([_semantic_pred], rois)
        # ins_semantic_masks = roi_align(
        #     _semantic_pred, rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, algined=True)
        # ins_semantic_masks = roi_align(
        #     _semantic_pred, rois, instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, 'avg', True)
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        concat_tensors.append(ins_semantic_masks)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        # import pdb
        # pdb.set_trace()
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = self.relu(self.upsample(fused_feats))

        # concat instance and semantic masks with fused feats again
        instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2], mode='bilinear', align_corners=True)
        fused_feats = torch.cat([fused_feats, instance_masks, ins_semantic_masks], dim=1)

        return instance_preds, fused_feats


@ROI_MASK_HEAD_REGISTRY.register()
class RefineMaskHead(nn.Module):
    @configurable
    def __init__(self,
                 *,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 dilations=[1, 3, 5],
                 semantic_out_stride=8,
                 mask_use_sigmoid=True,
                #  stage_num_classes=[80, 80, 80, 80],
                 stage_num_classes=[1203, 1203, 1203, 1],
                 stage_sup_size=[14, 28, 56, 112],
                 cls_agn = True,
                 loss_cfg=dict(
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)
                ):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        if cls_agn :
            stage_num_classes = [1] * len(stage_num_classes)
        self.stage_num_classes = stage_num_classes
        self.cls_agn = cls_agn

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = RefineCrossEntropyLoss(**loss_cfg)
        # self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                mask_use_sigmoid=mask_use_sigmoid,
                )

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.semantic_logits = nn.Conv2d(conv_out_channels_semantic, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.refine_mask = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        semantic_out_stride = cfg.MODEL.REFINE_MASK.SEMANTIC_OUT_STRIDE
        return {'semantic_out_stride': semantic_out_stride}

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in [self.final_instance_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    
    def forward(self, x, seg_x, instances, sem_seg_gt):
        for conv in self.instance_convs :
            seg_x = conv(seg_x)
        for conv in self.semantic_convs :
            x = conv(x)
        
        seg_pred = self.semantic_logits(seg_x)

        stage_ins_preds = []
        # TODO
        rois = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
        roi_labels = torch.cat([x.gt_classes if self.training else x.pred_classes for x in instances])
        if self.cls_agn :
            roi_labels = roi_labels.clamp(max=0)
        for i, stage in enumerate(self.stages) :
            ins_preds, x = stage(x, seg_x, seg_pred, rois, roi_labels)
            stage_ins_preds.append(ins_preds)
        
        ins_preds = self.final_instance_logits(x)
        stage_ins_preds.append(ins_preds)
        if self.training :
            stage_instance_targets = self.get_mask_targets(stage_ins_preds, seg_pred, instances)
            seg_pred = seg_pred.float()
            sem_seg_gt = (sem_seg_gt > 0).float()
            seg_pred = F.interpolate(seg_pred, scale_factor=self.semantic_out_stride, mode='bilinear')
            loss_mask, loss_seg = self.loss_func(stage_ins_preds, seg_pred, stage_instance_targets, sem_seg_gt)
            return dict(loss_mask=loss_mask, loss_seg=loss_seg)
        else :
            stage_ins_preds = stage_ins_preds[1:]
            for idx in range(len(stage_ins_preds) - 1):
                instance_pred = stage_ins_preds[idx].squeeze(1).sigmoid() >= 0.5
                non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                non_boundary_mask = F.interpolate(
                    non_boundary_mask.float(),
                    stage_ins_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                pre_pred = F.interpolate(
                    stage_ins_preds[idx],
                    stage_ins_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                stage_ins_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
            ins_preds = stage_ins_preds[-1]
            mask_rcnn_inference(ins_preds, instances)
            return instances
        

    def get_mask_targets(self, stage_ins_preds, seg_pred, instances):
        stage_gt_masks = []
        semantic_gts = []

        for ins_pred in stage_ins_preds :
            stage_gt_masks.append(get_gt_mask(ins_pred, instances))
        
        stage_gt_masks = [x.to(dtype=torch.float32) for x in stage_gt_masks]
        return stage_gt_masks


    def loss(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):

        loss_instance, loss_semantic = self.loss_func(
            stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target)

        return dict(loss_instance=loss_instance), dict(loss_semantic=loss_semantic)


if __name__ == '__main__':
    # head = RefineMaskHead()
    # from detectron2.structures import Boxes
    # import torch
    # a = torch.rand(16,256,14,14)
    # b = torch.rand(16,256,128,128)
    # rois = torch.zeros(16,4)
    # rois[:,2:] = 1
    # rois = Boxes(rois)
    # rois = [rois] * 16
    # print(head.__class__)
    # result = head(a,b,rois)

    net = ConvModule(256,256,kernel_size=1,padding=1,dilation=0).cuda()
    a = torch.rand(63,256,14,14).cuda()

    result = net(a)
    print(result.shape)