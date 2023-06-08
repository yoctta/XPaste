# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .lvis_v1 import custom_load_lvis_json
def custom_register_lvis_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="lvis", **metadata
    )


custom_register_lvis_instances(
    'syn4det_detic_pool_b21_30k',
    get_lvis_instances_meta('lvis_v1'),
    '/mnt/home/syn4det/DeticPool_b21_30K.json','/',
)

custom_register_lvis_instances(
    'syn_bg_pool',
    get_lvis_instances_meta('lvis_v1'),
    '/mnt/home/syn4det/SYN_BG_POOL.json','/',
)

custom_register_lvis_instances(
    'direct_30k',
    get_lvis_instances_meta('lvis_v1'),
    '/mnt/home/syn4det/direct_30k.json','/',
)