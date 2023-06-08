# Copyright (c) Facebook, Inc. and its affiliates.
from functools import partial
import logging
import copy
import os
import sys
import json
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    # inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
# from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config

# from xpaste.evaluation.evaluator import inference_on_dataset_exp as inference_on_dataset
from xpaste.evaluation.evaluator import inference_on_dataset_exp, inference_on_dataset
from xpaste.config import add_xpaste_config
from xpaste.data.custom_build_augmentation import build_custom_augmentation
from xpaste.data.custom_dataset_dataloader import  build_custom_train_loader
from xpaste.data.custom_dataset_mapper import CustomDatasetMapper
from xpaste.data.dataset_mapper_with_sem_seg import DatasetMapperWithSemSeg
from xpaste.data.dataset_mapper import DatasetMapper
from xpaste.custom_solver import build_custom_optimizer
from xpaste.evaluation.oideval import OIDEvaluator
from xpaste.evaluation.custom_coco_eval import CustomCOCOEvaluator
from xpaste.modeling.utils import reset_cls_test
import xpaste.modeling.roi_heads.refine_mask_head
from xpaste import ModelEma

logger = logging.getLogger("detectron2")
def do_test(cfg, model, model_ema=None):
    if model_ema is not None :
        model = model_ema.ema
    if cfg.TEST.ANALYSE:
        from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances
        lvis_reg = {"lvis_v1": {
            "lvis_v1_train_analyse": ("train_imgs", "val.json"),
        },
        }
        root = 'OUTPUT/gen_data/scpv1_log/'
        # def register_all_lvis(root):
        for dataset_name, splits_per_dataset in lvis_reg.items():
            for key, (image_root, json_file) in splits_per_dataset.items():
                register_lvis_instances(
                    key,
                    get_lvis_instances_meta(dataset_name),
                    os.path.join(root, json_file) if "://" not in json_file else json_file,
                    os.path.join(root, image_root),
                )

        infer_func = partial(inference_on_dataset_exp, save_path=os.path.join(cfg.OUTPUT_DIR,'inst'))
        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR,'inst')):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR,'inst'))
    else :
        infer_func = inference_on_dataset
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d])
        mapper = DatasetMapper(cfg, False)
        # mapper.is_train = True # so that can see instances annotations
        mapper = mapper if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis" or cfg.GEN_PSEDO_LABELS:
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = infer_func(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False, model_ema=None):
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    kwargs = {'model_ema':model_ema} if model_ema is not None else {}
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler, **kwargs
    )

    start_iter = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            # TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    use_custom_mapper = cfg.WITH_IMAGE_LABELS
    MapperClass = CustomDatasetMapper if use_custom_mapper else DatasetMapper
    if cfg.MODEL.ROI_MASK_HEAD.NAME == 'RefineMaskHead' :
        MapperClass = DatasetMapperWithSemSeg

    if cfg.INPUT.USE_INP_ROTATE :
        inp_root = cfg.INPUT.INP_ROOT
        inp_anno = cfg.INPUT.INP_ANNO
        inp_anno = json.load(open(inp_anno))
    else :
        inp_anno = {}
    inp_anno = {int(k):os.path.join(inp_root, v) for k, v in inp_anno.items()}

    mapper = MapperClass(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else  \
        MapperClass(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    from xpaste.data.custom_build_copypaste_mapper import CopyPasteMapper
    mapper = CopyPasteMapper(mapper, cfg)

    loader_kwargs = {}
    if cfg.INPUT.ONLY_RC :
        with open(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH) as f :
            freq_dict = json.load(f)
            cid_to_freq = dict()
            # convert 1-ind to 0-ind
            for x in freq_dict :
                cid_to_freq[x['id']-1] = x['frequency']

        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        new_dataset = []
        for data in dataset :
            new_anno = []
            for anno in data['annotations']:
                if cid_to_freq[anno['category_id']] in ('c', 'r'):
                    new_anno.append(anno)
            if len(new_anno):
                data = copy.deepcopy(data)
                data['annotations'] = new_anno
                new_dataset.append(data)
        loader_kwargs = {'dataset': new_dataset}

    if len(cfg.INPUT.SELECT_CATS_LIST):
        cats_list = cfg.INPUT.SELECT_CATS_LIST
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        new_dataset = []
        for data in dataset :
            new_anno = []
            for anno in data['annotations']:
                if anno['category_id'] in cats_list:
                    new_anno.append(anno)
            if len(new_anno):
                data = copy.deepcopy(data)
                data['annotations'] = new_anno
                new_dataset.append(data)
        loader_kwargs = {'dataset': new_dataset}

    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper, **loader_kwargs)
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper, **loader_kwargs)
    ### set_dataset
    mapper.set_dataset(copy.deepcopy(data_loader.dataset.dataset.dataset._dataset))
    if cfg.FP16:
        scaler = GradScaler()

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            ##############################
            ##print('data:',data[0]['image'].shape)
            ###########################
            if cfg.TEST.GEN_DATASET :
                print('iter', iteration)
                continue
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)
            extra_augment = {}
            if model_ema is not None :
                model_ema.update(model)
                extra_augment['model_ema'] = model_ema.state_dict()

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter):
                do_test(cfg, model, model_ema)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration, **extra_augment)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_xpaste_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    if '/amlt' in cfg.OUTPUT_DIR:
        file_name = os.environ.get('AMLT_OUTPUT_DIR','OUTPUT')
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/amlt', file_name)
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="xpaste")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        if cfg.SOLVER.MODEL_EMA > 0 :
            import tempfile
            tmp = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')
            tmp['model'] = tmp['model_ema']
            tmp_file = tempfile.NamedTemporaryFile()
            torch.save(tmp, tmp_file.name)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                tmp_file.name, resume=args.resume
            )
        else :
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0 :
        model_ema = ModelEma(model, cfg.SOLVER.MODEL_EMA)
    do_train(cfg, model, resume=args.resume, model_ema=model_ema)
    return do_test(cfg, model, model_ema)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        args.dist_url = 'tcp://{}:{}'.format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        # args.dist_url = 'env://'
        print('args dist url', args.dist_url)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
