# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, collate_data_and_cast2, DataAugmentationDINO, MaskingGenerator
import dinov2.distributed as distributed
for envname in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    ):
    print(envname, os.environ[envname])
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger, setup_logging
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from ssl_meta_arch import SSLMetaArch
from torchvision.datasets import ImageFolder
import webdataset as wds

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser

def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

def identity(x):
    return x

def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    def make_sample(sample, val=False):
        """Take a decoded sample dictionary, augment it, and return an (image, label) tuple."""
        assert not val, "only implemented training dataset for this notebook"
        image = sample["jpg"]
        label = 0 # sample["cls"]
        return data_transform(image), label
    def make_sample2(sample, val=False):
        """Take a decoded sample dictionary, augment it, and return an (image, label) tuple."""
        assert not val, "only implemented training dataset for this notebook"
        image = sample["jpg"]
        label = 0 # sample["cls"]
        return data_transform(image), label, sample["__key__"]

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    collate_fn2 = partial(
        collate_data_and_cast2,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    # setup data loader

    if False:
        # dataset = make_dataset(
        #     dataset_str=cfg.train.dataset_path,
        #     transform=data_transform,
        #     target_transform=lambda _: (),
        # )
        dataset = ImageFolder(
            root=cfg.train.dataset_path,
            transform=data_transform,
            target_transform=lambda _: (),
        )
        # sampler_type = SamplerType.INFINITE
        sampler_type = SamplerType.SHARDED_INFINITE
        data_loader = make_data_loader(
            dataset=dataset,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            seed=start_iter,  # TODO: Fix this -- cfg.train.seed
            sampler_type=sampler_type,
            sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=True,
            collate_fn=collate_fn,
        )
    elif False:
        training_urls = "file:///data/zhongz2/tcga_tars/{00000..00512}.tar.gz"
        ntrain = 159011314  #  number of patches

        # Parameters
        cache_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_dir')
        os.makedirs(cache_dir, exist_ok=True)

        dataset = (
            wds.WebDataset(
                training_urls, 
                resampled=True, 
                cache_dir=cache_dir, 
                shardshuffle=1000, 
                nodesplitter=None
            )
            .shuffle(5000)
            .decode("pil")
            .map(make_sample)
            .batched(cfg.train.batch_size_per_gpu, partial=False, collation_fn=collate_fn)
        )
        data_loader = wds.WebLoader(dataset, batch_size=None, num_workers=cfg.train.num_workers)
        nbatches = max(1, ntrain // (cfg.train.batch_size_per_gpu * distributed.get_global_size()))
        data_loader = data_loader.with_epoch(nbatches)
    else:
        training_urls = "file:///data/zhongz2/temp29/fake_dataset/{00000..00009}.tar.gz"
        ntrain = 200 #121 # 159011314  #  number of patches
        
        # Parameters
        cache_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_dir')
        os.makedirs(cache_dir, exist_ok=True)

        dataset = (
            wds.WebDataset(
                training_urls, 
                repeat=True,
                resampled=True,
                cache_size=2e11,   # 200G
                cache_dir=cache_dir, 
                shardshuffle=False, 
                nodesplitter=wds.split_by_node
            )
            .decode("pil")
            .map(make_sample2)
            .batched(cfg.train.batch_size_per_gpu, partial=False, collation_fn=collate_fn2)
        )
        data_loader = wds.WebLoader(dataset, batch_size=None, num_workers=cfg.train.num_workers)
        # nbatches = max(1, ntrain // (cfg.train.batch_size_per_gpu * distributed.get_global_size()))
        # data_loader = data_loader.with_epoch(nbatches)
    
    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    # import pdb
    # pdb.set_trace()
    os.environ["GOPEN_VERBOSE"] = "1"
    for data in metric_logger.log_every(
        data_loader,
        1,
        header,
        max_iter,
        start_iter,
    ):

        # if False: # wds
        #     if distributed.get_global_rank() == 0:
        #         print(type(data), len(data))
        #         print(type(data[0]),len(data[0]))
        #         print(type(data[0][0]))
        #         print(data[0][0].keys())
        #         # print(data.keys())
        #         # print(data['collated_global_crops'].shape)
        #         # print(data['collated_local_crops'].shape)
        # else:
        #     if distributed.get_global_rank() == 0:
        #         print(type(data), len(data))
        #         print(data.keys())
        #         print(data['collated_global_crops'].shape)
        #         print(data['collated_local_crops'].shape)  

        print(type(data), data.keys())
        logger.info('rank {}/{}, local_rank {}/{}, MYKEYS={}'.format(
            distributed.get_global_rank(), distributed.get_global_size(),
            distributed.get_local_rank(), distributed.get_local_size(),
            '|'.join(data['collated_keys']))
        )
        if iteration > max_iter:
            break

        continue

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    cfg = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


def check_log():
    import os
    log_dir = '/data/zhongz2/temp29/dinov2_output_v2_debug/logs'
    alllines = []
    for rank in range(4):
        if rank == 0:
            filename = os.path.join(log_dir, 'log.txt')
        else:
            filename = os.path.join(log_dir, f'log.txt.rank{rank}')
        ranklines = []
        with open(filename, 'r') as fp:
            for line in fp.readlines():
                if 'MYKEYS=' in line:
                    ranklines.extend(line.strip().split('MYKEYS=')[1].split('|'))
        ranklines = sorted(ranklines)
        alllines.extend(ranklines)
        print('====== rank {}, {}, {}'.format(rank, len(ranklines), len(set(ranklines))))
        # for line in ranklines:
        #     print(line)
    alllines = sorted(alllines)
    alllines1 = set(alllines)
    print(len(alllines1))


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
