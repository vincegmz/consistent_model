"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
from cm.ewc import EWC
import torch.distributed as dist
import copy
import os
import torch as th
from cm.unet import ResBlock, AttentionBlock
def main():
    args = create_argparser().parse_args()
    os.makedirs(args.ckpt_dir,exist_ok=True)
    ckpt_files = [file for file in os.listdir(args.ckpt_dir) if file.startswith('model') and file.endswith('.pt')]
    if len(ckpt_files) !=0:
        resume_ckpt = os.path.join(args.ckpt_dir,sorted(ckpt_files,reverse=True)[0])
    else:
        resume_ckpt = ""
    dist_util.setup_dist()
    logger.configure(dir = args.log_dir)

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        invert = args.invert
    )

    # if args.ewc:
    #     ewc_model = EWC(model,data)

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.
    if args.invert:
        if args.model_path is not None:
            model.load_state_dict(
                    dist_util.load_state_dict(args.model_path, map_location="cpu"),
            )
        else:
            print("For inversion, a pretrained model must be present")
            exit(1)
        for param in model.parameters():
            param.requires_grad = False
        if args.class_cond:
            for param in model.label_emb.parameters():
                param.requires_grad = True
        for i, block in enumerate(model.output_blocks):
            if i % (args.num_res_blocks+1) == 0:
                last_attblock = None
                for layer in block:
                    if isinstance(layer,AttentionBlock):
                        last_attblock = layer
                if last_attblock is not None:
                    for param in last_attblock.parameters():
                        param.requires_grad = True

        if teacher_model is not None:
            for param in teacher_model.parameters():
                param.requires_grad = False
            if args.class_cond:
                for param in teacher_model.label_emb.parameters():
                    param.requires_grad = True
            for i, block in enumerate(teacher_model.output_blocks):
                if i % (args.num_res_blocks+1) == 0:
                    last_attblock = None
                    for layer in block:
                        if isinstance(layer,AttentionBlock):
                            last_attblock = layer
                    if last_attblock is not None:
                        for param in last_attblock.parameters():
                            param.requires_grad = True
        
    logger.log("creating the target model")
    target_model, _ = create_model_and_diffusion(
        **model_and_diffusion_kwargs,
    )

    target_model.to(dist_util.dev())
    target_model.train()

    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=resume_ckpt,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        ckpt_dir = args.ckpt_dir,
        augment = args.augment,
        invert = args.invert
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir = "/media/minzhe_guo/ckpt/mnistm/cd_ckpt/exp1",
        ckpt_dir = "/media/minzhe_guo/ckpt/mnistm/cd_ckpt/exp1",
        invert = None,
        model_path = None,
        augment = 2,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
