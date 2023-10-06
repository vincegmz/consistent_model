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
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util import TrainLoop
import torch.distributed as dist
from cm.unet_adapter import ResBlock, AttentionBlock,AdapterForward
import os
def main():
    args = create_argparser().parse_args()
    os.makedirs(args.ckpt_dir,exist_ok=True)
    ckpt_files = [file for file in os.listdir(args.ckpt_dir) if file.startswith('model') and file.endswith('.pt')]
    if len(ckpt_files) !=0:
        if args.resume_checkpoint == "":
            resume_ckpt = os.path.join(args.ckpt_dir,sorted(ckpt_files,reverse=True)[0])
        else:
            resume_ckpt = args.resume_checkpoint
    else:
        resume_ckpt = ""
    dist_util.setup_dist()
    logger.configure(dir = args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if args.invert:
        if args.model_path is not None:
            model.load_state_dict(
                    dist_util.load_state_dict(args.model_path, map_location="cpu"),strict=False
            )
        else:
            print("For inversion, a pretrained model must be present")
            exit(1)
        for param in model.parameters():
            param.requires_grad = False
        # for param in model.output_blocks.parameters():
        #     param.requires_grad = True

        # for module in model.modules():
        #     if isinstance(module, AttentionBlock):
        #         for param in module.parameters():
        #             param.requires_grad = True

        def unfreeze_adapter(fineTune_attAdp, fineTune_resAdp):
            for name,module in model.named_modules():
                if isinstance(module, ResBlock) and fineTune_resAdp:
                    for n1, m1 in module.named_children():
                        if 'all_adapters' in n1:
                            for param in m1.parameters():
                                param.requires_grad = True
                elif isinstance(module,AttentionBlock) and fineTune_attAdp:
                    for n2, m2 in module.named_children():
                        if 'all_adapters' in n2:
                            for param in m2.parameters():
                                param.requires_grad = True
        if args.use_adapter:
            assert (args.fineTune_attAdp or args.fineTune_resAdp) is True
            unfreeze_adapter(args.fineTune_attAdp,args.fineTune_resAdp)

        # for name,module in model.named_modules():
        #     if 'all_adapters' in name:
        #         for param in module.parameters():
        #             param.requires_grad = True
        # for block in model.output_blocks:
        #     for layer in block:
        #         if isinstance(layer, AttentionBlock):
        #             for param in layer.parameters():
        #                 param.requires_grad = True

        # for block in model.input_blocks:
        #     for layer in block:
        #         if isinstance(layer, AttentionBlock):
        #             for param in layer.parameters():
        #                 param.requires_grad = True
        
        # last_resblock = None
        # for block in model.output_blocks:
        #     for layer in block:
        #         if isinstance(layer, ResBlock):
        #             last_resblock = layer
        # if last_resblock is not None:
        #     for param in last_resblock.parameters():
        #         param.requires_grad = True


        # for i, block in enumerate(model.output_blocks):
        #     if (i+1) // (args.num_res_blocks+1) == 1:
        #         last_resblock = None
        #         for layer in block:
        #             if isinstance(layer,ResBlock):
        #                 last_resblock = layer
        #                 break
        #         if last_resblock is not None:
        #             for param in last_resblock.parameters():
        #                 param.requires_grad = True
        # if args.class_cond:
        #     for param in model.label_emb.parameters():
        #         param.requires_grad = True
        # for i, block in enumerate(model.output_blocks):
        #     if (i+1) % (args.num_res_blocks+1) == 0:
        #         last_attblock = None
        #         for layer in block:
        #             if isinstance(layer,AttentionBlock):
        #                 last_attblock = layer
        #         if last_attblock is not None:
        #             for param in last_attblock.parameters():
        #                 param.requires_grad = True
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
        invert=args.invert,
        deterministic= args.deterministic
    )

    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoop(
        model=model,
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
        ckpt_dir=args.ckpt_dir,
        augment =args.augment,
        total_steps= args.total_steps,
        ResBlockRegularize= args.ResBlockRegularize,
        diversity_regularize = args.diversity_regularize,
        dynamic_weighted_sampling=args.dynamic_weighted_sampling
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
        augment = 1,
        deterministic = False,
        total_steps =10000,
        ResBlockRegularize = 0.0,
        diversity_regularize = 0.0,
        use_adapter = False,
        fineTune_attAdp = False,
        fineTune_resAdp = False,
        dynamic_weighted_sampling = False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
