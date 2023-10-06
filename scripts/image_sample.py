"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
from PIL import Image

def main():
    args = create_argparser().parse_args()

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    dist_util.setup_dist()
    for ckpt in range(args.begin_ckpt,args.end_ckpt,args.step_size):
        model_path = os.path.join(args.model_path,f'model{ckpt:06d}.pt')
        output_dir = f'output/{os.path.basename(args.model_path)}model{ckpt:06d}'
        logger.configure(dir = output_dir)
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("sampling...")
        if args.sampler == "multistep":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None

        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            for i in range(100):
                generator = get_generator(args.generator, args.num_samples, i)
                model_kwargs = {}
                num_classes = 1
                target_class = i%num_classes
                if args.class_cond:
                    classes = th.randint(
                        low=target_class, high=target_class+1, size=(args.batch_size,), device=dist_util.dev()
                    )
                    model_kwargs["y"] = classes

                sample = karras_sample(
                    diffusion,
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    steps=args.steps,
                    model_kwargs=model_kwargs,
                    device=dist_util.dev(),
                    clip_denoised=args.clip_denoised,
                    sampler=args.sampler,
                    sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max,
                    s_churn=args.s_churn,
                    s_tmin=args.s_tmin,
                    s_tmax=args.s_tmax,
                    s_noise=args.s_noise,
                    generator=generator,
                    ts=ts,
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                
                for index,img in enumerate(sample):
                    image = Image.fromarray(img.cpu().numpy())
                    image.save(os.path.join(output_dir,f'sample_{i}.png'))
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        dist.barrier()
        logger.log("sampling complete")


        #     if args.class_cond:
        #         gathered_labels = [
        #             th.zeros_like(classes) for _ in range(dist.get_world_size())
        #         ]
        #         dist.all_gather(gathered_labels, classes)
        #         all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        #     logger.log(f"created {len(all_images) * args.batch_size} samples")

        # arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples]
        # if args.class_cond:
        #     label_arr = np.concatenate(all_labels, axis=0)
        #     label_arr = label_arr[: args.num_samples]
        # if dist.get_rank() == 0:
        #     shape_str = "x".join([str(x) for x in arr.shape])
        #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        #     logger.log(f"saving to {out_path}")
        #     if args.class_cond:
        #         np.savez(out_path, arr, label_arr)
        #     else:
        #         np.savez(out_path, arr)



def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=40,
        ts="",
        output_dir = 'output/exp5.1-out',
        begin_ckpt = 0,
        end_ckpt = 0,
        step_size = 0,
        
        
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
