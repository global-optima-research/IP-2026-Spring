import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
GRADIO_TEMP_DIR = "./tmp_gradio"
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
from openai import OpenAI
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image
from safetensors import safe_open
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

def load_model(
    model_path,
    inpainting_branch,
    img_inpainting_model,
    id_adapter,
    device="cuda:0",
    dtype=torch.bfloat16
):

    branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(device, dtype=dtype)
    
    # load the transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        id_pool_resample_learnable=True,
    ).to(device, dtype=dtype)

    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        model_path,
        branch=branch,
        transformer=transformer,
        torch_dtype=dtype,
    )

    pipe.load_lora_weights(
        id_adapter, 
        weight_name="pytorch_lora_weights.safetensors", 
        adapter_name="test_1",
        target_modules=["transformer"]
        )
    # pipe.fuse_lora(lora_scale=1 / lora_rank)

    list_adapters_component_wise = pipe.get_list_adapters()
    print(f"list_adapters_component_wise: {list_adapters_component_wise}")

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to(device)
    pipe.to(device)
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    pipe_img_inpainting = FluxFillPipeline.from_pretrained(img_inpainting_model, torch_dtype=dtype).to(device)
    # pipe_img_inpainting = None
    return pipe, pipe_img_inpainting


def generate_frames(
        images, 
        masks, 
        pipe, 
        pipe_img_inpainting, 
        prompt, 
        image_inpainting_prompt,
        seed=42,
        cfg_scale=6.0,
        dilate_size=16
    ):
    # save the first frame
    images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame.png")
    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask.png")
    # for i in range(len(masks)):
    #     masks[i].save(f"{GRADIO_TEMP_DIR}/inpaint/mask_{i:03d}.png")

    print(f"Dilating the mask with size {dilate_size}...")
    for i in range(len(masks)):
        mask = cv2.dilate(np.array(masks[i]), np.ones((dilate_size, dilate_size)))
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        masks[i] = mask

    masks[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_mask_dilate.png")
    masks[-1].save(f"{GRADIO_TEMP_DIR}/inpaint/last_mask_dilate.png")

    print(f"Image inpainting prompt: {image_inpainting_prompt}")

    pipe_img_inpainting.to("cuda")
    image_inpainting = pipe_img_inpainting(
        prompt=image_inpainting_prompt,
        image=images[0],
        mask_image=masks[0],
        height=images[0].size[1],
        width=images[0].size[0],
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    pipe_img_inpainting.to("cpu")
    torch.cuda.empty_cache()
    images[0] = image_inpainting
    print(f"Image inpainting done! {np.array(images[0]).shape}")

    # save the first frame
    images[0].save(f"{GRADIO_TEMP_DIR}/inpaint/first_frame_inpainted.png")

    masks[0] = Image.fromarray(np.zeros_like(np.array(masks[0]))).convert("RGB")

    inpaint_outputs = pipe(
        prompt=prompt,
        image=images[0],
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=cfg_scale,
        generator=torch.Generator().manual_seed(seed),
        video=images,
        masks=masks,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride= int(49 - 0), # int(frames - down_sample_fps), frames,
        prev_clip_weight=0.0,
        id_pool_resample_learnable=False,
        output_type="np"
    ).frames[0]
    inpaint_outputs = inpaint_outputs[1:]
    print(f"Video inpainting done! {np.array(inpaint_outputs).shape}, {np.array(inpaint_outputs).min()}, {np.array(inpaint_outputs).max()}")
    torch.cuda.empty_cache()
    return inpaint_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--inpainting_branch", type=str, default="")
    parser.add_argument("--img_inpainting_model", type=str, default="../")
    args = parser.parse_args()


    validation_pipeline = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        img_inpainting_model=args.img_inpainting_model
    )
