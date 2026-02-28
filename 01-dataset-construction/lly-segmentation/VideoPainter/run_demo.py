# -*- coding: utf-8 -*-
"""
VideoPainter Demo - Command Line Version
Usage: python run_demo.py --video bunny.mp4 --click_x 360 --click_y 240 --prompt "A cute bunny plush toy on a table"
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from decord import VideoReader

def extract_frames(video_path, target_fps=8, max_seconds=6, target_h=480, target_w=720):
    """Load video, resize, resample to target fps, limit duration."""
    print(f"[1/5] Loading video: {video_path}")
    vr = VideoReader(video_path)
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / original_fps
    print(f"  Original: {total_frames} frames, {original_fps:.1f} fps, {duration:.1f}s")

    # Limit to max_seconds
    use_seconds = min(duration, max_seconds)
    use_frames = int(use_seconds * original_fps)

    # Sample at target_fps
    step = max(1, int(original_fps / target_fps))
    indices = list(range(0, use_frames, step))

    frames_raw = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)

    # Resize to target resolution
    frames = []
    for f in frames_raw:
        resized = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(resized)
    frames = np.array(frames)

    print(f"  Extracted: {len(frames)} frames at {target_fps}fps, {target_h}x{target_w}")
    return frames, target_fps

def create_masks_with_sam2(frames, click_points, click_labels, sam2_checkpoint, model_cfg):
    """Use SAM2 to segment and track object across frames."""
    print(f"[2/5] Running SAM2 segmentation and tracking...")

    # Add the app directory to path for SAM2
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
    app_dir = os.path.abspath(app_dir)
    sys.path.insert(0, app_dir)

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # Initialize with frames (numpy array)
    inference_state = predictor.init_state(images=frames, offload_video_to_cpu=True, async_loading_frames=True)
    predictor.reset_state(inference_state)

    # Add click points on frame 0
    points = np.array(click_points, dtype=np.float32)
    labels = np.array(click_labels, dtype=np.int32)

    frame_idx, obj_ids, mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        points=points,
        labels=labels,
    )

    # Show first frame mask quality
    mask0 = (mask_logits[0, 0] > 0).cpu().numpy()
    mask_area = mask0.sum()
    total_area = mask0.shape[0] * mask0.shape[1]
    print(f"  First frame mask: {mask_area} pixels ({100*mask_area/total_area:.1f}% of frame)")

    if mask_area < 100:
        print("  WARNING: Mask is very small! The click point might not be on the target object.")
        print(f"  Click point: {click_points}, try adjusting --click_x and --click_y")

    # Propagate to all frames
    print(f"  Tracking across {len(frames)} frames...")
    masks = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
        masks[frame_idx] = mask

    print(f"  Tracking complete! Mask shape: {masks.shape}")
    return masks

def save_visualization(frames, masks, output_dir, click_points=None):
    """Save first frame with mask overlay for verification."""
    print(f"[3/5] Saving visualization...")
    os.makedirs(output_dir, exist_ok=True)

    # First frame with mask overlay
    frame0 = frames[0].copy()
    mask0 = masks[0]

    # Draw click points
    if click_points:
        for pt in click_points:
            cv2.circle(frame0, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
            cv2.circle(frame0, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 2)

    # Overlay mask in red
    overlay = frame0.copy()
    overlay[mask0 > 0] = [255, 0, 0]
    blended = cv2.addWeighted(frame0, 0.6, overlay, 0.4, 0)

    cv2.imwrite(os.path.join(output_dir, "frame0_with_mask.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # Save mask only
    cv2.imwrite(os.path.join(output_dir, "mask0.png"), mask0 * 255)

    # Save original first frame
    cv2.imwrite(os.path.join(output_dir, "frame0_original.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

    print(f"  Saved to {output_dir}/")

def run_inpainting(frames, masks, prompt, model_path, inpainting_branch,
                   output_path, num_inference_steps=50, guidance_scale=6.0,
                   seed=42, dilate_size=32, first_frame_caption=""):
    """Run VideoPainter inpainting pipeline."""
    print(f"[4/5] Running VideoPainter inpainting...")
    print(f"  Prompt: {prompt}")
    print(f"  Frames: {len(frames)}, inference steps: {num_inference_steps}")

    from diffusers import (
        CogVideoXDPMScheduler,
        CogvideoXBranchModel,
        CogVideoXI2VDualInpaintAnyLPipeline,
    )
    from diffusers.utils import export_to_video

    dtype = torch.bfloat16

    # Prepare video, masked_video, binary_masks as PIL images
    video_pil = []
    masked_video_pil = []
    binary_masks_pil = []

    for i in range(len(frames)):
        frame = frames[i]
        mask = masks[i]

        video_pil.append(Image.fromarray(frame).convert("RGB"))

        # Dilate mask
        if dilate_size > 0:
            dilated = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8))
        else:
            dilated = mask

        # Masked frame (black out the masked region)
        masked_frame = frame.copy()
        masked_frame[dilated > 0] = 0
        masked_video_pil.append(Image.fromarray(masked_frame).convert("RGB"))

        # Binary mask (white = inpaint region)
        mask_rgb = np.zeros_like(frame)
        mask_rgb[dilated > 0] = 255
        binary_masks_pil.append(Image.fromarray(mask_rgb.astype(np.uint8)).convert("RGB"))

    # CogVideoX needs 4n+1 frames, max 49. Trim to fit.
    max_frames = 49
    n_available = len(video_pil)
    # Find largest 4n+1 <= min(n_available, max_frames)
    n_use = min(n_available, max_frames)
    num_frames = ((n_use - 1) // 4) * 4 + 1  # e.g. 34 -> 33

    video_pil = video_pil[:num_frames]
    masked_video_pil = masked_video_pil[:num_frames]
    binary_masks_pil = binary_masks_pil[:num_frames]
    print(f"  Using {num_frames} frames for inpainting")

    # First frame ground truth - clear mask for first frame
    gt_mask_first_frame = binary_masks_pil[0]
    binary_masks_pil[0] = Image.fromarray(np.zeros_like(np.array(binary_masks_pil[0]))).convert("RGB")
    gt_video_first_frame = video_pil[0]

    # Load models
    print("  Loading VideoPainter branch model...")
    branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(dtype=dtype)

    print("  Loading CogVideoX pipeline...")
    pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        model_path,
        branch=branch,
        torch_dtype=dtype,
    )

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Enable memory optimizations - use CPU offload to fit in 32GB VRAM
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    # Run inpainting
    print("  Running inference (this may take a few minutes)...")
    image = masked_video_pil[0]

    inpaint_outputs = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        video=masked_video_pil,
        masks=binary_masks_pil,
        strength=1.0,
        replace_gt=True,
        mask_add=True,
        stride=num_frames,
        prev_clip_weight=0.0,
        output_type="np"
    ).frames[0]

    print("  Inference complete!")

    # Visualize: original | masked | mask | generated
    video_generate = inpaint_outputs
    binary_masks_pil[0] = gt_mask_first_frame
    video_pil[0] = gt_video_first_frame

    # Build side-by-side visualization
    print("[5/5] Saving output video...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    original_video = pipe.video_processor.preprocess_video(video_pil[:len(video_generate)], height=video_generate.shape[1], width=video_generate.shape[2])
    masks_vis = pipe.masked_video_processor.preprocess_video(binary_masks_pil[:len(video_generate)], height=video_generate.shape[1], width=video_generate.shape[2])
    masked_video_vis = original_video * (masks_vis < 0.5)

    original_video_np = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video_np = pipe.video_processor.postprocess_video(video=masked_video_vis, output_type="np")[0]
    masks_np = masks_vis.squeeze(0).squeeze(0).numpy()
    masks_np = masks_np[..., np.newaxis].repeat(3, axis=-1)

    # Concatenate horizontally: original | masked | mask | generated
    concat_frames = []
    for i in range(len(video_generate)):
        row = np.concatenate([
            np.array(original_video_np[i]),
            np.array(masked_video_np[i]),
            np.array(masks_np[i]),
            np.array(video_generate[i])
        ], axis=1)
        concat_frames.append(row)

    # Save concatenated visualization
    vis_path = output_path.replace(".mp4", "_vis.mp4")
    export_to_video(concat_frames, vis_path, fps=8)
    print(f"  Visualization saved: {vis_path}")

    # Save generated video only
    export_to_video(list(video_generate), output_path, fps=8)
    print(f"  Output video saved: {output_path}")

    print("\nDone! Check the output files.")

def main():
    parser = argparse.ArgumentParser(description="VideoPainter Demo - Command Line")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--click_x", type=int, required=True, help="X coordinate of click point on first frame (0-719)")
    parser.add_argument("--click_y", type=int, required=True, help="Y coordinate of click point on first frame (0-479)")
    parser.add_argument("--prompt", type=str, required=True, help="Video description for inpainting")
    parser.add_argument("--first_frame_caption", type=str, default="", help="First frame caption (optional)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model_path", type=str, default="../ckpt/CogVideoX-5b-I2V")
    parser.add_argument("--inpainting_branch", type=str, default="../ckpt/VideoPainter/checkpoints/branch")
    parser.add_argument("--sam2_checkpoint", type=str, default="../ckpt/sam2_hiera_large.pt")
    parser.add_argument("--sam2_config", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dilate_size", type=int, default=32)
    parser.add_argument("--target_fps", type=int, default=8)
    parser.add_argument("--max_seconds", type=float, default=6.0)
    parser.add_argument("--mask_only", action="store_true", help="Only generate masks, don't run inpainting")

    args = parser.parse_args()

    # Step 1: Extract frames
    frames, fps = extract_frames(args.video, target_fps=args.target_fps, max_seconds=args.max_seconds)

    # Step 2: Create masks with SAM2
    click_points = [[args.click_x, args.click_y]]
    click_labels = [1]  # positive point
    masks = create_masks_with_sam2(
        frames, click_points, click_labels,
        args.sam2_checkpoint, args.sam2_config
    )

    # Step 3: Save visualization
    save_visualization(frames, masks, args.output_dir, click_points)

    if args.mask_only:
        print("\n--mask_only mode: Skipping inpainting. Check the mask visualization.")
        print(f"  Frame: {args.output_dir}/frame0_with_mask.png")
        print(f"  Mask: {args.output_dir}/mask0.png")
        return

    # Step 4: Run inpainting
    output_path = os.path.join(args.output_dir, "result.mp4")
    run_inpainting(
        frames, masks, args.prompt,
        args.model_path, args.inpainting_branch,
        output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        dilate_size=args.dilate_size,
        first_frame_caption=args.first_frame_caption,
    )

if __name__ == "__main__":
    main()
