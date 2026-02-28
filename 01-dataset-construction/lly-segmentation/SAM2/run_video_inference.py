"""
SAM2 Video Inference Script
Runs video segmentation on bunny.mp4 and saves results.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import subprocess

# Add project to path
SAM2_ROOT = "/data/liuluyan/SAM2/sam2"
sys.path.insert(0, SAM2_ROOT)
os.chdir(SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor

# Paths
CHECKPOINT = "/data/liuluyan/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
VIDEO_FILE = os.path.join(SAM2_ROOT, "notebooks/videos/bunny.mp4")
FRAMES_DIR = os.path.join(SAM2_ROOT, "notebooks/videos/bunny_frames")
OUTPUT_DIR = "/data/liuluyan/SAM2/results/video"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# ========================
# Step 1: Extract frames from video
# ========================
print("\n" + "="*60)
print("Step 1: Extracting frames from video")
print("="*60)

existing_frames = [f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')]
if len(existing_frames) < 5:
    cmd = f'ffmpeg -i "{VIDEO_FILE}" -q:v 2 -start_number 0 "{FRAMES_DIR}/%05d.jpg" -y'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print("Frame extraction complete!")
else:
    print(f"Frames already exist ({len(existing_frames)} frames), skipping extraction.")

# Get frame names
frame_names = [
    p for p in os.listdir(FRAMES_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(f"Total frames: {len(frame_names)}")

# ========================
# Step 2: Load model and initialize state
# ========================
print("\n" + "="*60)
print("Step 2: Loading SAM2 video predictor")
print("="*60)

predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device)
inference_state = predictor.init_state(video_path=FRAMES_DIR)
print("Model loaded and inference state initialized!")

# ========================
# Step 3: Get first frame info and add prompts
# ========================
print("\n" + "="*60)
print("Step 3: Adding prompts and tracking")
print("="*60)

first_frame = Image.open(os.path.join(FRAMES_DIR, frame_names[0]))
w, h = first_frame.size
print(f"Frame size: {w}x{h}")

# Use center point as the main prompt - tracking the main object
# For bunny video, the main subject is typically centered
center_x, center_y = w // 2, h // 2

# Add a positive click at center
ann_frame_idx = 0
ann_obj_id = 1

points = np.array([[center_x, center_y]], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)
print(f"Added point prompt at ({center_x}, {center_y})")

# Save first frame result
plt.figure(figsize=(9, 6))
plt.title(f"Frame {ann_frame_idx} - Initial prompt")
plt.imshow(np.array(first_frame))
plt.scatter(center_x, center_y, color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, "frame_000_prompt.png"), bbox_inches='tight', dpi=150)
plt.close()

# ========================
# Step 4: Propagate through video
# ========================
print("\n" + "="*60)
print("Step 4: Propagating masks through video")
print("="*60)

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
print(f"Propagation complete! Got masks for {len(video_segments)} frames")

# ========================
# Step 5: Save results
# ========================
print("\n" + "="*60)
print("Step 5: Saving results")
print("="*60)

# Save every 5th frame with mask overlay
frames_output_dir = os.path.join(OUTPUT_DIR, "frames_with_masks")
os.makedirs(frames_output_dir, exist_ok=True)

for frame_idx in range(0, len(frame_names), 5):
    plt.figure(figsize=(9, 6))
    plt.title(f"Frame {frame_idx}")
    frame_img = Image.open(os.path.join(FRAMES_DIR, frame_names[frame_idx]))
    plt.imshow(frame_img)
    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.axis('off')
    plt.savefig(os.path.join(frames_output_dir, f"frame_{frame_idx:05d}.png"), bbox_inches='tight', dpi=150)
    plt.close()

print(f"Saved {len(range(0, len(frame_names), 5))} frame visualizations")

# Create a comparison video (original vs masked)
print("\nCreating output video with mask overlay...")
video_output_path = os.path.join(OUTPUT_DIR, "bunny_segmented.mp4")

# Get frame dimensions
sample_frame = cv2.imread(os.path.join(FRAMES_DIR, frame_names[0]))
frame_h, frame_w = sample_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_output_path, fourcc, 15.0, (frame_w, frame_h))

cmap = plt.get_cmap("tab10")

for frame_idx in range(len(frame_names)):
    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_names[frame_idx]))

    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            mask = out_mask.squeeze()
            color_idx = out_obj_id % 10
            color = np.array(cmap(color_idx)[:3]) * 255
            color = color[::-1]  # RGB to BGR

            overlay = frame.copy()
            overlay[mask] = overlay[mask] * 0.4 + np.array(color) * 0.6
            frame = overlay.astype(np.uint8)

            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)

    out_video.write(frame)

out_video.release()
print(f"Output video saved to: {video_output_path}")

print("\n" + "="*60)
print(f"All video results saved to: {OUTPUT_DIR}")
print("="*60)
