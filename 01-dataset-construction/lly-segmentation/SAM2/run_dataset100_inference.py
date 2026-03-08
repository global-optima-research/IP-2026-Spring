"""
SAM2 Dataset100 Inference Script
Runs SAM2 automatic mask generation on dataset100 product images
and video tracking on dataset100 videos.
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
import json

MAX_IMAGE_SIZE = 2048

def resize_if_needed(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f'  Resizing from {w}x{h} to {new_w}x{new_h}')
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

SAM2_ROOT = '/data/liuluyan/SAM2/sam2'
sys.path.insert(0, SAM2_ROOT)
os.chdir(SAM2_ROOT)

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINT = '/data/liuluyan/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'

DATASET_DIR = '/data/datasets/pvtt_evaluation_datasets'
IMAGE_DIR = os.path.join(DATASET_DIR, 'product_images')
VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
OUTPUT_BASE = '/data/liuluyan/SAM2/dataset100_results'

# Same samples as Grounded-SAM-2 results
IMAGE_SAMPLES = ['handbag_1.jpg', 'sunglasses_1.jpg']
VIDEO_SAMPLES = ['0001-handfan1.mp4', '0006-handbag1_scene01.mp4']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if device.type == 'cuda':
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                    sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# ========================
# Part 1: Image - Automatic Mask Generation
# ========================
print('\n' + '=' * 60)
print('Part 1: Automatic Mask Generation on Dataset100 Images')
print('=' * 60)

image_output_dir = os.path.join(OUTPUT_BASE, 'image_results')
os.makedirs(image_output_dir, exist_ok=True)

sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)
predictor = SAM2ImagePredictor(sam2)

for img_file in IMAGE_SAMPLES:
    img_name = os.path.splitext(img_file)[0]
    sample_dir = os.path.join(image_output_dir, img_name)
    os.makedirs(sample_dir, exist_ok=True)

    print(f'\nProcessing: {img_file}')
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = np.array(Image.open(img_path).convert('RGB'))
    image = resize_if_needed(image)
    h, w = image.shape[:2]

    torch.cuda.empty_cache()

    # Auto mask generation
    masks = mask_generator.generate(image)
    print(f'  Generated {len(masks)} masks')

    # Save auto-mask result
    np.random.seed(3)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, f'{img_name}_auto_masks.png'),
                bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    # Save mask info as JSON
    mask_info = []
    for i, m in enumerate(masks):
        mask_info.append({
            'id': i,
            'area': int(m['area']),
            'predicted_iou': float(m['predicted_iou']),
            'stability_score': float(m['stability_score']),
            'bbox': [int(x) for x in m['bbox']],
        })
    with open(os.path.join(sample_dir, f'{img_name}_masks_info.json'), 'w') as f:
        json.dump(mask_info, f, indent=2)

    # Point prompt prediction (center point)
    torch.cuda.empty_cache()
    predictor.set_image(image)
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])
    pred_masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    pred_masks = pred_masks[sorted_ind]
    scores = scores[sorted_ind]

    # Save best mask
    np.random.seed(3)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(pred_masks[0], plt.gca(), borders=True)
    show_points(center_point, center_label, plt.gca())
    plt.title(f'Best Mask, Score: {scores[0]:.3f}', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(sample_dir, f'{img_name}_best_mask.png'),
                bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

    print(f'  Saved results to {sample_dir}')

# ========================
# Part 2: Video Tracking
# ========================
print('\n' + '=' * 60)
print('Part 2: Video Tracking on Dataset100 Videos')
print('=' * 60)

video_output_dir = os.path.join(OUTPUT_BASE, 'video_results')
os.makedirs(video_output_dir, exist_ok=True)

del sam2, mask_generator, predictor
torch.cuda.empty_cache()

video_predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device)

cmap = plt.get_cmap('tab10')

for video_file in VIDEO_SAMPLES:
    video_name = os.path.splitext(video_file)[0]
    sample_dir = os.path.join(video_output_dir, video_name)
    frames_dir = os.path.join(sample_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    print(f'\nProcessing video: {video_file}')
    video_path = os.path.join(VIDEO_DIR, video_file)

    # Extract frames
    existing_frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if len(existing_frames) < 5:
        cmd = f'ffmpeg -i "{video_path}" -q:v 2 -start_number 0 "{frames_dir}/%05d.jpg" -y'
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print('  Frame extraction complete')

    frame_names = sorted(
        [p for p in os.listdir(frames_dir) if p.endswith(('.jpg', '.jpeg'))],
        key=lambda p: int(os.path.splitext(p)[0]))
    print(f'  Total frames: {len(frame_names)}')

    # Initialize predictor
    inference_state = video_predictor.init_state(video_path=frames_dir)

    first_frame = Image.open(os.path.join(frames_dir, frame_names[0]))
    fw, fh = first_frame.size
    center_x, center_y = fw // 2, fh // 2

    points = np.array([[center_x, center_y]], dtype=np.float32)
    labels = np.array([1], np.int32)

    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    print(f'  Added center point prompt at ({center_x}, {center_y})')

    # Propagate
    video_segments = {}
    for out_frame_idx, out_obj_ids_prop, out_mask_logits_prop in \
            video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits_prop[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids_prop)
        }
    print(f'  Propagation complete: {len(video_segments)} frames')

    # Save first frame visualization
    plt.figure(figsize=(9, 6))
    plt.imshow(np.array(first_frame))
    plt.scatter(center_x, center_y, color='green', marker='*',
                s=375, edgecolor='white', linewidth=1.25)
    if 0 in video_segments:
        for oid, mask in video_segments[0].items():
            show_mask(mask.squeeze().astype(np.uint8), plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f'{video_name}_frame0_prompt.png'),
                bbox_inches='tight', dpi=150)
    plt.close()

    # Create output video
    sample_frame = cv2.imread(os.path.join(frames_dir, frame_names[0]))
    frame_h, frame_w = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        os.path.join(sample_dir, f'{video_name}_tracking.mp4'),
        fourcc, 15.0, (frame_w, frame_h))

    for frame_idx in range(len(frame_names)):
        frame = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
        if frame_idx in video_segments:
            for oid, mask in video_segments[frame_idx].items():
                m = mask.squeeze()
                color = np.array(cmap(oid % 10)[:3]) * 255
                color = color[::-1]
                overlay = frame.copy()
                overlay[m] = overlay[m] * 0.4 + np.array(color) * 0.6
                frame = overlay.astype(np.uint8)
                contours, _ = cv2.findContours(
                    m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)
        out_video.write(frame)
    out_video.release()

    # Reset state for next video
    video_predictor.reset_state(inference_state)
    torch.cuda.empty_cache()

    print(f'  Saved results to {sample_dir}')

print('\n' + '=' * 60)
print(f'All dataset100 results saved to: {OUTPUT_BASE}')
print('=' * 60)
