"""
SAM2 Image Inference Script
Runs automatic mask generation on all images and saves results.
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

MAX_IMAGE_SIZE = 2048  # Resize images larger than this to prevent OOM

def resize_if_needed(image, max_size=MAX_IMAGE_SIZE):
    """Resize image if any dimension exceeds max_size, keeping aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"  Resizing from {w}x{h} to {new_w}x{new_h}")
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Add project to path
SAM2_ROOT = "/data/liuluyan/SAM2/sam2"
sys.path.insert(0, SAM2_ROOT)
os.chdir(SAM2_ROOT)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Paths
CHECKPOINT = "/data/liuluyan/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMAGE_DIR = os.path.join(SAM2_ROOT, "notebooks/images")
OUTPUT_DIR = "/data/liuluyan/SAM2/results/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
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
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# ========================
# Part 1: Automatic Mask Generation
# ========================
print("\n" + "="*60)
print("Part 1: Automatic Mask Generation")
print("="*60)

sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} images: {image_files}")

for img_file in image_files:
    print(f"\nProcessing: {img_file}")
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = np.array(Image.open(img_path).convert("RGB"))
    image = resize_if_needed(image)

    torch.cuda.empty_cache()
    masks = mask_generator.generate(image)
    print(f"  Generated {len(masks)} masks")

    # Save auto-mask result
    np.random.seed(3)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.tight_layout()
    out_name = os.path.splitext(img_file)[0] + "_auto_masks.png"
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"  Saved: {out_name}")

# ========================
# Part 2: Point/Box Prompt Prediction
# ========================
print("\n" + "="*60)
print("Part 2: Point Prompt Prediction (center point)")
print("="*60)

predictor = SAM2ImagePredictor(sam2)

for img_file in image_files:
    print(f"\nProcessing: {img_file}")
    img_path = os.path.join(IMAGE_DIR, img_file)
    image = np.array(Image.open(img_path).convert("RGB"))
    image = resize_if_needed(image)
    h, w = image.shape[:2]

    torch.cuda.empty_cache()
    predictor.set_image(image)

    # Use center point as prompt
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    # Save top 3 masks
    for i, (mask, score) in enumerate(zip(masks[:3], scores[:3])):
        np.random.seed(3)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=True)
        show_points(center_point, center_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        out_name = os.path.splitext(img_file)[0] + f"_point_mask{i+1}_score{score:.3f}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        print(f"  Saved: {out_name}")

print("\n" + "="*60)
print(f"All image results saved to: {OUTPUT_DIR}")
print("="*60)
