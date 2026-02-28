import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_image, predict
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.misc import clean_state_dict
from grounding_dino.groundingdino.util.slconfig import SLConfig
from utils.video_utils import create_video_from_images

# Step 1: Environment settings
device = "cuda"

# Init SAM2 models
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading SAM2 video predictor...")
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
print("Loading SAM2 image predictor...")
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# Init Grounding DINO (local, with local bert-base-uncased)
print("Loading Grounding DINO (local)...")
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
args = SLConfig.fromfile(GROUNDING_DINO_CONFIG)
args.device = device
args.text_encoder_type = "./bert-base-uncased"
grounding_model = build_model(args)
checkpoint = torch.load(GROUNDING_DINO_CHECKPOINT, map_location="cpu")
grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
grounding_model.eval()

# Setup video
text = "stuffed toy. plush rabbit."
video_dir = "notebooks/videos/bunny_frames"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
print(f"Total frames: {len(frame_names)}")

# Init video predictor state
print("Initializing video inference state...")
inference_state = video_predictor.init_state(video_path=video_dir)

ann_frame_idx = 0

# Step 2: Detect objects in first frame using local Grounding DINO
img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
image_source, image_transformed = load_image(img_path)

print("Running Grounding DINO on first frame...")
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image_transformed,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=device
)

# Convert boxes from cxcywh normalized to xyxy pixel coordinates
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

OBJECTS = labels
print(f"Detected {len(input_boxes)} objects: {list(zip(OBJECTS, confidences.numpy()))}")

if len(input_boxes) == 0:
    print("No objects detected! Try adjusting the text prompt or threshold.")
    exit(1)

# Enable autocast AFTER Grounding DINO inference (as in local demo)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Segment with SAM2 image predictor
print("Segmenting objects in first frame...")
image_predictor.set_image(image_source)

masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)

# Step 3: Register masks to video predictor (mask prompt is most stable)
print("Registering masks to video predictor...")
for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=object_id,
        mask=mask
    )

# Step 4: Propagate masks across all frames
print("Propagating masks across video...")
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
print(f"Propagation done. Got masks for {len(video_segments)} frames.")

# Step 5: Visualize and save
save_dir = "./notebooks/videos/bunny_tracking_results"
os.makedirs(save_dir, exist_ok=True)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

print("Rendering annotated frames...")
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks_list = list(segments.values())
    masks_arr = np.concatenate(masks_list, axis=0)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks_arr),
        mask=masks_arr,
        class_id=np.array(object_ids, dtype=np.int32),
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections,
        labels=[ID_TO_OBJECTS[i] for i in object_ids]
    )

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

# Step 6: Create output video
output_video_path = "./notebooks/videos/bunny_tracking_result.mp4"
print(f"Creating output video at {output_video_path}...")
create_video_from_images(save_dir, output_video_path, frame_rate=24)

print("Done!")
