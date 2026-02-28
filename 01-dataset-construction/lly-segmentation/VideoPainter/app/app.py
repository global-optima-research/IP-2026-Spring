import os
# Set Gradio temp directory via environment variable
GRADIO_TEMP_DIR = "./tmp_gradio"
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.makedirs(f"{GRADIO_TEMP_DIR}/track", exist_ok=True)
os.makedirs(f"{GRADIO_TEMP_DIR}/inpaint", exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR
import warnings
warnings.filterwarnings("ignore")
import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import scipy
from collections import OrderedDict
import requests
import json
import torchvision
import torch
import psutil
from omegaconf import OmegaConf
import time
from PIL import Image
from openai import OpenAI

from decord import VideoReader

from utils import load_model, generate_frames

from sam2.build_sam import build_sam2_video_predictor

vlm_model = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--inpainting_branch", type=str, default="")
parser.add_argument("--id_adapter", type=str, default="")
parser.add_argument("--img_inpainting_model", type=str, default="../")
args = parser.parse_args()

sam2_checkpoint = "../ckpt/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
print(f"Build SAM2 predictor done!")

validation_pipeline, validation_pipeline_img = load_model(
        model_path=args.model_path,
        inpainting_branch=args.inpainting_branch,
        id_adapter=args.id_adapter,
        img_inpainting_model=args.img_inpainting_model
    )

print(f"Load model done!")

# Add after imports
EXAMPLES = [
    # objects
    [
        "./assets/pexels/pexels/raw_video/000000001/000000001523_856207_seg_350_548.mp4",  # video_input
        "A white ferry with red and blue accents, named 'COLONIA', cruises on a calm river, its two-tiered structure featuring large windows and a German flag, against a backdrop of modern and traditional buildings. As it moves, the ferry, now identified with 'COLONIA' and 'Rundfahrt' text, continues its journey under an overcast sky, with a stone wall and a mix of modern and historical architecture in the background. The scene shifts to show the ferry with 'COLONIA' and 'Rundfahrt' text, cruising on a river that reflects the sky, with a stone wall and a mix of modern and traditional buildings, including a Gothic church, under a clear sky.",  # video caption
        "White and red passenger ferry boat labeled 'COLONIA 6' with multiple windows, life buoys, and upper deck seating.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[320, 240]], [1]],  
    ],
    [
        "./assets/pexels/pexels/raw_video/000000088/000000088909_6188648_seg_580_752.mp4",  # video_input
        "A bustling city street at night is illuminated by festive lights, with a red double-decker bus marked 'Park Lane' and the number 59 leading a procession of vehicles. The wet pavement reflects the glow from street lamps and the bus's headlights, while pedestrians walk along sidewalks, some pausing to admire the holiday decorations. The scene is set against a backdrop of modern buildings, and a digital sign reading 'HELLO' adds a cheerful touch. As the bus continues, it is surrounded by a mix of vehicles and pedestrians, with the festive atmosphere highlighted by the 'HELLO' sign and the soft glow of street lamps.",  # video caption
        "The rear of a black car with illuminated red tail lights and a visible license plate.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[200, 400]], [1]],  
        
    ],
    # landscape
    [
        "./assets/pexels/pexels/raw_video/000000246/000000246916_10020471_seg_290_481.mp4",  # video_input
        "A tranquil sea under a vast sky is showcased, with the sun's rays piercing through cumulus clouds, creating a radiant glow on the water. Initially, the scene is devoid of human presence, emphasizing the natural beauty and serenity of the seascape. As time passes, the sun's position shifts slightly, casting a brilliant light across the sea and sky, with the clouds' textures and the sea's gentle waves highlighted. The consistent absence of human activity throughout maintains the focus on the natural beauty and serenity of the seascape.",  # video caption
        "Waves on a sunlit sea.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[320, 440]], [1]],  
        
    ],
    [
        "./assets/pexels/pexels/raw_video/000000000/000000000855_854996_seg_696_890.mp4",  # video_input
        "A solitary figure stands on a pebble-strewn beach at sunset, silhouetted against the setting sun. The calm sea reflects the golden light, with distant islets visible. The sky transitions from orange to blue, creating a serene atmosphere. The scene is marked by a sense of solitude and contemplation. Two seconds later, the figure remains on the beach, now with a black backpack and a white surfboard nearby, suggesting a recent or intended surf session. The tranquil sea and the sky's warm glow continue to enhance the peaceful mood.",  # video caption
        "Sunset with orange and yellow hues, silhouetted mountain landscape.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[320, 40]], [1]],  
        
    ],
    [
        "./assets/pexels/pexels/raw_video/000000001/000000001114_855455_seg_50_289.mp4",  # video_input
        "The video features a serene coastal landscape with towering palm trees and lush greenery along the shore. The ocean's gentle waves lap against the sandy beach, and a cliff adorned with shrubs and small trees slopes down to the water. A faint outline of a city skyline is visible in the distance, suggesting a nearby urban area. The sky is a soft blue with wispy clouds, contributing to the tranquil atmosphere. As time passes, the scene remains largely unchanged, maintaining its peaceful and secluded ambiance.",  # video caption
        "Ocean waves near the coastline.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[263, 159]], [1]],  
        
    ],

    [
        "./assets/yuewen_dog_king.mp4",  # video_input
        "In a grand, opulent throne room adorned with golden chandeliers and rich red drapes, a majestic dog king sits regally on an ornate throne. The dog, with a luxurious, flowing mane and piercing green eyes, wears a royal crown encrusted with jewels and a velvet cape trimmed with ermine. The throne, intricately carved with feline motifs, stands on a raised dais, surrounded by loyal subjects, including mice and birds, who gaze up in awe. The dog king's posture exudes authority and grace, as beams of sunlight filter through stained glass windows, casting a divine glow upon the scene.",  # video caption
        "A regal dog with a golden crown, white fur and a soft, pink nose.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[366, 158]], [1]],  
    ],

    [
        "./assets/yuewen_hat.mp4",  # video_input
        "A skeletal figure, adorned in a vintage top hat, stands eerily amidst a vast sea of vibrant red flowers, their petals swaying gently in the breeze. The skeleton's bony fingers clutch an ornate cane, adding an air of sophistication to its ghostly presence. The crimson blooms stretch endlessly, creating a surreal and hauntingly beautiful landscape. The sky above is a muted gray, casting a somber light over the scene, while the skeleton's empty eye sockets seem to gaze into the distance, evoking a sense of timeless mystery and melancholy.",  # video caption
        "Straw hat with red underside, purple and pink flowers, and white embroidered pattern.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[332, 48]], [1]],  
    ],
    [
        "./assets/yuewen_jobs.mp4",  # video_input
        "Jobs spoke at the press conference wearing a black shirt.",  # video caption
        "Black shirt",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[189, 328], [279, 167]], [1, 0]],  
    ],
    # 
    [
        "./assets/pexels/pexels/raw_video/000000316/000000316125_20329165_seg_290_578.mp4",  # video_input
        "The video features a serene beach at dusk with the calm sea reflecting the pink and blue hues of the twilight sky. A tree with delicate branches stands in the foreground, its silhouette contrasting with the soft blues and pinks of the evening. As time passes, the scene remains tranquil, with the tree's branches reaching out over the sandy shore. The sky transitions from deep blue to soft pink, and the absence of people or wildlife emphasizes the stillness and natural beauty of the coastal landscape.",  # video caption
        "Tree branches with thin, elongated leaves against a blue sky.",  # first frame caption
        "Positive",  # point_prompt
        "Inpaint",  # model_type
        "",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[280, 137]], [1]],  
        
    ],
    # change
    [
        "./assets/pexels/pexels/raw_video/000000208/000000208144_8641513_seg_350_553.mp4",  # video_input
        "A man and Anne Hathaway share an intimate moment, foreheads touching and eyes locked in a silent exchange that speaks volumes of their connection. Anne Hathaway, with her blonde hair tied back and wearing a black leather jacket over a striped shirt, and the man, sporting curly black hair and a mustache, dressed in a white shirt with a red and beige pattern, are illuminated by soft lighting that casts gentle shadows on their faces. This lighting suggests it might be early morning or late evening, adding to the romantic atmosphere of their encounter. The background, though blurred, hints at an urban setting, possibly near a car.",  # video caption
        "Anne Hathaway's profile with soft features, fair skin, and natural makeup.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (change)",  # model_type
        "Change the woman to Anne Hathaway",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[261, 250]], [1]],  
    ],
    [
        "./assets/pexels/pexels/raw_video/000000112/000000112319_6693826_seg_720_901.mp4",  # video_input
        "A young Black woman with voluminous, curly hair, wearing a white bathrobe, stands in a serene bathroom holding a glass tray with spa essentials. The tray includes amber glass bottles with black pump dispensers, a white candle, and a bowl of white flowers, all contributing to a tranquil atmosphere. The room features mosaic tiles, a classic clawfoot bathtub, and a vase of white flowers, enhancing the spa-like setting. Her calm and confident demeanor suggests she is a professional or host, ready to provide a relaxing experience.",  # video caption
        "Transparent glass tray with smooth, reflective surface and rounded edges.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Change)",  # model_type
        "Change the material of the tray to glass.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[320, 440]], [1]],  
    ],
    [
        "./assets/pexels/pexels/raw_video/000000134/000000134923_7169873_seg_92_228.mp4",  # video_input
        "A woman with bright blue hair in a grey kimono is seated at a wooden table, engaged in a traditional Japanese tea ceremony. She is focused on pouring green tea from a ceramic jug into a small cup on a bamboo tray, surrounded by a serene setting that includes a blue and white porcelain teapot, a silver tea scoop, and various other tea utensils. The room is filled with natural light, enhancing the tranquil atmosphere. A vintage desk and chair, along with a book and writing implements, suggest a space dedicated to cultural practices. The scene is bathed in natural light, highlighting the warmth and tranquility of the setting.",  # video caption
        "Bright blue, wavy hair cascading over the shoulder, maintaining natural gloss and texture.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Change)",  # model_type
        "Change the hair color to bright blue.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[576, 136], [576, 136]], [1, 1]],  
    ],
    # swap
    [
        "./assets/pexels/pexels/raw_video/000000008/000000008483_3173833_seg_174_462.mp4",  # video_input
        "A waterfall sits atop a white base with classical columns, surrounded by a dense forest. A dirt path leads to the base, suggesting human presence. As the view continues, the Bwaterfall is seen atop a white building with a red-tiled roof, set against a backdrop of lush greenery, a partly cloudy sky, and a cascading waterfall. The scene is serene, with the waterfall, and the surrounding forest, sky, and waterfall enhancing the tranquil atmosphere.",  # video caption
        "Cascading waterfall with lush greenery and mist in the background.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Swap)",  # model_type
        "Swap the statue with a waterfall.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[415, 177]], [1]],  
    ],
    [
        "./assets/pexels/pexels/raw_video/000000162/000000162292_7691629_seg_300_498.mp4",  # video_input
        "Four professionals, two men and two women, are engaged in a collaborative meeting in an office. Initially, a man in a navy blazer and a woman in a brown checkered blazer are seen discussing a laptop screen, with another man in a white sweater and a woman in a black and white patterned coat observing. In the background, there is a large chalkboard filled with complex equations. As time progresses, the group's dynamic shifts slightly; the man in the navy blazer now points at the laptop screen, indicating a discussion or presentation, while the woman in the brown blazer smiles and engages with the content. The man in the white sweater gestures thoughtfully, and the woman in the black and white patterned coat appears contemplative, hand to cheek.",  # video caption
        "Large green chalkboard with white complex mathematical equations and diagrams.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Swap)",  # model_type
        "Swap the background with a large chalkboard with complex equations.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[479, 133], [518, 58]], [1, 1]],  
    ],
    [
        "./assets/yuewen_dog_king.mp4",  # video_input
        "In a grand, opulent throne room adorned with golden chandeliers and rich red drapes, a majestic cat king sits regally on an ornate throne. The cat, with a luxurious, flowing mane and piercing green eyes, wears a royal crown encrusted with jewels and a velvet cape trimmed with ermine. The throne, intricately carved with feline motifs, stands on a raised dais, surrounded by loyal subjects, including mice and birds, who gaze up in awe. The cat king's posture exudes authority and grace, as beams of sunlight filter through stained glass windows, casting a divine glow upon the scene.",  # video caption
        "A regal cat with a golden crown, white fur and a soft, pink nose.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Swap)",  # model_type
        "Swap the dog into a cat.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[366, 158]], [1]],  
    ],
    # add
    [
        "./assets/pexels/pexels/raw_video/000000189/000000189013_8285848_seg_0_298.mp4",  # video_input
        "A drone glides over a densely populated cityscape, showcasing a mix of modern skyscrapers and traditional buildings under an overcast sky. Hovering above the buildings, a giant futuristic spaceship adds an intriguing, otherworldly element to the scene. The city's architecture varies from sleek glass facades to aged structures with red-tiled roofs, creating a serene yet somber mood. As the drone moves, the urban environment's stillness is highlighted, with no visible human activity. The hazy sky enhances the mood of quietude, while the drone's flight and the presence of the spaceship add dynamic elements to the otherwise static scene.",  # video caption
        "A massive metallic spaceship with glowing blue lights hovering above the skyline.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Add)",  # model_type
        "Add a giant futuristic spaceship hovering above the buildings.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        16,  # dilate_size
        [[[327, 80]], [1]],  
    ],
    [
        "./assets/pexels/pexels/raw_video/000000015/000000015767_3982751_seg_0_288.mp4",  # video_input
        "Two young women, dressed in cozy winter attire, stand together in a tranquil snowy winter landscape, engaging in the warmth of sharing a hot drink. Initially, they are seen in a moment of camaraderie, with one woman sipping from a mug, another pouring from a thermos. As moments pass, their activities slightly change; one examines a small metal cup, and the other holds a stainless steel thermos. Their expressions of contentment and enjoyment, along with the soft natural lighting, highlight the serene atmosphere of their gathering.",  # video caption
        "Snow-covered trees and ground with a soft, overcast winter sky.",  # first frame caption
        "Positive",  # point_prompt
        "Edit (Remove)",  # model_type
        "Remove the woman.",  # editing_instruction (empty for Inpaint type)
        42,  # seed_param
        6.0,  # cfg_scale
        0,  # dilate_size
        [[[148, 157], [146, 68], [107, 277], [148, 164]], [1, 1, 1, 1]],  
    ],
]

class StatusMessage:
    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"
    SUCCESS = "Success"

def create_status(message, status_type=StatusMessage.INFO):
    timestamp = time.strftime("%H:%M:%S")
    return [("", ""), (f"[{timestamp}]: {message}\n", status_type)]

def update_status(previous_status, new_message, status_type=StatusMessage.INFO):
    timestamp = time.strftime("%H:%M:%S")
    history = previous_status[-3:]  
    history.append((f"[{timestamp}]: {new_message}\n", status_type))
    return [("", "")] + history

def init_state(
        offload_video_to_cpu=False,
        offload_state_to_cpu=False
        ):
    inference_state = {}
    inference_state["images"] = torch.zeros([1,3,100,100])
    inference_state["num_frames"] = 1
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["video_height"] = 100
    inference_state["video_width"] = 100
    inference_state["device"] = torch.device("cuda")
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = torch.device("cuda")
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state = gr.State(inference_state)
    return inference_state

# convert points input to prompt state
def get_prompt(click_state, click_input):
    print(f"click_input: {click_input}")
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    video_path = video_input
    frames = []
    user_name = time.time()
    vr = VideoReader(video_path)
    original_fps = vr.get_avg_fps()
    
    # If fps > 8, downsample frames to 8fps
    if original_fps > 8:
        total_frames = len(vr)
        sample_interval = max(1, int(original_fps / 8))
        frame_indices = list(range(0, total_frames, sample_interval))
        frames = vr.get_batch(frame_indices).asnumpy()
    else:
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    
    # Take only first 49 frames
    frames = frames[:49]
    
    # Resize all frames to 480x720
    resized_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (720, 480))
        resized_frames.append(resized_frame)
    frames = np.array(resized_frames)

    init_start = time.time() 
    inference_state = predictor.init_state(images=frames, offload_video_to_cpu=True, async_loading_frames=True)
    init_time = time.time() - init_start
    print(f"Inference state initialization took {init_time:.2f}s")
    
    fps = 8
    image_size = (frames[0].shape[0],frames[0].shape[1])
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "ann_obj_id": 0
        }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)

    video_input = generate_video_from_frames(frames, output_path=f"{GRADIO_TEMP_DIR}/inpaint/original_{video_state['video_name']}", fps=video_state["fps"])

    return gr.update(visible=True), \
    gr.update(visible=True), \
    inference_state, \
    video_state, \
    video_info, \
    video_state["origin_images"][0], \
    gr.update(visible=False, maximum=len(frames), value=1, interactive=True), \
    gr.update(visible=False, maximum=len(frames), value=len(frames), interactive=True), \
    gr.update(visible=True, interactive=True), \
    gr.update(visible=True, interactive=True), \
    gr.update(visible=True, interactive=True), \
    gr.update(visible=True), \
    gr.update(visible=True, interactive=False), \
    create_status("Upload video already. Try click the image for adding targets to track and inpaint.", StatusMessage.SUCCESS), \
    video_input
# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, previous_status):
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider
    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, \
           update_status(previous_status, f"Set the tracking start at frame {image_selection_slider}. Try click image and add mask for tracking.", StatusMessage.INFO)

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state, previous_status):
    interactive_state["track_end_number"] = track_pause_number_slider
    return video_state["painted_images"][track_pause_number_slider], interactive_state, \
           update_status(previous_status, f"Set the tracking finish at frame {track_pause_number_slider}. Try click image and add mask for tracking", StatusMessage.INFO)


# use sam to get the mask
def sam_refine(inference_state, video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData, previous_status):
    ann_obj_id = 0
    ann_frame_idx = video_state["select_frame_number"]

    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    print(f"point_prompt: {point_prompt}, click_state: {click_state}")

    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    points=np.array(prompt["input_point"])
    labels=np.array(prompt["input_label"])
    height, width = video_state["origin_images"][0].shape[0:2]

    for i in range(len(points)):
        points[i,0] = int(points[i,0])
        points[i,1] = int(points[i,1])
    print(f"func-sam_refine: {points}, {labels}")
    frame_idx, obj_ids, mask = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    mask_ = mask.cpu().squeeze().detach().numpy()
    print(mask.shape)
    mask_[mask_<=0] = 0
    mask_[mask_>0] = 1
    org_image = video_state["origin_images"][video_state["select_frame_number"]]
    mask_ = cv2.resize(mask_, (width, height))
    mask_ = mask_[:, :, None]
    mask_[mask_>0.5] = 1
    mask_[mask_<=0.5] = 0
    color = 63*np.ones((height, width, 3)) * np.array([[[np.random.randint(5),np.random.randint(5),np.random.randint(5)]]])
    painted_image = np.uint8((1-0.5*mask_)*org_image + 0.5*mask_*color)

    video_state["masks"][video_state["select_frame_number"]] = mask_
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    return painted_image, video_state, interactive_state, \
           update_status(previous_status, "Use SAM2 for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Tracking button when you are satisfied with the segment", StatusMessage.SUCCESS)


def clear_click(inference_state, video_state, click_state, previous_status):
    predictor.reset_state(inference_state)
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return inference_state, template_frame, click_state, \
           update_status(previous_status, "Clear points history and reinitialize the video status.", StatusMessage.INFO)

# tracking vos
def vos_tracking_video(inference_state, video_state, interactive_state, previous_status):
    height, width = video_state["origin_images"][0].shape[0:2]

    masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = np.zeros([480, 720, 1])
        for i in range(len(out_mask_logits)):
            out_mask = out_mask_logits[i].cpu().squeeze().detach().numpy()
            out_mask[out_mask>0] = 1
            out_mask[out_mask<=0] = 0
            out_mask = out_mask[:,:,None]
            mask += out_mask
        mask = cv2.resize(mask, (width, height))
        mask = mask[:,:,None]
        mask[mask>0.5] = 1
        mask[mask<1] = 0
        mask = scipy.ndimage.binary_dilation(mask, iterations=6)
        masks.append(mask)
    masks = np.array(masks)

    painted_images = None
    if interactive_state["track_end_number"]:
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        org_images = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
        color = 255*np.ones((1, org_images.shape[-3], org_images.shape[-2], 3)) * np.array([[[[0,1,1]]]])
        painted_images = np.uint8((1-0.5*masks)*org_images + 0.5*masks*color)
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"] = masks
        org_images = video_state["origin_images"]
        color = 255*np.ones((1, org_images.shape[-3], org_images.shape[-2], 3)) * np.array([[[[0,1,1]]]])
        painted_images = np.uint8((1-0.5*masks)*org_images + 0.5*masks*color)
        video_state["painted_images"] = painted_images
    if painted_images is not None:
        video_output = generate_video_from_frames(video_state["painted_images"], output_path=f"{GRADIO_TEMP_DIR}/track/{video_state['video_name']}", fps=video_state["fps"])
    else:
        raise ValueError("No tracking images found")
    interactive_state["inference_times"] += 1
    
    print(f"func-vos_tracking_video: {video_output}")
    return (
        inference_state, 
        video_output, 
        video_state, 
        interactive_state, 
        update_status(previous_status, "Track the selected target region, and then you can use the masks for inpainting.", StatusMessage.SUCCESS),
        gr.Button.update(interactive=True),  # inpaint_video_predict_button
        gr.Button.update(interactive=True),  # enhance_button
        gr.Button.update(interactive=True),  # enhance_target_region_frame1_button
        gr.Button.update(interactive=True),  # enhance_editing_instruction_button
        gr.Accordion(open=True)  # Add this line to open the accordion
    )

# inpaint
def inpaint_video(video_state, video_caption, target_region_frame1_caption, interactive_state, previous_status, seed_param, cfg_scale, dilate_size):
    # Convert seed_param directly since it's already a number
    seed = int(seed_param) if int(seed_param) >= 0 else np.random.randint(0, 2**32 - 1)
    
    validation_images = video_state["origin_images"][list(range(0, len(video_state["origin_images"]),1))]
    validation_masks = video_state["masks"][list(range(0, len(video_state["origin_images"]),1))]
    print(f"func-inpaint_video-before: {np.array(validation_images).shape}, {np.array(validation_images).min()}, {np.array(validation_images).max()}, {np.array(validation_masks).shape}, {np.array(validation_masks).min()}, {np.array(validation_masks).max()}")
    
    validation_masks = [np.squeeze(mask) for mask in validation_masks]  
    validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]  
    
    validation_masks = [np.stack([m, m, m], axis=-1) for m in validation_masks] 
    
    validation_images = [Image.fromarray(np.uint8(img)).convert('RGB') for img in validation_images]
    validation_masks = [Image.fromarray(np.uint8(mask)).convert('RGB') for mask in validation_masks]
    
    validation_images = [img.resize((720, 480)) for img in validation_images]
    validation_masks = [mask.resize((720, 480)) for mask in validation_masks]
    # validation_masks[0] = Image.fromarray(np.zeros_like(np.array(validation_masks[0]))).convert("RGB")

    # (25, 360, 640, 3) 
    # (25, 360, 640, 1)
    print(f"func-inpaint_video-after: {np.array(validation_images).shape}, {np.array(validation_images).min()}, {np.array(validation_images).max()}, {np.array(validation_masks).shape}, {np.array(validation_masks).min()}, {np.array(validation_masks).max()}")


    print(str(video_caption))
    print(str(target_region_frame1_caption))

    images = generate_frames(
        images=validation_images, 
        masks=validation_masks, 
        pipe=validation_pipeline, 
        pipe_img_inpainting=validation_pipeline_img, 
        prompt=str(video_caption), 
        image_inpainting_prompt=str(target_region_frame1_caption),
        seed=seed,  # Use the converted seed value
        cfg_scale=float(cfg_scale),
        dilate_size=int(dilate_size)
    )
    images = (images * 255).astype(np.uint8)

    video_output = generate_video_from_frames(images, output_path=f"{GRADIO_TEMP_DIR}/inpaint/{video_state['video_name']}", fps=8)
    print(f"func-inpaint_video: {video_output}")
    return video_output, update_status(previous_status, "Inpainting the selected target region based on the video caption and the first frame caption.", StatusMessage.SUCCESS)


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=8):
    # print(f"func-generate_video_from_frames: {frames.shape}, {frames.dtype}")
    #  torch.Size([49, 480, 720, 3]), torch.uint8
    frames = torch.from_numpy(np.asarray(frames))
    frames = frames.to(torch.uint8) 
    # print(f"func-generate_video_from_frames: {frames.shape}, {frames.dtype}")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def echo_text(text1, text2):
    print(f"你输入的文本是: pos {text1}, neg {text2}")
    return f"你输入的文本是: pos {text1}, neg {text2}"

def process_example(video_input, video_caption, target_region_frame1_caption, prompt, click_state):
    print(f"func-process_example-video_input: {video_input}")
    if video_input is None or video_input == "":
        return (
            gr.update(value=""), 
            gr.update(value=""),
            init_state(), 
            {                       # video_state
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "inpaint_masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 8,
                "ann_obj_id": 0
            },
            "", 
            None,
            gr.update(value=1, visible=False, interactive=False), 
            gr.update(value=1, visible=False, interactive=False),
            gr.update(value="Positive", interactive=False), 
            gr.update(visible=True, interactive=False),
            gr.update(visible=True, interactive=False),
            gr.update(value=None), 
            gr.update(visible=True, interactive=False),
            create_status("Reset complete. Ready for new input.", StatusMessage.INFO),
            gr.update(value=None),  # video_input
        )
    
    print(f"Begin function process_example!!")
    video_state = gr.State({
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 8,
        "ann_obj_id": 0
    })
    results = get_frames_from_video(video_input, video_state)
    print(f"func-process_example-results: {len(results)}")
    
    if click_state[0] and click_state[1]:
        
        print("Example detected, executing sam_refine")
        (video_caption, 
        target_region_frame1_caption, inference_state, video_state, video_info, template_frame,
         image_selection_slider, track_pause_number_slider, point_prompt, clear_button,
         tracking_button, video_output, inpaint_button, run_status, video_input) = results
        
        class MockEvent:
            def __init__(self, points, point_idx=0):
                self.index = points[point_idx]
        
        for i_click in range(len(click_state[0])):
            evt = MockEvent(click_state[0], i_click)
            prompt = "Positive" if click_state[1][i_click] == 1 else "Negative"
            
            template_frame, video_state, interactive_state, run_status = sam_refine(
                inference_state=inference_state,
                video_state=video_state,
                point_prompt=prompt,
                click_state=click_state,
                interactive_state = {
                    "inference_times": 0,
                    "negative_click_times" : 0,
                    "positive_click_times": 0,
                    "mask_save": False,
                    "multi_mask": {
                        "mask_names": [],
                        "masks": []
                    },
                    "track_end_number": None,
                },
                evt=evt,
                previous_status=run_status
            )
            
        return (
            video_caption,
            target_region_frame1_caption, 
            inference_state, 
            video_state, 
            video_info, 
            template_frame,
            image_selection_slider, 
            track_pause_number_slider, 
            point_prompt, 
            clear_button,
            tracking_button, 
            video_output, 
            inpaint_button, 
            run_status, 
            video_input
        )
    
    return results

def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = vlm_model
    text = prompt.strip()

    sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:
You will only ever output a single video description per user request.
When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="gpt-4o",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt

def convert_prompt_target_region_frame1(prompt, video_state, retry_times = 3) -> str:

    validation_images = video_state["origin_images"][list(range(0, len(video_state["origin_images"]),1))]
    validation_masks = video_state["masks"][list(range(0, len(video_state["origin_images"]),1))]
    
    
    validation_masks = [np.squeeze(mask) for mask in validation_masks]  
    validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]  
    
    validation_masks = [np.stack([m, m, m], axis=-1) for m in validation_masks]  
    
    validation_images = [Image.fromarray(np.uint8(img)).convert('RGB') for img in validation_images]
    validation_masks = [Image.fromarray(np.uint8(mask)).convert('RGB') for mask in validation_masks]
    
    validation_images = [img.resize((720, 480)) for img in validation_images]
    validation_masks = [mask.resize((720, 480)) for mask in validation_masks]

    masked_image = np.where(np.array(validation_masks[0]) == 255, np.array(validation_images[0]), 0)
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    masked_image.save(f"{GRADIO_TEMP_DIR}/inpaint/caption_masked_image.png")

    import base64
    from io import BytesIO
    buffered = BytesIO()
    masked_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    system_prompt ="""You are an expert in visual scene understanding and image editing. Your task is to:
    1. Analyze the visible (unmasked) elements in the provided image
    2. Generate a precise description of how the target region content should look"""
    
    user_prompt = f"""Based on the visible elements in this image, generate a concise description for image inpainting. 
    Requirements:
    1. Focus ONLY on describing the target edited region (masked area)
    2. Maintain visual consistency with unmasked elements
    3. Use specific, descriptive language (colors, textures, shapes)
    4. Avoid mentioning the mask or editing process itself
    5. Keep the description under 20 words

    Example:
    Good: "Vibrant orange and pink sunset sky with scattered wispy clouds", "White and red passenger ferry boat labeled 'COLONIA 6' with multiple windows, life buoys, and upper deck seating.", "The rear of a black car with illuminated red tail lights and a visible license plate.
    Bad: "A masked region showing sunset colors"

    Only return the description, no explanations or additional text."""


    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = vlm_model

    for i in range(retry_times):
        response = vlm_model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,  # Add some creativity while maintaining coherence
            max_tokens=50     # Limit response length for conciseness
        )
        if response.choices:
            print(f"convert_prompt_target_region_frame1-success: {response.choices[0].message.content}")
            return response.choices[0].message.content
    print(f"convert_prompt_target_region_frame1-failed: {prompt}")
    return prompt

def convert_prompt_editing_instruction(
    editing_instruction, 
    video_caption, 
    target_region_frame1_caption, 
    video_state, 
    retry_times = 3
    ) -> str:

    if not os.environ.get("OPENAI_API_KEY"):
        return video_caption, target_region_frame1_caption

    validation_images = video_state["origin_images"][list(range(0, len(video_state["origin_images"]),1))]
    validation_masks = video_state["masks"][list(range(0, len(video_state["origin_images"]),1))]
    
    validation_masks = [np.squeeze(mask) for mask in validation_masks] 
    validation_masks = [(mask > 0).astype(np.uint8) * 255 for mask in validation_masks]  
    
    validation_masks = [np.stack([m, m, m], axis=-1) for m in validation_masks]  
    
    validation_images = [Image.fromarray(np.uint8(img)).convert('RGB') for img in validation_images]
    validation_masks = [Image.fromarray(np.uint8(mask)).convert('RGB') for mask in validation_masks]
    
    validation_images = [img.resize((720, 480)) for img in validation_images]
    validation_masks = [mask.resize((720, 480)) for mask in validation_masks]

    masked_image = np.where(np.array(validation_masks[0]) == 255, np.array(validation_images[0]), 0)
    masked_image = Image.fromarray(masked_image.astype(np.uint8))
    masked_image.save(f"{GRADIO_TEMP_DIR}/inpaint/caption_masked_image.png")

    system_prompt = """You are a video description editing expert. You need to edit the original video description based on the user's editing instruction.
    Requirements:
    1. Keep the description coherent and natural
    2. Ensure the modification conforms to the user's instruction, strengthen the editing objects corresponding to the editing commands
    3. Retain important details in the original description that are not affected by the instruction
    """
    
    user_prompt = f"""Original video description: {video_caption}
    Editing instruction: {editing_instruction}
    Please edit the video description based on the editing instruction. Only return the edited description, no other words."""

    response = vlm_model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    video_caption = response.choices[0].message.content


    import base64
    from io import BytesIO
    buffered = BytesIO()
    masked_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    system_prompt ="""You are an expert in visual scene understanding and image editing. Your task is to:
    1. Analyze the visible (unmasked) elements in the provided image
    2. Consider the editing instruction carefully
    3. Generate a precise description of how the edited content should look, assuming the editing instruction is already applied to the original image"""
    
    user_prompt = f"""Based on the visible elements in this image and the editing instruction, generate a concise description for image inpainting. 
    Original target object caption: {target_region_frame1_caption}
    Editing instruction: {editing_instruction}
    Requirements:
    1. Focus ONLY on describing the target edited region (masked area), assuming the editing instruction is already applied to the original image
    2. Maintain visual consistency with unmasked elements
    3. Use specific, descriptive language (colors, textures, shapes)
    4. Avoid mentioning the mask or editing process itself
    5. Keep the description under 20 words
    6. Ensure the description aligns with the editing instruction

    Example:
    Editing instruction: "Change the sky to sunset"
    Good: "Vibrant orange and pink sunset sky with scattered wispy clouds"
    Bad: "A masked region showing sunset colors"

    Only return the description, no explanations or additional text."""

    # Call OpenAI vision API with image
    response = vlm_model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    }
                ]
            }
        ],
        temperature=0.7,  # Add some creativity while maintaining coherence
        max_tokens=50     # Limit response length for conciseness
    )

    target_region_frame1_caption = response.choices[0].message.content


    
    return video_caption, target_region_frame1_caption

def reset_all():
    return (
        gr.update(value=None),  # video_input
        gr.update(value=""),    # video_caption
        gr.update(value=""),    # target_region_frame1_caption
        init_state(),           # inference_state
        {                       # video_state
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 8,
            "ann_obj_id": 0
        },
        {                       # interactive_state
            "inference_times": 0,
            "negative_click_times": 0,
            "positive_click_times": 0,
            "mask_save": False,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        },
        [[],[]],               # click_state
        None,                  # video_output
        gr.update(visible=True, interactive=True),                  # template_frame
        "",                    # video_info
        gr.update(value=1, visible=False, interactive=False),    # image_selection_slider
        gr.update(value=1, visible=False, interactive=False),    # track_pause_number_slider
        gr.update(value="Positive", interactive=False),  # point_prompt
        gr.Button.update(interactive=False),  # clear_button_click
        gr.Button.update(interactive=False),  # tracking_video_predict_button
        gr.Button.update(interactive=False),  # inpaint_video_predict_button
        gr.Button.update(interactive=False),  # enhance_button
        gr.Button.update(interactive=False),  # enhance_target_region_frame1_button
        gr.Button.update(interactive=False),  # enhance_editing_instruction_button
        gr.Number.update(value=42),  # seed_param
        gr.Slider.update(value=6.0),  # cfg_scale
        gr.Slider.update(value=16),  # dilate_size
        create_status("Reset complete. Ready for new input.", StatusMessage.INFO)  # run_status
    )

title = """<p><h1 align="center">VideoPainter</h1></p>
    """
with gr.Blocks() as iface:
    gr.HTML(
        """
<div style="text-align: center;">
    <h1 style="text-align: center; color: #333333;">🖌️ VideoPainter</h1>
    <h3 style="text-align: center; color: #333333;">Any-length Video Inpainting and Editing with
Plug-and-Play Context Control</h3>
    <p style="text-align: center; font-weight: bold">
        <a href="https://yxbian23.github.io/project/video-painter/">🌍 Project Page</a> | 
        <a href="https://arxiv.org/abs/2503.05639">📃 ArXiv Preprint</a> | 
        <a href="https://github.com/TencentARC/VideoPainter">🧑‍💻 Github Repository</a>
    </p>
    <p style="text-align: left; font-size: 1.1em;">
    Welcome to the demo of <strong>VideoPainter</strong>. Follow the steps below to explore its capabilities:
    </p>
</div>
<div style="text-align: left; margin: 0 auto; ">
    <ol style="font-size: 1.1em;">
    <li><strong>Upload the original video (Or click a example):</strong> Our demo will automatically extract the video information, resize to 480 * 720, resample to 8fps with the first 6 seconds due to the base model limitation.</li>
    <li><strong>Click the target region you want to inpaint:</strong> Use the Positive and Negative point prompts to guide the SAM2 segmentation process.</li>
    <li><strong>Click the tracking button:</strong> The demo will track the target region and generate the tracking segmentation video.</li>
    <li><strong>Input and enhance global video caption:</strong> Input a global video caption for inpainting/editing results.</li>
    <li><strong>Input or generate target region caption:</strong> Input or automatically create a localized description for the area to be inpainted/edited (which can be automatically generated after providing the global caption or manually entered by the user).</li>
    <li><strong>Input the editing instruction (Optional):</strong> The demo will use the editing instruction to modify the video caption and the target object caption for following video editing.</li>
    <li><strong>Click the inpainting button:</strong> The demo will inpaint the target region based on the video caption, the target object caption and the editing instruction.</li>
    </ol>
    <p>
    ⏱️ <b>ZeroGPU Time Limit</b>: Hugging Face ZeroGPU has a inference time limit of 180 seconds.
    If some cases you may meet the time limit, please consider log in with a Pro account or run the gradio demo on your local machine. 
    </p>
    <p style="text-align: left; font-size: 1.1em;">🤗 Please 🌟 star our <a href="https://github.com/TencentARC/VideoPainter"> GitHub repo </a> 
    and click on the ❤️ like button above if you find our work helpful.
    <a href="https://github.com/TencentARC/VideoPainter"><img src="https://img.shields.io/github/stars/TencentARC%2FVideoPainter"/></a> </p>
</div>
    """
    )
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": False,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 8,
        "ann_obj_id": 0
        }
    )
    inference_state = init_state()
    # gr.Markdown(title)
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row():
                video_input = gr.Video(label="Original Video", visible=True)
                
            with gr.Row():
                with gr.Column(scale=3):
                    template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame", visible=True)
                with gr.Column(scale=1):
                    with gr.Accordion("Segmentation Point Prompt", open=True):
                        point_prompt = gr.Radio(
                        choices=["Positive",  "Negative"],
                        value="Positive",
                        label="Point Type",
                        interactive=False,
                        visible=True)
                        clear_button_click = gr.Button(value="Clear clicks", interactive=False, visible=True)
                        gr.Markdown(
                            "✨Positive: Include the target region. <br> ✨Negative: Exclude the target region. <br> ✨Clear: Clear all history point prompts."
                        )
                        image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False, interactive=False)
                        track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False, interactive=False)
                
            
            video_output = gr.Video(label="Generated Video", visible=True)
            with gr.Row():
                tracking_video_predict_button = gr.Button(value="Tracking", interactive=False, visible=True)
                inpaint_video_predict_button = gr.Button(value="Inpainting", interactive=False, visible=True)
                reset_button = gr.Button(value="Reset All", interactive=True, visible=True)  # Add reset button here
            


        with gr.Column():
            with gr.Accordion("Global Video Caption", open=True):
                video_caption = gr.Textbox(
                    label="Global Video Caption", 
                    placeholder="Please input the global video caption...", 
                    interactive=True, 
                    visible=True,
                    max_lines=5,  # Limit visible lines
                    show_copy_button=True  # Add copy button for convenience
                )
                with gr.Row():
                    gr.Markdown(
                        "✨Upon pressing the enhanced prompt button, we will use [GPT-4o](https://openai.com/index/hello-gpt-4o/) to polish the video caption."
                    )
                    enhance_button = gr.Button("✨ Enhance Prompt(Optional)", interactive=False)
            with gr.Accordion("Target Object Caption", open=True):
                target_region_frame1_caption = gr.Textbox(
                    label="Target Object Caption", 
                    placeholder="Please input the target object caption...", 
                    interactive=True, 
                    visible=True,
                    max_lines=5,  # Limit visible lines
                    show_copy_button=True  # Add copy button for convenience
                )
                with gr.Row():
                    gr.Markdown(
                        "✨Upon pressing the target prompt generation button, we will use [GPT-4o](https://openai.com/index/hello-gpt-4o/) to generate the target object caption."
                    )
                    enhance_target_region_frame1_button = gr.Button("✨ Target Prompt Generation (Optional)", interactive=False)

            with gr.Accordion("Editing Instruction", open=False):
                gr.Markdown(
                    "✨Upon pressing the button, we will use [GPT-4o](https://openai.com/index/hello-gpt-4o/) to modify the video caption and the target object caption based on the editing instruction."
                )
                with gr.Row():
                    editing_instruction = gr.Textbox(
                        label="User Editing Instruction", 
                        placeholder="Please input the editing instruction...", 
                        interactive=True, 
                        visible=True,
                        max_lines=5,  # Limit visible lines
                        show_copy_button=True  # Add copy button for convenience
                    )
                    enhance_editing_instruction_button = gr.Button("✨ Modify Caption(For Editing)", interactive=False)
            with gr.Accordion("Advanced Sampling Settings", open=False):
                cfg_scale = gr.Slider(
                    value=6.0,
                    label="Classifier-Free Guidance Scale",
                    minimum=1,
                    maximum=10,
                    step=0.1,
                    interactive=True
                )
                seed_param = gr.Number(label="Inference Seed (Enter a positive number, -1 for random)", interactive=True, value=42)
                dilate_size = gr.Slider(
                    value=16,
                    label="Mask Dilate Size",
                    minimum=0,
                    maximum=32,
                    step=1,
                    interactive=True
                )
                video_info = gr.Textbox(label="Video Info", visible=True, interactive=False)
                model_type = gr.Textbox(label="Type", placeholder="Please input the model type...", interactive=True, visible=False)

            # Create the accordion with a reference
            notes_accordion = gr.Accordion("Notes", open=False)
            with notes_accordion:
                gr.HTML(
                    """
<p style="font-size: 1.1em; line-height: 1.6; color: #555;">
🧐 <b>Reminder</b>:
    As a generative model, VideoPainter may occasionally produce unexpected outputs.
    Try adjusting the video caption, target object caption, editing instruction, random seed or CFG scale to explore different results.
<br>
🤔 <b>Longer Generation</b>:
    If you need longer video, you can refer to our <a href="https://github.com/TencentARC/VideoPainter">GitHub</a> and run script with your own GPU. <br>
🤗 <b>Limitation</b>:
    This is the initial beta version of VideoPainter.
    Its generalizability may be limited in certain scenarios, and artifacts can appear with large camera motions due to the current foundation model's constraints.
    We are looking for <b>collaboration opportunities from the community</b>. <br>
✨ We welcome your feedback and questions. Thank you! </p>
                    """
                )
                run_status = gr.HighlightedText(
                    value=[("", "")],
                    visible=True,
                    label="Operation Status",
                    show_label=True,
                    color_map={
                        "Success": "green",
                        "Error": "red",
                        "Warning": "orange",
                        "Info": "blue"
                    }
                )
                    

    with gr.Row():
        examples = gr.Examples(
            label="Quick Examples",
            examples=EXAMPLES,
            inputs=[
                video_input,
                video_caption,
                target_region_frame1_caption,
                point_prompt,
                model_type,
                editing_instruction,
                seed_param,
                cfg_scale,
                dilate_size,
                click_state,
            ],
            examples_per_page=20,
            cache_examples=False,
        )

    video_input.change(
        fn=process_example,
        inputs=[
            video_input,
            video_caption,
            target_region_frame1_caption,
            point_prompt,
            click_state
        ],
        outputs=[
            video_caption, 
            target_region_frame1_caption, 
            inference_state, 
            video_state, 
            video_info, 
            template_frame,
            image_selection_slider, 
            track_pause_number_slider,
            point_prompt, 
            clear_button_click,
            tracking_video_predict_button,
            video_output, 
            inpaint_video_predict_button,
            run_status,
            video_input
        ]
    )


    # second step: select images from slider
    image_selection_slider.release(
        fn=select_template,
        inputs=[image_selection_slider, video_state, interactive_state, run_status],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    track_pause_number_slider.release(
        fn=get_end_number,
        inputs=[track_pause_number_slider, video_state, interactive_state, run_status],
        outputs=[template_frame, interactive_state, run_status]
    )

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[inference_state, video_state, point_prompt, click_state, interactive_state, run_status],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[inference_state, video_state, interactive_state, run_status],
        outputs=[
            inference_state, 
            video_output, 
            video_state, 
            interactive_state, 
            run_status,
            inpaint_video_predict_button,
            enhance_button,
            enhance_target_region_frame1_button,
            enhance_editing_instruction_button,
            notes_accordion  # Use the accordion reference instead of string
        ]
    )

    # inpaint video from select image and mask
    inpaint_video_predict_button.click(
        fn=inpaint_video,
        inputs=[video_state, video_caption, target_region_frame1_caption, interactive_state, run_status, seed_param, cfg_scale, dilate_size],
        outputs=[video_output, run_status],
        api_name=False,
        show_progress="full",
    )

    def enhance_prompt_func(video_caption):
        return convert_prompt(video_caption, retry_times=1)
    
    def enhance_target_region_frame1_prompt_func(target_region_frame1_caption, video_state):
        return convert_prompt_target_region_frame1(target_region_frame1_caption, video_state, retry_times=1)

    def enhance_editing_instruction_prompt_func(editing_instruction, video_caption, target_region_frame1_caption, video_state):
        return convert_prompt_editing_instruction(editing_instruction, video_caption, target_region_frame1_caption, video_state, retry_times=1)[0], convert_prompt_editing_instruction(editing_instruction, video_caption, target_region_frame1_caption, video_state, retry_times=1)[1]

    enhance_button.click(enhance_prompt_func, inputs=[video_caption], outputs=[video_caption])

    enhance_target_region_frame1_button.click(enhance_target_region_frame1_prompt_func, inputs=[target_region_frame1_caption, video_state], outputs=[target_region_frame1_caption])

    enhance_editing_instruction_button.click(enhance_editing_instruction_prompt_func, inputs=[editing_instruction, video_caption, target_region_frame1_caption, video_state], outputs=[video_caption, target_region_frame1_caption])

    
    video_input.clear(
        lambda: (
        gr.update(visible=True),
        gr.update(visible=True),
        init_state(),
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 8,
        "ann_obj_id": 0
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": False,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": 0,
        },
        [[],[]],
        None,
        None,
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True, value=[]), gr.update(visible=True), \
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Button.update(interactive=False),
        ),
        [],
        [
            video_caption,
            target_region_frame1_caption,
            inference_state,
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider, track_pause_number_slider, point_prompt,
            clear_button_click, template_frame, tracking_video_predict_button, video_output, inpaint_video_predict_button, run_status,
            enhance_button,
            enhance_target_region_frame1_button,
            enhance_editing_instruction_button
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn=clear_click,
        inputs=[inference_state, video_state, click_state, run_status],
        outputs=[inference_state, template_frame, click_state, run_status]
    )

    # Add reset button click event
    reset_button.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            video_input,
            video_caption,
            target_region_frame1_caption,
            inference_state,
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            video_info,
            image_selection_slider,
            track_pause_number_slider,
            point_prompt,
            clear_button_click,
            tracking_video_predict_button,
            inpaint_video_predict_button,
            enhance_button,
            enhance_target_region_frame1_button,
            enhance_editing_instruction_button,
            seed_param,
            cfg_scale,
            dilate_size,
            run_status
        ]
    )

iface.queue().launch(
    server_name="0.0.0.0", 
    server_port=12346, 
    share=False,
    allowed_paths=["./assets/"],
)
