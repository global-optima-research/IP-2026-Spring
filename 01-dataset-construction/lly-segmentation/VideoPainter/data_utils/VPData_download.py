import pandas as pd
import os
import requests
from tqdm import tqdm

def download_video(url, file_path):
    res = requests.get(url, stream=True)
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as f:
        for chunk in tqdm(res.iter_content(chunk_size=10240), desc="Downloading video"):
            f.write(chunk)

output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)


# videovo_df = pd.read_csv("../data/videovo.csv")
pexels_df = pd.read_csv("../data/pexels.csv")


# for index, row in tqdm(videovo_df[:].iterrows(), desc="Downloading videovo videos"):
#     video_url = row["video_url"]
#     file_path = os.path.join(output_dir, "videovo/raw_video", row["file_path"])
#     download_video(video_url, file_path)


for index, row in tqdm(pexels_df[:].iterrows(), desc="Downloading pexels videos"):
    video_url = row["link"]
    dir_prefix = f"{index:012d}"
    formatted_filename = f"{dir_prefix[:9]}/{dir_prefix}_{row['videoId']}.mp4"
    file_path = os.path.join(output_dir, "pexels/pexels/raw_video", formatted_filename)
    download_video(video_url, file_path)
