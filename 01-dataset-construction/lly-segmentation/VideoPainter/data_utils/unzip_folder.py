import os
import zipfile
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='./data/videovo_masks')
parser.add_argument('--target_dir', type=str, default='./data/video_inpainting/videovo')
args = parser.parse_args()

source_dir = args.source_dir
target_dir = args.target_dir

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

zip_files = glob.glob(os.path.join(source_dir, '*.zip'))
zip_files.sort()

for zip_file in zip_files:
    print(f"Unzip {zip_file}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Unzip {zip_file} done")
