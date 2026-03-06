#!/usr/bin/env python3
"""
convert_to_webdataset.py — Convert OpenVid-1M videos + captions to WebDataset format.

Reads the CSV metadata and video files, filters by quality/duration,
and writes WebDataset tar shards (mp4 + txt pairs) for FastGen training.

Usage:
    python convert_to_webdataset.py \
        --csv /data/datasets/OpenVid-1M/OpenVid-1M.csv \
        --video_dir /data/datasets/OpenVid-1M/videos \
        --output_dir /data/datasets/OpenVid-1M/webdataset \
        --max_samples 50000 \
        --shard_size 1000

Output structure:
    webdataset/
    ├── shard-000000.tar   (1000 samples each)
    │   ├── 000000.mp4
    │   ├── 000000.txt
    │   ├── 000001.mp4
    │   ├── 000001.txt
    │   └── ...
    ├── shard-000001.tar
    └── ...
"""

import argparse
import csv
import io
import os
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenVid-1M to WebDataset format")
    parser.add_argument("--csv", type=str, required=True, help="Path to OpenVid-1M.csv")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing extracted mp4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for WebDataset shards")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max number of samples to convert")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of samples per shard tar")
    parser.add_argument("--min_frames", type=int, default=81, help="Minimum frame count filter")
    parser.add_argument("--min_aesthetic", type=float, default=5.0, help="Minimum aesthetic score filter")
    parser.add_argument("--min_seconds", type=float, default=2.0, help="Minimum duration in seconds")
    return parser.parse_args()


def load_and_filter_csv(
    csv_path: str,
    video_dir: str,
    max_samples: int,
    min_frames: int,
    min_aesthetic: float,
    min_seconds: float,
) -> list[dict]:
    """Load CSV and filter by quality criteria, only keep samples with existing video files."""
    records = []
    skipped_missing = 0
    skipped_filter = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Quality filters
            try:
                frames = int(row.get("frame", 0))
                aesthetic = float(row.get("aesthetic_score", 0))
                seconds = float(row.get("seconds", 0))
            except (ValueError, TypeError):
                skipped_filter += 1
                continue

            if frames < min_frames or aesthetic < min_aesthetic or seconds < min_seconds:
                skipped_filter += 1
                continue

            # Check video file exists
            video_name = row["video"]
            video_path = os.path.join(video_dir, video_name)
            if not os.path.isfile(video_path):
                skipped_missing += 1
                continue

            records.append({
                "video": video_name,
                "video_path": video_path,
                "caption": row.get("caption", ""),
            })

            if len(records) >= max_samples:
                break

    print(f"Loaded {len(records)} samples (skipped {skipped_missing} missing, {skipped_filter} filtered)")
    return records


def write_shards(records: list[dict], output_dir: str, shard_size: int) -> None:
    """Write records into WebDataset tar shards."""
    os.makedirs(output_dir, exist_ok=True)

    total = len(records)
    shard_idx = 0
    sample_idx = 0
    tar = None

    for i, rec in enumerate(records):
        # Start new shard if needed
        if i % shard_size == 0:
            if tar is not None:
                tar.close()
            shard_name = f"shard-{shard_idx:06d}.tar"
            shard_path = os.path.join(output_dir, shard_name)
            tar = tarfile.open(shard_path, "w")
            print(f"  Writing {shard_name} ({i}/{total})...")
            shard_idx += 1

        key = f"{sample_idx:06d}"

        # Add mp4
        tar.add(rec["video_path"], arcname=f"{key}.mp4")

        # Add txt (caption)
        caption_bytes = rec["caption"].encode("utf-8")
        txt_info = tarfile.TarInfo(name=f"{key}.txt")
        txt_info.size = len(caption_bytes)
        tar.addfile(txt_info, io.BytesIO(caption_bytes))

        sample_idx += 1

    if tar is not None:
        tar.close()

    num_shards = shard_idx
    print(f"Done: {sample_idx} samples in {num_shards} shards -> {output_dir}")
    print(f"Use in FastGen config:")
    print(f'  dataloader_train.datatags=["WDS:{output_dir}/shard-{{000000..{num_shards-1:06d}}}.tar"]')


def main() -> None:
    args = parse_args()

    print("=== OpenVid-1M -> WebDataset Converter ===")
    print(f"CSV: {args.csv}")
    print(f"Video dir: {args.video_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Filters: frames>={args.min_frames}, aesthetic>={args.min_aesthetic}, seconds>={args.min_seconds}")
    print()

    records = load_and_filter_csv(
        args.csv,
        args.video_dir,
        args.max_samples,
        args.min_frames,
        args.min_aesthetic,
        args.min_seconds,
    )

    if not records:
        print("ERROR: No samples found after filtering. Check video_dir and CSV paths.")
        return

    print()
    write_shards(records, args.output_dir, args.shard_size)


if __name__ == "__main__":
    main()
