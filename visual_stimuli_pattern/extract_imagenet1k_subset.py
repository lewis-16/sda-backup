#!/usr/bin/env python3
import json
import shutil
import os
from pathlib import Path

IMAGENET_ROOT = Path("/media/ubuntu/sda/visual_stimuli_pattern/Imagenet")
OUTPUT_ROOT = Path("/media/ubuntu/sda/visual_stimuli_pattern/imagenet1k")
CLASS_JSON = Path("/media/ubuntu/sda/visual_stimuli_pattern/imagenet1k_class.json")
IMAGES_PER_CLASS = 100


def main():
    with open(CLASS_JSON, "r") as f:
        class_dict = json.load(f)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for idx in range(1000):
        class_id = str(idx)
        nxxx = class_dict[class_id][0]
        src_dir = IMAGENET_ROOT / nxxx
        dst_dir = OUTPUT_ROOT / nxxx

        if not src_dir.exists():
            print(f"Skip {nxxx}: source dir not found")
            continue

        images = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
        if len(images) < IMAGES_PER_CLASS:
            print(f"Warn {nxxx}: only {len(images)} images, need {IMAGES_PER_CLASS}")

        dst_dir.mkdir(parents=True, exist_ok=True)
        for i, src_path in enumerate(images[:IMAGES_PER_CLASS]):
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)

        if (idx + 1) % 100 == 0:
            print(f"Done {idx + 1}/1000")


if __name__ == "__main__":
    main()
