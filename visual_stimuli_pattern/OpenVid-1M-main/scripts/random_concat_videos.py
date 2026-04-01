#!/usr/bin/env python3
import os
import random
import argparse
import subprocess
import tempfile
from pathlib import Path


def get_video_files(input_dir):
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                paths.append(p)
    return paths


def probe_size(path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


def run_ffmpeg(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)


def main():
    parser = argparse.ArgumentParser(description="随机读取目录中的视频并拼接，中间插入灰屏")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/media/ubuntu/sda/visual_stimuli_pattern/OpenVid-1M-main/video_30fps",
        help="输入视频目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="random_concat.mp4",
        help="输出视频文件名",
    )
    parser.add_argument(
        "--gray-duration",
        type=float,
        default=1.0,
        help="视频之间的灰屏时长（秒）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="随机选取的视频数量，0 表示使用目录下的全部视频",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="输出视频帧率",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {input_dir}")

    videos = get_video_files(str(input_dir))
    if not videos:
        raise SystemExit(f"目录中没有找到视频文件: {input_dir}")

    random.shuffle(videos)
    if args.limit > 0:
        videos = videos[: args.limit]

    w, h = probe_size(videos[0])

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        norm_paths = []
        for idx, v in enumerate(videos):
            norm_path = tmpdir_path / f"norm_{idx}.mp4"
            norm_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(v),
                "-vf",
                f"scale={w}:{h},fps={args.fps}",
                "-c:v",
                "libx264",
                "-an",
                "-pix_fmt",
                "yuv420p",
                str(norm_path),
            ]
            run_ffmpeg(norm_cmd)
            norm_paths.append(norm_path)

        gray_path = tmpdir_path / "gray.mp4"
        gray_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={w}x{h}:d={args.gray_duration}",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-an",
            "-pix_fmt",
            "yuv420p",
            str(gray_path),
        ]
        run_ffmpeg(gray_cmd)

        list_path = tmpdir_path / "concat_list.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for idx, v in enumerate(norm_paths):
                f.write(f"file {str(v)}\n")
                if idx != len(videos) - 1:
                    f.write(f"file {str(gray_path)}\n")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-an",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        run_ffmpeg(concat_cmd)


if __name__ == "__main__":
    main()

