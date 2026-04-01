import argparse
import json
import os
import pickle

from tqdm import tqdm


def load_paths(stimuli_dir: str, limit: int) -> list[str]:
    files = sorted([f for f in os.listdir(stimuli_dir) if f.lower().endswith(".bmp")])
    paths = [os.path.join(stimuli_dir, f) for f in files]
    return paths[:limit]


def load_captions(caption_path: str, limit: int):
    with open(caption_path, "rb") as f:
        caps = pickle.load(f)
    return caps[:limit]


def caption_to_text(c_arr) -> str:
    if hasattr(c_arr, "flat"):
        if getattr(c_arr, "size", 0) == 0:
            return ""
        v = c_arr.flat[0]
        return v.decode("utf-8", errors="ignore") if isinstance(v, (bytes, bytearray)) else str(v)
    if c_arr:
        return str(c_arr[0])
    return ""


def build_prompt(caption: str, keyword: str) -> str:
    return (
        "Given the image and caption, first describe the background color style of the image with 3-5 words. "
        "Second, detect the TWO most important objects in the image. "
        f"Then, describe each of the objects and their relationship using: '{keyword}' with TWO sentences. "
        "For each sentence, use 5-10 words and as easy as possible.\n"
        "Then, detect the absolute position of the two objects in the image, and select from [right, left, top, bottom]. "
        "\"left\" and \"right\" should appear together for horizontal objects, and \"top\" and \"bottom\" should appear together for vertical objects. DO NOT mix.\n"
        "Example:\n"
        "### Background color style: Grayscale urban.\n"
        "### The Man [left]\n"
        "1. The man is standing near the sidewalk edge. The Man is close to the building wall.\n"
        "### The Suitcase [right]\n"
        "1. The suitcase is beside the man's foot. The Suitcase is placed on the street's curved edge.\n"
        f"Now, given the image I uploaded and the caption \"{caption}\", detect the two most important objects with absolute position, "
        f"describe them using '{keyword}' with EXACTLY the example format."
    )


def safe_keyword(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli-dir", default="/media/ubuntu/sda/TrippleN/stimuli")
    ap.add_argument("--caption-path", default="/media/ubuntu/sda/TrippleN/customize/coco_captions_1000x5.pkl")
    ap.add_argument("--image-base", default="https://stimuli.oss-cn-beijing.aliyuncs.com/stimuli")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="qwen3.5-flash")
    ap.add_argument("--batch-url", default="/v1/chat/completions")
    ap.add_argument("--limit", type=int, default=666)
    ap.add_argument(
        "--keywords",
        default="spatial layout,color attribute,action relation,part-whole relation,positional relation,functional relation",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    paths = load_paths(args.stimuli_dir, args.limit)
    caps = load_captions(args.caption_path, args.limit)

    n = min(len(paths), len(caps), args.limit)
    paths = paths[:n]
    caps = caps[:n]

    for keyword in keywords:
        safe = safe_keyword(keyword)
        out_path = os.path.join(args.out_dir, f"requests_{safe}.jsonl")
        if os.path.exists(out_path):
            os.remove(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            for idx, p in tqdm(list(enumerate(paths)), total=len(paths), desc=safe):
                basename = os.path.basename(p)
                image_url = args.image_base.rstrip("/") + "/" + basename
                caption = caption_to_text(caps[idx])
                prompt = build_prompt(caption, keyword)
                body = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                }
                rec = {
                    "custom_id": f"{safe}-{idx:04d}",
                    "method": "POST",
                    "url": args.batch_url,
                    "body": body,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
