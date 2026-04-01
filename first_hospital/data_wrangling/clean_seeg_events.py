# -*- coding: utf-8 -*-
import re
import ast

INPUT_FILE = "sEEG_check_output.txt"
OUTPUT_FILE = "sEEG_check_output_cleaned.txt"

DELETE_SEMIOLOGY = {"误打", "◆视频好！！", "找家长", "EEG好！", "EEG好！！", "◆"}


def keep_event(label):
    label = label.strip()
    if not label:
        return False
    if label.startswith("◆发作") or label == "◆发作":
        return True
    if label in ("end", "END", "发作"):
        return True
    if re.match(r"^[a-zA-Z]+$", label):
        return True
    if "Stim" in label or "stim" in label:
        return True
    if label == "后放电":
        return True
    if label in DELETE_SEMIOLOGY:
        return False
    if label.startswith("Segment") or label.startswith("A1+A2") or "RESET" in label:
        return False
    if re.search(r"[\u4e00-\u9fff]", label):
        return True
    return False


def parse_events_line(line):
    line = line.strip()
    if "事件类型:" not in line:
        return None
    idx = line.find("事件类型:")
    s = line[idx + len("事件类型:"):].strip()
    if not s or s[0] != "[":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.split("------------------------------------------------------------")
    out_blocks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("共处理"):
            out_blocks.append(block)
            continue
        lines = block.split("\n")
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "  事件数:" in line and i + 1 < len(lines) and "  事件类型:" in lines[i + 1]:
                events = parse_events_line(lines[i + 1])
                if events is not None:
                    kept = [e for e in events if keep_event(e)]
                    kept_unique = list(dict.fromkeys(kept))
                    new_lines.append("  事件数: %d" % len(kept_unique))
                    new_lines.append("  事件类型: %s" % repr(kept_unique))
                    i += 2
                    continue
            new_lines.append(line)
            i += 1
        out_blocks.append("\n".join(new_lines))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n------------------------------------------------------------\n".join(out_blocks))

    print("Cleaned output written to %s" % OUTPUT_FILE)


if __name__ == "__main__":
    main()
