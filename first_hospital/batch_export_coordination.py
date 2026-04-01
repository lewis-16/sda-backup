# -*- coding: utf-8 -*-
"""
遍历 不同年龄段的SEEG原始数据2026-2-1 下每位患者，从 Robot SEEG 导出的 export_info
中解析植入电极坐标与各触点坐标，在各患者目录下生成 coordination.md。
"""
import os
import re
import glob

BASE = "/media/ubuntu/sda/first_hospital/不同年龄段的SEEG原始数据2026-2-1"


def contact_positions_from_expression(expression_str, target, entry):
    """
    根据 electrodeExpression（0,2,1.5,2,1.5,... 单位 mm）和 entry/target，
    计算各触点中心在物理空间中的坐标。靶点为轨迹深端，触点沿 target->entry 方向排列。
    返回 [(x,y,z), ...]，触点 1 为最深（靠近 target）。
    """
    try:
        arr = [float(x.strip()) for x in expression_str.split(",")]
    except ValueError:
        return []
    if len(arr) < 2:
        return []
    tx, ty, tz = target[0], target[1], target[2]
    ex, ey, ez = entry[0], entry[1], entry[2]
    dx = ex - tx
    dy = ey - ty
    dz = ez - tz
    from math import sqrt
    norm = sqrt(dx*dx + dy*dy + dz*dz)
    if norm < 1e-6:
        return []
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    positions_mm = []
    pos = 0.0
    for i in range(1, len(arr)):
        seg = arr[i]
        if abs(seg - 2.0) < 0.01:
            positions_mm.append(pos + 1.0)
            pos += 2.0
        else:
            pos += seg
    return [
        (tx + p * dx, ty + p * dy, tz + p * dz) for p in positions_mm
    ]


def extract_plan_xml(data):
    link_marker = '<link><template>Basis</template>'.encode('utf-16-le')
    plan_start_marker = '<plan>'.encode('utf-16-le')
    plan_end_marker = '</plan>'.encode('utf-16-le')
    pos_link = data.find(link_marker)
    if pos_link < 0:
        pos_link = data.find('<trajectories>'.encode('utf-16-le'))
    if pos_link < 0:
        pos_link = data.find('<link>'.encode('utf-16-le'))
    if pos_link < 0:
        try:
            text = data.decode('utf-16-le', errors='replace')
            start = text.find('<plan>')
            end = text.rfind('</plan>')
            if start != -1 and end != -1 and end > start:
                return text[start:end + len('</plan>')]
        except Exception:
            pass
        return None
    pos_plan = data.rfind(plan_start_marker, 0, pos_link + 1)
    pos_end = data.rfind(plan_end_marker)
    if pos_plan < 0 or pos_end < 0 or pos_end <= pos_plan:
        return None
    try:
        return data[pos_plan:pos_end + len(plan_end_marker)].decode('utf-16-le', errors='replace')
    except Exception:
        return None


def parse_trajectories(xml_str):
    if not xml_str:
        return []
    trajectories = []
    block_pattern = re.compile(r'<trajectory>(.*?)</trajectory>', re.DOTALL)
    for blk in block_pattern.finditer(xml_str):
        frag = blk.group(1)
        name = re.search(r'<name>([^<]+)</name>', frag)
        entry = re.search(r'<entry>([^<]+)</entry>', frag)
        target = re.search(r'<target>([^<]+)</target>', frag)
        electrode_name = re.search(r'<electrodeName>([^<]+)</electrodeName>', frag)
        expression = re.search(r'<electrodeExpression>([^<]+)</electrodeExpression>', frag)
        if not (name and entry and target and electrode_name):
            continue
        try:
            entry_xyz = [float(x.strip()) for x in entry.group(1).split(',')]
            target_xyz = [float(x.strip()) for x in target.group(1).split(',')]
        except ValueError:
            continue
        trajectories.append({
            'name': name.group(1).strip(),
            'entry': entry_xyz,
            'target': target_xyz,
            'electrode_name': electrode_name.group(1).strip(),
            'electrode_expression': expression.group(1).strip() if expression else "",
        })
    return trajectories


def find_export_info(patient_dir):
    cand = glob.glob(os.path.join(patient_dir, "*", "export_info"))
    if not cand:
        return None
    return cand[0]


def write_coordination_md(patient_dir, trajectories, patient_name):
    out_path = os.path.join(patient_dir, "coordination.md")
    lines = [
        "# 植入电极坐标",
        "",
        "来源：Robot SEEG 导出 export_info，坐标系为规划基准影像(Basis)物理坐标，单位 mm。",
        "",
        "## 1. 各电极信息",
        "",
        "| 电极 | 入颅点 entry (x, y, z) mm | 靶点 target (x, y, z) mm | 电极型号 |",
        "|------|---------------------------|----------------------------|----------|",
    ]
    for t in trajectories:
        e = t["entry"]
        tg = t["target"]
        lines.append("| {} | {:.4f}, {:.4f}, {:.4f} | {:.4f}, {:.4f}, {:.4f} | {} |".format(
            t["name"], e[0], e[1], e[2], tg[0], tg[1], tg[2], t["electrode_name"]))
    lines.extend(["", "说明：entry=入颅点，target=靶点。触点编号 1 为最深（靠近 target），向 entry 方向递增。", ""])
    lines.extend(["## 2. 各电极触点坐标", ""])
    for t in trajectories:
        expr = t.get("electrode_expression") or ""
        contacts = contact_positions_from_expression(expr, t["target"], t["entry"])
        lines.append("### 电极 {} ({})".format(t["name"], t["electrode_name"]))
        lines.append("")
        if not contacts:
            lines.append("（未解析到触点排布，无触点坐标）")
            lines.append("")
            continue
        lines.append("| 触点 | x (mm) | y (mm) | z (mm) |")
        lines.append("|------|--------|--------|--------|")
        for k, (cx, cy, cz) in enumerate(contacts, 1):
            lines.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(k, cx, cy, cz))
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    if not os.path.isdir(BASE):
        print("目录不存在:", BASE)
        return
    patient_dirs = [os.path.join(BASE, d) for d in os.listdir(BASE)
                    if os.path.isdir(os.path.join(BASE, d))]
    patient_dirs.sort()
    done = 0
    failed = []
    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)
        export_path = find_export_info(patient_dir)
        if not export_path:
            failed.append((patient_name, "未找到 export_info"))
            continue
        try:
            with open(export_path, "rb") as f:
                data = f.read()
        except Exception as e:
            failed.append((patient_name, str(e)))
            continue
        xml_str = extract_plan_xml(data)
        if not xml_str:
            failed.append((patient_name, "export_info 中未解析到 plan"))
            continue
        trajectories = parse_trajectories(xml_str)
        if not trajectories:
            failed.append((patient_name, "未解析到任何轨迹"))
            continue
        out_path = write_coordination_md(patient_dir, trajectories, patient_name)
        print("OK  {}  -> {}  ({} 条轨迹)".format(patient_name, out_path, len(trajectories)))
        done += 1
    print("\n完成: {} 人已生成 coordination.md".format(done))
    if failed:
        print("未处理:", failed)


if __name__ == "__main__":
    main()
