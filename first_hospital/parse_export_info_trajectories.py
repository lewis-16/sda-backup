# -*- coding: utf-8 -*-
"""
从华科精准 Robot.SEEG 的 export_info 二进制文件中解析植入电极轨迹坐标。
用法: python parse_export_info_trajectories.py <export_info路径> [--csv]
"""
import sys
import re


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
        direction = re.search(r'<electrodeDirection>([^<]+)</electrodeDirection>', frag)
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
            'direction': direction.group(1).strip() if direction else '',
        })
    return trajectories


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print('用法: python parse_export_info_trajectories.py <export_info路径> [--csv]')
        sys.exit(1)
    out_csv = '--csv' in sys.argv

    with open(path, 'rb') as f:
        data = f.read()

    xml_str = extract_plan_xml(data)
    if not xml_str:
        print('未在文件中找到 <plan>...</plan> 内容')
        sys.exit(2)

    trajectories = parse_trajectories(xml_str)
    if not trajectories:
        print('未解析到任何轨迹')
        sys.exit(3)

    if out_csv:
        print('name,entry_x,entry_y,entry_z,target_x,target_y,target_z,electrode_name')
        for t in trajectories:
            e, tg = t['entry'], t['target']
            print('{},{},{},{},{},{},{},{}'.format(
                t['name'], e[0], e[1], e[2], tg[0], tg[1], tg[2], t['electrode_name']))
        return

    print('轨迹数: {}\n'.format(len(trajectories)))
    print('{:<6} {:>12} {:>12} {:>12}   |   {:>12} {:>12} {:>12}   {:<20}'.format(
        '名称', 'entry_x', 'entry_y', 'entry_z', 'target_x', 'target_y', 'target_z', '电极型号'))
    print('-' * 100)
    for t in trajectories:
        e, tg = t['entry'], t['target']
        print('{:<6} {:12.4f} {:12.4f} {:12.4f}   |   {:12.4f} {:12.4f} {:12.4f}   {:<20}'.format(
            t['name'], e[0], e[1], e[2], tg[0], tg[1], tg[2], t['electrode_name']))
    print('\n说明: entry=入颅点(mm), target=靶点(mm)，坐标系为规划基准影像(Basis)下的物理坐标。')


if __name__ == '__main__':
    main()
