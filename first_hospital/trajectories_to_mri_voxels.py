# -*- coding: utf-8 -*-
"""
将 export_info 中的电极轨迹从物理坐标(mm)换算到规划基准(Basis) MRI 的体素坐标。
Basis 即 export 自带的 MR 序列，与轨迹同坐标系，仅做几何变换即可。

用法:
  python trajectories_to_mri_voxels.py <export_info路径> [--basis-dir <Basis的DICOM目录>] [--csv]
若省略 --basis-dir，则从 export_info 所在目录推断: <export_info目录>/files/importFile/2024/{Basis系列UUID}/
"""
import sys
import re
import os
import argparse

try:
    from pydicom import dcmread
    import numpy as np
except ImportError as e:
    print("需要安装: pip install pydicom numpy")
    raise SystemExit(1) from e


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
        })
    return trajectories


def get_basis_series_uuid(xml_str):
    m = re.search(r'<template>Basis</template>.*?<seriesuuid>\{([^}]+)\}</seriesuuid>', xml_str, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def is_dicom(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0x80)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def build_basis_geometry(basis_dir):
    files = [os.path.join(basis_dir, n) for n in os.listdir(basis_dir)
             if os.path.isfile(os.path.join(basis_dir, n)) and is_dicom(os.path.join(basis_dir, n))]
    if not files:
        raise FileNotFoundError('Basis 目录下未找到 DICOM 文件: {}'.format(basis_dir))
    def sloc(f):
        try:
            return dcmread(f, force=True).SliceLocation
        except Exception:
            return 0
    files_sorted = sorted(files, key=sloc)
    ds0 = dcmread(files_sorted[0], force=True)
    ds1 = dcmread(files_sorted[1], force=True)
    iop = ds0.ImageOrientationPatient
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    slice_dir = np.cross(row, col)
    pos0 = np.array(ds0.ImagePositionPatient, dtype=float)
    pos1 = np.array(ds1.ImagePositionPatient, dtype=float)
    sp_slice = np.abs(np.dot(pos1 - pos0, slice_dir))
    if sp_slice < 0.01:
        sp_slice = float(getattr(ds0, 'SliceThickness', 1.0) or 1.0)
    ps = float(ds0.PixelSpacing[0])
    R = np.eye(4)
    R[:3, 0] = row * ps
    R[:3, 1] = col * ps
    R[:3, 2] = slice_dir * sp_slice
    R[:3, 3] = pos0
    R3 = R[:3, :3]
    t = R[:3, 3]
    R_inv = np.linalg.inv(R3)
    n_slices = len(files_sorted)
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    shape = (n_slices, rows, cols)

    def world_to_voxel(world_mm):
        return R_inv @ (np.asarray(world_mm, dtype=float) - t)

    return {
        'world_to_voxel': world_to_voxel,
        'shape': shape,
        'spacing': (sp_slice, ps, ps),
        'origin': pos0,
    }


def main():
    parser = argparse.ArgumentParser(description='将轨迹物理坐标换算到 Basis MRI 体素')
    parser.add_argument('export_info_path', help='export_info 文件路径')
    parser.add_argument('--basis-dir', default=None, help='Basis 序列 DICOM 目录（默认自动推断）')
    parser.add_argument('--csv', action='store_true', help='输出 CSV')
    args = parser.parse_args()

    export_path = os.path.abspath(args.export_info_path)
    if not os.path.isfile(export_path):
        print('文件不存在:', export_path)
        sys.exit(1)
    export_dir = os.path.dirname(export_path)

    with open(export_path, 'rb') as f:
        data = f.read()
    xml_str = extract_plan_xml(data)
    if not xml_str:
        print('未在 export_info 中找到 <plan> 内容')
        sys.exit(2)
    trajectories = parse_trajectories(xml_str)
    if not trajectories:
        print('未解析到任何轨迹')
        sys.exit(3)

    basis_dir = args.basis_dir
    if not basis_dir:
        uuid = get_basis_series_uuid(xml_str)
        if not uuid:
            print('未找到 Basis 系列 UUID，请用 --basis-dir 指定 DICOM 目录')
            sys.exit(4)
        basis_dir = os.path.join(export_dir, 'files', 'importFile', '2024', '{{{}}}'.format(uuid))
    basis_dir = os.path.abspath(basis_dir)
    if not os.path.isdir(basis_dir):
        print('Basis 目录不存在:', basis_dir)
        sys.exit(5)

    geom = build_basis_geometry(basis_dir)
    world_to_voxel = geom['world_to_voxel']
    shape = geom['shape']
    nz, ny, nx = shape

    for t in trajectories:
        t['entry_vox'] = world_to_voxel(t['entry']).tolist()
        t['target_vox'] = world_to_voxel(t['target']).tolist()

    if args.csv:
        print('name,entry_i,entry_j,entry_k,target_i,target_j,target_k,entry_x_mm,entry_y_mm,entry_z_mm,target_x_mm,target_y_mm,target_z_mm,electrode_name')
        for t in trajectories:
            ei, ej, ek = t['entry_vox']
            ti, tj, tk = t['target_vox']
            ex, ey, ez = t['entry']
            tx, ty, tz = t['target']
            print('{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                t['name'], ei, ej, ek, ti, tj, tk, ex, ey, ez, tx, ty, tz, t['electrode_name']))
        return

    print('Basis MRI 体素尺寸 (k,j,i): {} (对应 切片,行,列)\n'.format(shape))
    print('{:<6} {:>8} {:>8} {:>8}   |   {:>8} {:>8} {:>8}   {:<16}'.format(
        '名称', 'entry_i', 'entry_j', 'entry_k', 'target_i', 'target_j', 'target_k', '电极'))
    print('-' * 90)
    for t in trajectories:
        ei, ej, ek = t['entry_vox']
        ti, tj, tk = t['target_vox']
        print('{:<6} {:8.2f} {:8.2f} {:8.2f}   |   {:8.2f} {:8.2f} {:8.2f}   {:<16}'.format(
            t['name'], ei, ej, ek, ti, tj, tk, t['electrode_name']))
    print('\n说明: 体素坐标为 (k,j,i)，k=切片、j=行、i=列，对应 numpy 数组 vol[k,j,i]。')
    print('      超出 [0, shape-1] 表示该点在 MRI FOV 外（如入颅点常在颅骨附近）。')


if __name__ == '__main__':
    main()
