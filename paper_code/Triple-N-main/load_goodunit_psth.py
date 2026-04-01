# -*- coding: utf-8 -*-
"""
从 /home/ubuntu/Downloads/TrippleN/GoodUnit 下所有 GoodUnit*.mat 中读取数据。
提供两种处理方式：
1. raster-based PSTH: psth (n_neuron, n_trial, time_bins) 保存为 pkl
2. response_matrix_img: rmi (n_neuron, n_image, time_bins) 保存为 npy

MAT 为 MATLAB v7.3 (HDF5)，含大量引用/嵌套结构，需用 h5py 解析。
"""

import os
import pickle
import numpy as np
import h5py
from pathlib import Path
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def _scalar(x):
    v = np.array(x)
    if v.size == 1:
        return float(v.flatten()[0]) if np.issubdtype(v.dtype, np.floating) else int(v.flatten()[0])
    return v


def _deref(f, ref):
    if ref is None:
        return None
    try:
        return f[ref]
    except Exception:
        return None


def raster_to_psth(raster, window_ms):
    """
    raster: (n_trial, time_bins), 每格为 spike count (1ms bin)
    window_ms: 滑窗宽度 (ms)，与 MATLAB psth_window_size_ms 一致
    返回: (n_trial, time_bins) Hz，与 PostProcess_function.m 中 psth_raw 算法一致
    """
    n_trial, n_bins = raster.shape
    half = int(window_ms // 2)
    cum = np.cumsum(np.column_stack([np.zeros((n_trial, 1), dtype=np.float64), raster.astype(np.float64)]), axis=1)
    ts = np.arange(n_bins)
    low = np.maximum(0, ts - half)
    high = np.minimum(n_bins, ts + half + 1)
    win_len = np.maximum(1, high - low)
    win_sum = cum[:, high] - cum[:, low]
    psth = 1000.0 * win_sum / win_len[np.newaxis, :]
    return psth


def load_one_session(mat_path, n_workers_psth=0):
    """
    加载一个 GoodUnit*.mat，解析 HDF5 中的引用与嵌套。
    mat_path: 文件路径（str 或 Path）
    n_workers_psth: 单文件内并行计算各 unit 的 PSTH 时使用的线程数，0 表示不并行
    返回:
        psth: (n_neuron, n_trial, time_bins)
        image_id: (n_trial,) 每个 trial 对应的图像 id
        session_name: 文件名（不含路径与扩展名）
    若解析失败返回 (None, None, session_name)。
    """
    mat_path = Path(mat_path)
    if not mat_path.suffix.lower() == '.mat' or 'GoodUnit' not in mat_path.name:
        return None, None, mat_path.stem

    with h5py.File(mat_path, 'r', rdcc_nbytes=0) as f:
        if 'meta_data' not in f or 'GoodUnitStrc' not in f or 'global_params' not in f:
            return None, None, mat_path.stem

        tvi = np.array(f['meta_data']['trial_valid_idx'][:]).flatten()
        image_id = np.asarray(tvi[tvi != 0], dtype=np.int64)

        gp = f['global_params']
        pre_onset = int(_scalar(gp['pre_onset'][()]))
        post_onset = int(_scalar(gp['post_onset'][()]))
        psth_window_ms = int(_scalar(gp['psth_window_size_ms'][()]))

        gus = f['GoodUnitStrc']
        raster_ds = gus['Raster']
        n_units = raster_ds.shape[0]
        n_trial = len(image_id)
        time_bins = pre_onset + post_onset

        def is_unit_valid(i):
            ref = raster_ds[i, 0]
            obj = _deref(f, ref)
            if obj is None or obj.ndim != 2:
                return False
            sh = obj.shape
            return (sh[0] == n_trial and sh[1] == time_bins) or (sh[1] == n_trial and sh[0] == time_bins)

        n_valid = sum(1 for i in range(n_units) if is_unit_valid(i))
        if n_valid == 0:
            return None, None, mat_path.stem
        psth = np.zeros((n_valid, n_trial, time_bins), dtype=np.float64)

        def one_unit(i):
            ref = raster_ds[i, 0]
            obj = _deref(f, ref)
            if obj is None:
                return None
            arr = np.array(obj[:], dtype=np.float64)
            if arr.ndim != 2:
                return None
            if arr.shape[0] == n_trial and arr.shape[1] == time_bins:
                r = arr
            elif arr.shape[1] == n_trial and arr.shape[0] == time_bins:
                r = arr.T
            else:
                return None
            return raster_to_psth(r, psth_window_ms)

        valid_idx = [0]
        lock = threading.Lock()

        def fill_slot(i):
            row = one_unit(i)
            if row is not None:
                with lock:
                    idx = valid_idx[0]
                    psth[idx] = row
                    valid_idx[0] += 1

        if n_workers_psth and n_units > 1:
            with ThreadPoolExecutor(max_workers=n_workers_psth) as ex:
                list(ex.map(fill_slot, range(n_units)))
        else:
            for i in range(n_units):
                fill_slot(i)
        if valid_idx[0] == 0:
            return None, None, mat_path.stem

    return psth, image_id, mat_path.stem


def _load_one_session_wrap(mat_path, n_workers_psth):
    psth, image_id, name = load_one_session(mat_path, n_workers_psth=n_workers_psth)
    if psth is None or image_id is None:
        return None
    return {'session_name': name, 'psth': psth, 'image_id': image_id}


def load_one_session_response_matrix(mat_path):
    """
    加载一个 GoodUnit*.mat，提取每个 unit 的 response_matrix_img。
    mat_path: 文件路径（str 或 Path）
    返回:
        response_matrix: (n_neuron, n_image, time_bins) float32
        session_name: 文件名（不含路径与扩展名）
    若解析失败返回 (None, session_name)。
    """
    mat_path = Path(mat_path)
    if not mat_path.suffix.lower() == '.mat' or 'GoodUnit' not in mat_path.name:
        return None, mat_path.stem

    with h5py.File(mat_path, 'r', rdcc_nbytes=0) as f:
        if 'meta_data' not in f or 'GoodUnitStrc' not in f or 'global_params' not in f:
            return None, mat_path.stem

        gp = f['global_params']
        pre_onset = int(_scalar(gp['pre_onset'][()]))
        post_onset = int(_scalar(gp['post_onset'][()]))
        time_bins = pre_onset + post_onset

        gus = f['GoodUnitStrc']
        raster_ds = gus['Raster']
        rmi_ds = gus['response_matrix_img']
        n_units = raster_ds.shape[0]

        # 获取第一个 unit 的 response_matrix_img 以确定 n_image
        first_ref = rmi_ds[0, 0]
        first_data = _deref(f, first_ref)
        if first_data is None or first_data.ndim != 2:
            return None, mat_path.stem

        rmi_shape = first_data.shape
        if rmi_shape[0] == time_bins:
            n_image = rmi_shape[1]
        elif rmi_shape[1] == time_bins:
            n_image = rmi_shape[0]
        else:
            return None, mat_path.stem

        response_matrix = np.zeros((n_units, n_image, time_bins), dtype=np.float32)

        for i in range(n_units):
            ref = rmi_ds[i, 0]
            data = _deref(f, ref)
            if data is None or data.ndim != 2:
                return None, mat_path.stem

            arr = np.array(data[:], dtype=np.float32)
            if arr.shape == (time_bins, n_image):
                response_matrix[i] = arr.T
            elif arr.shape == (n_image, time_bins):
                response_matrix[i] = arr
            else:
                return None, mat_path.stem

    return response_matrix, mat_path.stem


def process_and_save_one_session(mat_path, output_dir, n_workers_psth=0):
    """
    加载一个 GoodUnit*.mat，计算 PSTH，只写入该 session 的 pkl，不返回大数组。
    mat_path: 单个 mat 路径
    output_dir: 每个 session 的 pkl 写入此目录，文件名为 psth_{session_name}.pkl
    n_workers_psth: 单文件内 PSTH 线程数，0 表示不并行
    返回: 成功则返回 session_name，失败返回 None。调用方不持有 psth/image_id。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    psth, image_id, name = load_one_session(mat_path, n_workers_psth=n_workers_psth)
    if psth is None or image_id is None:
        return None
    out_path = output_dir / f"psth_{name}.pkl"
    with open(out_path, 'wb') as fp:
        pickle.dump({'session_name': name, 'psth': psth, 'image_id': image_id}, fp)
    del psth, image_id
    return name


def process_and_save_response_matrix(mat_path, output_dir):
    """
    加载一个 GoodUnit*.mat，提取每个 unit 的 response_matrix_img 并保存为 npy。
    mat_path: 单个 mat 路径
    output_dir: npy 文件写入此目录，文件名为 rmi_{session_name}.npy
    返回: 成功则返回 session_name，失败返回 None。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_matrix, name = load_one_session_response_matrix(mat_path)
    if response_matrix is None:
        return None
    out_path = output_dir / f"rmi_{name}.npy"
    np.save(out_path, response_matrix)
    del response_matrix
    return name


def _process_response_matrix_wrap(args):
    mat_path, output_dir = args
    return process_and_save_response_matrix(mat_path, output_dir)


def process_all_sessions_response_matrix(goodunit_dir, output_dir=None, max_sessions=None, n_workers=1):
    """
    按 session 逐个处理并各自落盘。每个 session 一个 npy：
    output_dir / rmi_{session_name}.npy，内容为 (n_neuron, n_image, time_bins) float32。
    goodunit_dir: 存放 GoodUnit*.mat 的目录
    output_dir: 写入目录，默认等于 goodunit_dir/psth
    max_sessions: 最多处理的 session 数，None 表示不限制
    n_workers: 并行处理的文件数。>1 时多进程同时各处理一个 session。
    返回: 成功写盘的 session_name 列表（顺序与文件名一致）。
    """
    goodunit_dir = Path(goodunit_dir)
    if not goodunit_dir.is_dir():
        return []
    out = Path(output_dir) if output_dir is not None else goodunit_dir / 'psth'
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(goodunit_dir.glob('GoodUnit*.mat'))
    if max_sessions is not None:
        paths = paths[:max_sessions]
    if not paths:
        return []
    paths = [str(p) for p in paths]
    out_str = str(out)
    if n_workers <= 1:
        done = []
        for p in paths:
            name = process_and_save_response_matrix(p, out_str)
            if name is not None:
                done.append(name)
        return done
    n_workers = min(n_workers, len(paths), os.cpu_count() or 4)
    tasks = [(p, out_str) for p in paths]
    done = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_response_matrix_wrap, t) for t in tasks]
        for fut in as_completed(futures):
            name = fut.result()
            if name is not None:
                done.append(name)
    name_to_idx = {Path(p).stem: i for i, p in enumerate(paths)}
    done.sort(key=lambda n: name_to_idx.get(n, len(paths)))
    return done


def _process_and_save_wrap(args):
    mat_path, output_dir, n_workers_psth = args
    return process_and_save_one_session(mat_path, output_dir, n_workers_psth)


def process_all_sessions_to_dir(goodunit_dir, output_dir=None, max_sessions=None, n_workers=1, n_workers_psth=0):
    """
    按 session 逐个处理并各自落盘，不在内存中聚合。每个 session 一个 pkl：
    output_dir / psth_{session_name}.pkl，内容为 {'session_name', 'psth', 'image_id'}。
    goodunit_dir: 存放 GoodUnit*.mat 的目录
    output_dir: 写入目录，默认等于 goodunit_dir
    max_sessions: 最多处理的 session 数，None 表示不限制
    n_workers: 并行处理的文件数。>1 时多进程同时各处理一个 session，总内存约 n_workers * 单 session 峰值；建议内存紧张时设为 1。
    n_workers_psth: 单文件内 PSTH 线程数，0 表示不并行（内存最低）
    返回: 成功写盘的 session_name 列表（顺序与文件名一致）。
    """
    goodunit_dir = Path(goodunit_dir)
    if not goodunit_dir.is_dir():
        return []
    out = Path(output_dir) if output_dir is not None else goodunit_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(goodunit_dir.glob('GoodUnit*.mat'))
    if max_sessions is not None:
        paths = paths[:max_sessions]
    if not paths:
        return []
    paths = [str(p) for p in paths]
    out_str = str(out)
    if n_workers <= 1:
        done = []
        for p in paths:
            name = process_and_save_one_session(p, out_str, n_workers_psth)
            if name is not None:
                done.append(name)
        return done
    n_workers = min(n_workers, len(paths), os.cpu_count() or 4)
    tasks = [(p, out_str, n_workers_psth) for p in paths]
    done = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_process_and_save_wrap, t) for t in tasks]
        for fut in as_completed(futures):
            name = fut.result()
            if name is not None:
                done.append(name)
    name_to_idx = {Path(p).stem: i for i, p in enumerate(paths)}
    done.sort(key=lambda n: name_to_idx.get(n, len(paths)))
    return done


def load_all_sessions(goodunit_dir, max_sessions=None, n_workers=1, n_workers_psth=0):
    """
    遍历 goodunit_dir 下所有 GoodUnit*.mat，计算 PSTH，全部结果放入内存并返回。
    若 session 很多、矩阵很大，建议改用 process_all_sessions_to_dir 按 session 落盘。
    """
    goodunit_dir = Path(goodunit_dir)
    if not goodunit_dir.is_dir():
        return []
    paths = sorted(goodunit_dir.glob('GoodUnit*.mat'))
    if max_sessions is not None:
        paths = paths[:max_sessions]
    if not paths:
        return []
    paths = [str(p) for p in paths]
    if n_workers <= 1:
        results = []
        for p in paths:
            rec = _load_one_session_wrap(p, n_workers_psth)
            if rec is not None:
                results.append(rec)
        return results
    n_workers = min(n_workers, len(paths), os.cpu_count() or 4)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_one_session_wrap, p, n_workers_psth): p for p in paths}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                results.append(rec)
    name_to_idx = {Path(p).stem: i for i, p in enumerate(paths)}
    results.sort(key=lambda d: name_to_idx.get(d['session_name'], len(paths)))
    return results


if __name__ == '__main__':
    goodunit_root = '/media/ubuntu/sda/TrippleN/GoodUnit'
    n_proc = 1

    # # 选项1: 保存 raster-based PSTH (n_neuron, n_trial, time_bins) 到 pkl
    # print("=" * 60)
    # print("选项1: 保存 raster-based PSTH 到 pkl")
    # print("=" * 60)
    # output_dir = Path(goodunit_root) / 'psth_per_session'
    # done_pkl = process_all_sessions_to_dir(
    #     goodunit_root,
    #     output_dir=output_dir,
    #     n_workers=n_proc,
    #     n_workers_psth=0,
    # )
    # for name in done_pkl:
    #     p = output_dir / f"psth_{name}.pkl"
    #     print(name, '->', p)
    # print('sessions saved:', len(done_pkl), 'under', output_dir)

    # 选项2: 保存 response_matrix_img (n_neuron, n_image, time_bins) 到 npy
    print("\n" + "=" * 60)
    print("选项2: 保存 response_matrix_img 到 npy")
    print("=" * 60)
    psth_output_dir = '/media/ubuntu/sda/TrippleN/psth'
    done_npy = process_all_sessions_response_matrix(
        goodunit_root,
        output_dir=psth_output_dir,
        max_sessions=None,
        n_workers=10,
    )
    for name in done_npy:
        p = Path(psth_output_dir) / f"rmi_{name}.npy"
        print(name, '->', p)
    print('sessions saved:', len(done_npy), 'under', psth_output_dir)
