"""
Build (n_voxel, n_stimuli) trial-averaged response matrix and save newdata + newdatastim.
Logic matches example06_basicbetaloading.m exactly: ROI = floc_facestval box & t>5,
z-score per session along trials, only images with exactly 3 trials, betas_fithrf.

Note: Official NSD pipeline (export.m) writes only .nii.gz and .hdf5 for betas; it does
NOT write .mat. Example06 reads .mat; we read .hdf5 (same numeric content). When std=0
in a session, z-score gives NaN in both MATLAB and Python.

Usage:
  python nsd_trial_averaged_response.py /path/to/naturescenedataset [--nsess 32] [--out out.npz]
  Use --roi /path/to/mask.nii.gz for custom ROI; use --include_all_reps to include 1/2-rep images.

Output .npz: newdata, newdatastim (10k design index 1..10000), newdatastim_73k (image ID for
nsd_stimuli.hdf5 /imgBrick, when exp has subjectim), voxel_info (n_voxel, n_ROI_type) numeric
matrix exactly matching ROI volumes sampled on the mask, roi_type_names.
"""
import os
import argparse
import numpy as np
import nibabel as nib
import h5py
from concurrent.futures import ProcessPoolExecutor

ATLAS_BINARY_THRESHOLD = 2


def load_atlas_label_names(roi_name, base_path):
    if not (roi_name == "HCP_MMP1" or (isinstance(roi_name, str) and "HCP" in roi_name and "MMP" in roi_name)):
        return None
    for search_dir in (os.path.dirname(os.path.abspath(__file__)), base_path, os.path.join(base_path, "roi")):
        if not search_dir:
            continue
        path = os.path.join(search_dir, "HCP_MMP1_labels.txt")
        if os.path.isfile(path):
            out = {}
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, start=1):
                    name = line.strip()
                    if name:
                        out[idx] = name
            return out if out else None
    try:
        from urllib.request import urlopen
        for url in (
            "https://raw.githubusercontent.com/nilearn/nilearn/main/nilearn/datasets/data/glasser_360_region_names.txt",
            "https://raw.githubusercontent.com/brainspaces/glasser360/master/glasser360NodeNames.txt",
        ):
            try:
                with urlopen(url, timeout=8) as resp:
                    lines = resp.read().decode("utf-8", errors="replace").strip().splitlines()
                if len(lines) >= 360:
                    return {i: lines[i - 1].strip() for i in range(1, min(361, len(lines) + 1))}
            except Exception:
                continue
    except Exception:
        pass
    return None


def _is_binary_region_column(voxel_info_numeric, j, roi_name):
    col = voxel_info_numeric[:, j]
    non_zero = col[col != 0]
    unique_vals = set(np.unique(np.round(non_zero)).astype(int)) if len(non_zero) else set()
    if len(unique_vals) > ATLAS_BINARY_THRESHOLD:
        return False
    if str(roi_name).lower() == "brainmask":
        return False
    return True


def voxel_info_numeric_to_string_matrix(voxel_info_numeric, roi_type_names, base_path):
    n_vox, n_roi = voxel_info_numeric.shape
    out = np.empty((n_vox, n_roi), dtype=object)
    for j in range(n_roi):
        col = voxel_info_numeric[:, j]
        col = np.asarray(col, dtype=float)
        mask_valid = (~np.isnan(col)) & (col != 0)
        non_zero = col[mask_valid]
        unique_vals = (
            set(np.unique(np.round(non_zero)).astype(int)) if len(non_zero) else set()
        )
        if len(unique_vals) <= ATLAS_BINARY_THRESHOLD:
            out[:, j] = np.where(mask_valid & (col > 0), roi_type_names[j], "")
        else:
            label_to_name = load_atlas_label_names(roi_type_names[j], base_path)
            if label_to_name is None:
                label_to_name = {int(v): f"{roi_type_names[j]}_{int(v)}" for v in unique_vals}
            for i in range(n_vox):
                val = voxel_info_numeric[i, j]
                if not np.isfinite(val) or val == 0:
                    out[i, j] = ""
                else:
                    key = int(round(val))
                    out[i, j] = label_to_name.get(key, f"{roi_type_names[j]}_{key}")
    return out


def merge_brain_region_columns(voxel_info_numeric, voxel_info_str, roi_type_names):
    n_vox, n_roi = voxel_info_numeric.shape
    region_indices = [
        j for j in range(n_roi)
        if _is_binary_region_column(voxel_info_numeric, j, roi_type_names[j])
    ]
    if not region_indices:
        return voxel_info_str, roi_type_names
    merged = np.empty(n_vox, dtype=object)
    for i in range(n_vox):
        parts = [voxel_info_str[i, j] for j in region_indices if voxel_info_str[i, j]]
        merged[i] = "; ".join(parts) if parts else ""
    non_region = [j for j in range(n_roi) if j not in region_indices]
    new_names = [roi_type_names[j] for j in non_region] + ["brain_region"]
    new_matrix = np.empty((n_vox, len(new_names)), dtype=object)
    for k, j in enumerate(non_region):
        new_matrix[:, k] = voxel_info_str[:, j]
    new_matrix[:, len(non_region)] = merged
    return new_matrix, new_names


def load_exp_design(exp_path):
    if exp_path.endswith(".npz"):
        with np.load(exp_path, allow_pickle=False) as z:
            masterordering = z["masterordering"].flatten()
        subjectim = z["subjectim"] if "subjectim" in z else None
        return masterordering, subjectim
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError(
            "scipy is required to load .mat. Alternatively save masterordering to .npz: "
            "np.savez('exp_ordering.npz', masterordering=loadmat('nsd_expdesign.mat')['masterordering'])"
        )
    m = loadmat(exp_path, struct_as_record=False, squeeze_me=True)
    masterordering = np.atleast_1d(m["masterordering"]).flatten()
    subjectim = m["subjectim"] if "subjectim" in m else None
    if subjectim is not None:
        subjectim = np.atleast_2d(subjectim)
    return masterordering, subjectim


def get_roi_linear_indices(mask_vol):
    mask = np.asarray(mask_vol, dtype=bool).squeeze()
    if mask.ndim != 3:
        raise ValueError("mask must be 3D")
    ii_flat = np.where(mask.ravel())[0]
    return ii_flat


def get_ref_shape_from_beta(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        b = f["betas"][:]
    if b.ndim != 4:
        raise ValueError("betas must be 4D")
    if b.shape[0] == 750:
        return tuple(b.shape[1:4])
    return tuple(b.shape[:3])


def align_vol_to_ref(vol, ref_shape):
    vol = np.asarray(vol).squeeze()
    if vol.shape == ref_shape:
        return vol
    if vol.shape == (ref_shape[2], ref_shape[1], ref_shape[0]):
        return np.transpose(vol, (2, 1, 0))
    raise ValueError(
        f"ROI shape {vol.shape} cannot be aligned to ref_shape {ref_shape}"
    )


def list_roi_paths(base_path, subj_id, space="func1pt8mm"):
    roi_dir = os.path.join(
        base_path,
        f"sub{subj_id:02d}",
        space,
        "roi",
    )
    extra = []
    for name in ["brainmask.nii.gz", "floc_facestval.nii.gz"]:
        p = os.path.join(base_path, f"sub{subj_id:02d}", space, name)
        if os.path.isfile(p):
            extra.append(p)
    paths = []
    names = []
    if os.path.isdir(roi_dir):
        for f in sorted(os.listdir(roi_dir)):
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                paths.append(os.path.join(roi_dir, f))
                names.append(os.path.splitext(os.path.splitext(f)[0])[0])
    for p in extra:
        paths.append(p)
        names.append(os.path.splitext(os.path.splitext(os.path.basename(p))[0])[0])
    return paths, names


def _load_one_roi_column(args):
    path, ii_flat, ref_shape = args
    try:
        img = nib.load(path)
        vol = np.asarray(img.get_fdata()).squeeze()
        aligned = align_vol_to_ref(vol, ref_shape)
        return aligned.ravel()[ii_flat].astype(np.float32)
    except Exception:
        return None


def build_voxel_info(ii_flat, ref_shape, roi_paths, roi_names, n_workers=1):
    n_vox = len(ii_flat)
    n_roi = len(roi_paths)
    voxel_info = np.zeros((n_vox, n_roi), dtype=np.float32)
    if n_workers <= 1:
        for j, path in enumerate(roi_paths):
            img = nib.load(path)
            vol = np.asarray(img.get_fdata()).squeeze()
            try:
                aligned = align_vol_to_ref(vol, ref_shape)
            except ValueError:
                continue
            voxel_info[:, j] = aligned.ravel()[ii_flat].astype(np.float32)
    else:
        args_list = [(p, ii_flat, ref_shape) for p in roi_paths]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            columns = list(ex.map(_load_one_roi_column, args_list))
        for j, col in enumerate(columns):
            if col is not None:
                voxel_info[:, j] = col
    return voxel_info, roi_names


def make_example06_roi_mask(base_path, subj_id, space="func1pt8mm"):
    floc_path = os.path.join(
        base_path,
        f"sub{subj_id:02d}",
        space,
        "floc_facestval.nii.gz",
    )
    if not os.path.isfile(floc_path):
        raise FileNotFoundError(
            f"example06 ROI requires floc_facestval.nii.gz at {floc_path}"
        )
    img = nib.load(floc_path)
    vol = np.asarray(img.get_fdata()).squeeze()
    boxvol = np.zeros(vol.shape, dtype=bool)
    lrix = slice(57, 66)
    paix = slice(30, 54)
    isix = slice(24, 32)
    boxvol[lrix, paix, isix] = True
    mask = boxvol & (vol > 5)
    return mask


def load_roi_betas_session(hdf5_path, ii_flat, psc_scale=300.0):
    with h5py.File(hdf5_path, "r") as f:
        betas = f["betas"][:]
    if betas.ndim != 4:
        raise ValueError("betas must be 4D (e.g. 750 x X x Y x Z)")
    if betas.shape[0] == 750:
        betas = np.transpose(betas, (1, 2, 3, 0))
    vol = np.asarray(betas, dtype=np.float64) / psc_scale
    flat = vol.reshape(-1, vol.shape[-1])
    roi = flat[ii_flat, :]
    return roi


def zscore_per_session(data_3d, axis=1):
    out = np.zeros_like(data_3d, dtype=np.float64)
    for s in range(data_3d.shape[2]):
        slab = data_3d[:, :, s]
        mu = np.nanmean(slab, axis=axis, keepdims=True)
        sigma = np.nanstd(slab, axis=axis, keepdims=True)
        sigma[sigma == 0] = np.nan
        out[:, :, s] = (slab - mu) / sigma
    return out


def zscore_2d(slab, axis=1):
    mu = np.nanmean(slab, axis=axis, keepdims=True)
    sigma = np.nanstd(slab, axis=axis, keepdims=True)
    sigma[sigma == 0] = np.nan
    return ((slab - mu) / sigma).astype(np.float32)


def build_trial_averaged(
    base_path,
    subj_id=1,
    nsess=40,
    beta_version="betas_fithrf",
    roi_path=None,
    exp_path=None,
    include_all_reps=False,
    n_workers=1,
    voxel_chunk_size=0,
    space="func1pt8mm",
):
    if exp_path is None:
        exp_npz = os.path.join(base_path, "nsd_expdesign_ordering.npz")
        exp_mat = os.path.join(base_path, "nsd_expdesign.mat")
        exp_path = exp_npz if os.path.isfile(exp_npz) else exp_mat
    subject_label = f"sub{subj_id:02d}"
    beta_dir = os.path.join(
        base_path,
        f"{subject_label}_betas",
        space,
        beta_version,
    )
    first_beta = os.path.join(beta_dir, "betas_session01.hdf5")
    if not os.path.isfile(first_beta):
        raise FileNotFoundError(f"Beta file not found: {first_beta}")
    ref_shape = get_ref_shape_from_beta(first_beta)

    if roi_path is None:
        mask_native = make_example06_roi_mask(base_path, subj_id, space=space)
        mask = align_vol_to_ref(mask_native.astype(np.float64), ref_shape)
        ii_flat = get_roi_linear_indices(mask > 0)
    else:
        roi_img = nib.load(roi_path)
        roi_vol = np.asarray(roi_img.get_fdata()).squeeze()
        roi_aligned = align_vol_to_ref(roi_vol, ref_shape)
        ii_flat = get_roi_linear_indices(roi_aligned > 0)
    n_vox = len(ii_flat)

    roi_paths, roi_names = list_roi_paths(base_path, subj_id, space=space)
    voxel_info, roi_type_names = build_voxel_info(
        ii_flat, ref_shape, roi_paths, roi_names, n_workers=n_workers
    )
    if roi_path is not None:
        roi_basename = os.path.basename(roi_path).replace(".nii.gz", "").replace(".nii", "")
        if roi_basename in roi_type_names:
            j = roi_type_names.index(roi_basename)
            if not (voxel_info[:, j] > 0).all():
                raise RuntimeError(
                    "voxel_info column for extraction ROI has zeros; alignment may be wrong"
                )

    masterordering, subjectim = load_exp_design(exp_path)
    n_trials_total = 750 * nsess
    theorder = masterordering[:n_trials_total]
    uniqueix = np.unique(theorder)
    im_id_to_col = {int(uid): i for i, uid in enumerate(uniqueix)}
    n_im = len(uniqueix)
    run_count = np.zeros(n_im, dtype=np.int32)
    for g in range(n_trials_total):
        run_count[im_id_to_col[int(theorder[g])]] += 1

    chunk_size = voxel_chunk_size if voxel_chunk_size > 0 else n_vox
    newdata_list = []
    if include_all_reps:
        keep = run_count > 0
    else:
        keep = run_count == 3
    cols_keep = np.where(keep)[0]

    for chunk_start in range(0, n_vox, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_vox)
        n_chunk = chunk_end - chunk_start
        run_sum = np.zeros((n_chunk, n_im), dtype=np.float32)

        for sess in range(1, nsess + 1):
            fpath = os.path.join(beta_dir, f"betas_session{sess:02d}.hdf5")
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Beta file not found: {fpath}")
            roi_betas = load_roi_betas_session(fpath, ii_flat)
            slab = roi_betas[chunk_start:chunk_end, :]
            del roi_betas
            dataZ = zscore_2d(slab, axis=1)
            for t in range(750):
                g = (sess - 1) * 750 + t
                col = im_id_to_col[int(theorder[g])]
                run_sum[:, col] += dataZ[:, t]
            del dataZ

        if len(cols_keep) == 0:
            newdata_list.append(np.empty((n_chunk, 0), dtype=np.float32))
        else:
            out_chunk = np.zeros((n_chunk, len(cols_keep)), dtype=np.float32)
            for i, c in enumerate(cols_keep):
                cnt = run_count[c]
                out_chunk[:, i] = run_sum[:, c] / cnt
            newdata_list.append(out_chunk)
        del run_sum

    newdata = np.vstack(newdata_list) if newdata_list else np.empty((n_vox, 0), dtype=np.float32)
    if len(cols_keep) > 0:
        newdatastim = uniqueix[cols_keep].astype(np.int64)
        if subjectim is not None and subj_id <= subjectim.shape[0]:
            idx_10k = (newdatastim - 1).astype(np.intp)
            newdatastim_73k = np.asarray(
                subjectim[subj_id - 1, idx_10k], dtype=np.int64
            ).flatten()
        else:
            newdatastim_73k = None
    else:
        newdatastim = np.array([], dtype=np.int64)
        newdatastim_73k = None
    del run_count

    return newdata, newdatastim, newdatastim_73k, theorder, voxel_info, roi_type_names


def main():
    parser = argparse.ArgumentParser(
        description="Build (n_voxel, n_stimuli) trial-averaged response and save newdata, newdatastim."
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        default="/media/ubuntu/sda/naturescenedataset",
        help="Base path containing sub01_betas, sub01, nsd_expdesign.mat",
    )
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--nsess", type=int, default=40)
    parser.add_argument(
        "--beta_version",
        type=str,
        default="betas_fithrf",
        help="Match example06: betas_fithrf; use betas_fithrf_GLMdenoise_RR if only HDF5 available",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Custom ROI NIfTI (default: example06 ROI = floc_facestval box & t>5)",
    )
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument(
        "--include_all_reps",
        action="store_true",
        help="Include images with 1 or 2 trials (default: only 3 trials, like example06)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path (default: base_path/trial_averaged_sub01.npz)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Parallel workers for building voxel_info (default 1)",
    )
    parser.add_argument(
        "--voxel_chunk",
        type=int,
        default=0,
        help="Process voxels in chunks of this size to reduce memory (default 0 = all)",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="func1pt8mm",
        choices=["func1pt8mm", "func1mm"],
        help="Volume space: func1pt8mm (1.8mm) or func1mm (1mm, default func1pt8mm)",
    )
    args = parser.parse_args()

    newdata, newdatastim, newdatastim_73k, theorder, voxel_info, roi_type_names = build_trial_averaged(
        args.base_path,
        subj_id=args.subj,
        nsess=args.nsess,
        beta_version=args.beta_version,
        roi_path=args.roi,
        exp_path=args.exp,
        include_all_reps=args.include_all_reps,
        n_workers=args.n_workers,
        voxel_chunk_size=args.voxel_chunk,
        space=args.space,
    )

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            args.base_path,
            f"trial_averaged_sub{args.subj:02d}_{args.space}.npz",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    roi_type_names_arr = np.array(roi_type_names, dtype=object)
    save_dict = dict(
        newdata=newdata,
        newdatastim=newdatastim,
        voxel_info=voxel_info,
        roi_type_names=roi_type_names_arr,
    )
    if newdatastim_73k is not None:
        save_dict["newdatastim_73k"] = newdatastim_73k
    np.savez(out_path, allow_pickle=True, **save_dict)
    msg = (
        f"Saved newdata {newdata.shape}, newdatastim (10k) {newdatastim.shape}, "
        f"voxel_info {voxel_info.shape} -> {out_path}"
    )
    if newdatastim_73k is not None:
        msg += "; newdatastim_73k (image ID for nsd_stimuli.hdf5 /imgBrick)"
    print(msg)


if __name__ == "__main__":
    main()
