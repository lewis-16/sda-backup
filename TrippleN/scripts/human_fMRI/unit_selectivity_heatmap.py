import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat


def _infer_region_name(path: Path) -> str:
    name = path.stem
    if name.startswith("resp_"):
        name = name[len("resp_") :]
    if name.endswith("_func1py8mm"):
        name = name[: -len("_func1py8mm")]
    return name


def _load_cluster_info(mat_path: Path) -> np.ndarray:
    m = loadmat(str(mat_path))
    if "Cluster_idx" not in m:
        raise KeyError(f"Cluster_idx not found in {mat_path}")
    cluster = np.asarray(m["Cluster_idx"]).flatten()
    return cluster


def _normalize_to_pm1(R_raw: np.ndarray) -> np.ndarray:
    rmin = R_raw.min(axis=1, keepdims=True)
    rmax = R_raw.max(axis=1, keepdims=True)
    return 2.0 * (R_raw.astype(np.float64) - rmin) / (rmax - rmin + 1e-10) - 1.0


def compute_si_by_cluster(responses_1000: np.ndarray, cluster_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.unique(cluster_flat)
    n_units = responses_1000.shape[0]
    n_stim = responses_1000.shape[1]
    if cluster_flat.shape[0] != n_stim:
        raise ValueError(f"cluster length {cluster_flat.shape[0]} != n_stim {n_stim}")

    R = _normalize_to_pm1(responses_1000)
    n_clusters = len(cluster_ids)
    SI_clusters = np.zeros((n_units, n_clusters), dtype=np.float64)

    for j in range(n_clusters):
        mask_c = (cluster_flat == cluster_ids[j])
        mask_n = ~mask_c
        m_cat = R[:, mask_c].mean(axis=1)
        m_non = R[:, mask_n].mean(axis=1)
        v_cat = R[:, mask_c].var(axis=1)
        v_non = R[:, mask_n].var(axis=1)
        SI_clusters[:, j] = (m_cat - m_non) / np.sqrt(0.5 * (v_cat + v_non) + 1e-10)

    return SI_clusters, cluster_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="/media/ubuntu/sda/TrippleN/human_fMRI")
    ap.add_argument("--cluster_mat", type=str, default="/media/ubuntu/sda/TrippleN/ClusInfo.mat")
    ap.add_argument("--out_dir", type=str, default="/media/ubuntu/sda/TrippleN/customize/human_fMRI/selectivity")
    ap.add_argument("--selectivity_threshold", type=float, default=0.3)
    ap.add_argument("--n_plot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cluster_flat = _load_cluster_info(Path(args.cluster_mat))

    files = sorted(in_dir.glob("resp_*_func1py8mm.npy"))
    if len(files) == 0:
        raise FileNotFoundError(f"No resp_*_func1py8mm.npy found in {in_dir}")

    rows = []
    si_mats = []
    region_offsets = []
    cursor = 0

    for p in files:
        region = _infer_region_name(p)
        arr = np.load(str(p), allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{p} expected 2D array, got shape {arr.shape}")
        SI_clusters, cluster_ids = compute_si_by_cluster(arr, cluster_flat)
        n_units = SI_clusters.shape[0]
        n_clusters = SI_clusters.shape[1]

        df = pd.DataFrame(SI_clusters, columns=[f"C{j}_SI" for j in range(n_clusters)])
        for j in range(n_clusters):
            df[f"C{j}_selectivity"] = (df[f"C{j}_SI"] > args.selectivity_threshold).astype(int)
        df.insert(0, "unit_index", np.arange(n_units, dtype=int))
        df.insert(0, "region", region)
        df.insert(0, "global_unit_id", np.arange(cursor, cursor + n_units, dtype=int))
        rows.append(df)

        si_mats.append(SI_clusters)
        region_offsets.append((region, cursor, cursor + n_units))
        cursor += n_units

    unit_info = pd.concat(rows, ignore_index=True)
    SI_all = np.concatenate(si_mats, axis=0)

    n_clusters = SI_all.shape[1]
    si_cols = [f"C{j}_SI" for j in range(n_clusters)]
    sel_cols = [f"C{j}_selectivity" for j in range(n_clusters)]

    unit_csv = out_dir / "human_fMRI_unit_selectivity.csv"
    unit_pkl = out_dir / "human_fMRI_unit_selectivity.pkl"
    si_csv = out_dir / "human_fMRI_SI_matrix.csv"
    sel_csv = out_dir / "human_fMRI_selectivity_matrix_binary.csv"
    meta_csv = out_dir / "human_fMRI_unit_index_map.csv"

    unit_info.to_csv(unit_csv, index=False)
    unit_info.to_pickle(unit_pkl)
    pd.DataFrame(SI_all, columns=si_cols).to_csv(si_csv, index=False)
    pd.DataFrame(unit_info[sel_cols].to_numpy(dtype=int), columns=sel_cols).to_csv(sel_csv, index=False)
    pd.DataFrame(region_offsets, columns=["region", "global_unit_id_start", "global_unit_id_end"]).to_csv(meta_csv, index=False)

    idx_plot = np.arange(SI_all.shape[0])
    if SI_all.shape[0] > args.n_plot:
        rng = np.random.default_rng(args.seed)
        idx_plot = rng.choice(SI_all.shape[0], size=args.n_plot, replace=False)
    SI_plot = SI_all[idx_plot]

    sns.set_style("white")
    plt.rcParams["axes.grid"] = False

    g = sns.clustermap(
        SI_plot,
        row_cluster=False,
        col_cluster=True,
        method="average",
        cmap="RdBu_r",
        center=0,
        figsize=(8, 10),
        yticklabels=False,
        xticklabels=[f"C{j}" for j in range(n_clusters)],
    )
    g.ax_heatmap.set_xlabel("Cluster")
    g.ax_heatmap.set_ylabel("Unit")
    plt.suptitle("Unit selectivity (SI) heatmap (subsampled)", y=1.02)

    out_pdf = fig_dir / "human_fMRI_SI_heatmap.pdf"
    g.fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(g.fig)

    print("Saved:", unit_csv)
    print("Saved:", unit_pkl)
    print("Saved:", si_csv)
    print("Saved:", sel_csv)
    print("Saved:", meta_csv)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()

