"""
Export masterordering from nsd_expdesign.mat to .npz so nsd_trial_averaged_response.py
can run without scipy (e.g. when scipy and numpy 2 are incompatible).
Run once: python export_nsd_exp_ordering.py /path/to/naturescenedataset
"""
import sys
import os
import numpy as np

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/media/ubuntu/sda/naturescenedataset"
    mat_path = os.path.join(base, "nsd_expdesign.mat")
    out_path = os.path.join(base, "nsd_expdesign_ordering.npz")
    try:
        from scipy.io import loadmat
    except ImportError:
        print("scipy required: pip install 'numpy<2' scipy")
        sys.exit(1)
    m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    masterordering = np.atleast_1d(m["masterordering"]).flatten()
    subjectim = np.atleast_2d(m["subjectim"]) if "subjectim" in m else None
    if subjectim is not None:
        np.savez(out_path, masterordering=masterordering, subjectim=subjectim)
        print(f"Saved masterordering {masterordering.shape}, subjectim {subjectim.shape} -> {out_path}")
    else:
        np.savez(out_path, masterordering=masterordering)
        print(f"Saved masterordering shape {masterordering.shape} -> {out_path}")

if __name__ == "__main__":
    main()
