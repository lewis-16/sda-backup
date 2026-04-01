import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

STIMULI_DIR = '/media/ubuntu/sda/TrippleN/stimuli'
CLUSINFO_PATH = '/media/ubuntu/sda/TrippleN/ClusInfo.mat'
OUT_PDF = '/media/ubuntu/sda/TrippleN/cluster_images.pdf'

def main():
    cluster_info = loadmat(CLUSINFO_PATH)['Cluster_idx']
    cluster_flat = np.asarray(cluster_info).flatten()
    cluster_ids = np.unique(cluster_flat)

    image_files = sorted([f for f in os.listdir(STIMULI_DIR) if f.endswith('.bmp')])[:1000]
    if len(image_files) < 1000:
        raise RuntimeError(f'stimuli 下 .bmp 不足 1000 张，当前 {len(image_files)}')

    with PdfPages(OUT_PDF) as pdf:
        for cid in cluster_ids:
            idx = np.where(cluster_flat == cid)[0]
            paths = [os.path.join(STIMULI_DIR, image_files[i]) for i in idx]
            n = len(paths)
            ncol = 10
            nrow = (n + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(14, 1.4 * nrow))
            if nrow == 1 and ncol == 1:
                axes = np.array([[axes]])
            elif nrow == 1:
                axes = axes[np.newaxis, :]
            elif ncol == 1:
                axes = axes[:, np.newaxis]
            for ax in axes.flat:
                ax.set_axis_off()
            for k, (i, p) in enumerate(zip(idx, paths)):
                r, c = k // ncol, k % ncol
                try:
                    img = np.array(Image.open(p).convert('RGB'))
                    axes[r, c].imshow(img)
                    axes[r, c].set_title(str(i), fontsize=6)
                except Exception:
                    axes[r, c].text(0.5, 0.5, 'err', ha='center', va='center', fontsize=8)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            fig.suptitle(f'Cluster {int(cid)} (n={n})', fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f'已保存: {OUT_PDF}，共 {len(cluster_ids)} 页')

if __name__ == '__main__':
    main()
