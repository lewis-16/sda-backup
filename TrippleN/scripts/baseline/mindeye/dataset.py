import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def default_paths():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return {
        "neuron_responses": os.path.join(root, "customize", "neuron_responses_1000.npy"),
        "stimuli_dir": os.path.join(root, "stimuli"),
    }


class TrippleNDataset(Dataset):
    def __init__(
        self,
        neuron_responses_path=None,
        stimuli_dir=None,
        indices=None,
        n_max=1000,
    ):
        if neuron_responses_path is None:
            neuron_responses_path = default_paths()["neuron_responses"]
        if stimuli_dir is None:
            stimuli_dir = default_paths()["stimuli_dir"]

        self.stimuli_dir = stimuli_dir
        data = np.load(neuron_responses_path).astype(np.float32)
        if data.shape[0] == n_max and data.shape[1] != n_max:
            data = data.T
        elif data.shape[1] == n_max:
            data = data.T
        self.responses = data
        self.n_samples, self.n_neurons = self.responses.shape

        image_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith(".bmp")])[:n_max]
        if len(image_files) < self.n_samples:
            raise ValueError(
                f"stimuli has {len(image_files)} .bmp files, need at least {self.n_samples}"
            )
        self.image_files = image_files[: self.n_samples]

        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
        else:
            self.indices = np.arange(self.n_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        neuron_response = torch.from_numpy(self.responses[real_idx].copy())
        path = os.path.join(self.stimuli_dir, self.image_files[real_idx])
        image = Image.open(path).convert("RGB")
        image_tensor = (transforms.ToTensor()(image)[:3] - 0.5) / 0.5
        return neuron_response, image_tensor


def train_val_split(n_samples, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_train = int(n_samples * train_ratio)
    train_indices = idx[:n_train]
    val_indices = idx[n_train:]
    return train_indices, val_indices
