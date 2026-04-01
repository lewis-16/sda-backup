import os
import re
import h5py
import torch
import warnings
from datetime import datetime
    

def get_device():
    """
    Return the available PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: Computation device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() == False:
        warnings.warn("⚠️ CUDA is not available. The model will run on CPU, which may be slower.", RuntimeWarning)
    return device


def create_timestamped_logdir(base_dir="runs", extension="", prefix ="siamese_" ):
    """
    Create a new log directory with a timestamp.

    Args:
        base_dir (str): Base directory to save logs.
        extension (str): Optional string to append to the log name.

    Returns:
        str: Full path to the new log directory.
    """
    timestamp = prefix + datetime.now().strftime("%Y-%m-%d_%H-%M") + extension
    log_dir = os.path.join(base_dir, f"{timestamp}")
    return log_dir


# Utility functions for parameter counting for models

def count_parameters(model):
    """
    Print the total number of trainable parameters in a model,
    along with a breakdown by layer name.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}\n")
    print("Trainable parameters by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} {param.numel():,}")


def print_model_summary(model, train_loader, device):
    """
    Run a sample forward pass through the model and print shape info and parameter count.

    Args:
        model (torch.nn.Module): The model whose summary is to be printed.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): Device on which the model and data are placed.
    """
    print("===== Sample Forward Pass Shape Info =====")
    x_sample = next(iter(train_loader))[0].to(device)
    model.encoder(x_sample)
    print("\nModel Summary:\n")
    print(model)
    count_parameters(model)
    

# Utility functions for handling MATLAB .mat recording files and parsing labels
def load_micro_labels(mat_path):
    """
    Load 'microLabels' from a MATLAB .mat file stored in HDF5 format.

    Args:
        mat_path (str): Path to the .mat file.

    Returns:
        list of str: List of decoded label strings.
    """
    with h5py.File(mat_path, 'r') as f:
        # Assumes 'microLabels' is a cell array of strings (char vectors)
        ref = f['microLabels'][0]
        labels = []
        for obj_ref in ref:
            obj = f[obj_ref]
            label = ''.join([chr(c[0]) for c in obj[:]])
            labels.append(label)
    return labels


def parse_label(label):
    """
    Parse a label string to extract region, side, and electrode number.

    Args:
        label (str): Raw label string (e.g. "microGPi1_L_5_CommonFiltered_lfs").

    Returns:
        tuple: (region: str, side: str, electrode: int)

    Raises:
        ValueError: If the label format is not recognized.
    """
    match = re.search(r'micro([A-Za-z]+[0-9]?)_([LR])_(\d+)', label)
    if not match:
        raise ValueError(f"Could not parse label: {label}")
    region = match.group(1)
    side = match.group(2)
    electrode = int(match.group(3))
    return region, side, electrode