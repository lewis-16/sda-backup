import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, iirnotch, welch

def lowpass_filter(signal, cutoff, fs, order=2):
    """
    Apply a low-pass Butterworth filter to a 1D signal.

    Args:
        signal (np.ndarray): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Low-pass filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

def notch_filter(signal, fs, freq=60.0, quality=30.0):
    """
    Apply a notch filter to remove a specific frequency (e.g., 60 Hz powerline noise).

    Args:
        signal (np.ndarray): 1D or 2D signal.
        fs (float): Sampling frequency in Hz.
        freq (float): Frequency to notch out (default: 60 Hz).
        quality (float): Quality factor of the filter.

    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = iirnotch(freq, quality, fs)
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    elif signal.ndim == 2:
        print("signal 2D")
        return np.array([filtfilt(b, a, ch) for ch in signal])
    else:
        raise ValueError("Signal must be 1D or 2D")


def filter_and_resample(signal, original_fs=24000, target_fs=1000, cutoff=500):
    """
    Apply low-pass filtering and downsample the signal to a new sampling rate.

    Args:
        signal (np.ndarray): Input 1D signal.
        original_fs (int): Original sampling rate.
        target_fs (int): Target sampling rate.
        cutoff (float): Cutoff frequency for anti-aliasing low-pass filter.

    Returns:
        np.ndarray: Filtered and resampled signal.
    """
    
    # 1. Low-pass filter to prevent aliasing
    filtered = lowpass_filter(signal, cutoff=cutoff, fs=original_fs)

    # 2. Resample to target_fs
    duration_sec = len(signal) / original_fs
    target_num_samples = int(duration_sec * target_fs)
    resampled = resample(filtered, target_num_samples)

    return resampled



def plot_psd(signal, fs, title="Power Spectral Density", fmax=100):
    """
    Plot the Power Spectral Density (PSD) of a signal using Welch's method.

    Args:
        signal (np.ndarray): 1D signal.
        fs (float): Sampling rate in Hz.
        title (str): Title of the plot.
        fmax (float): Max frequency shown in the plot.

    Returns:
        None
    """
    f, Pxx = welch(signal, fs=fs, nperseg=fs*10)  # ~0.5s segments

    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx)
    plt.axvline(60, color='red', linestyle='--', label='60 Hz')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.xlim([0, fmax])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def normalize_signal(signal, method='zscore'):
    """
    Normalize a 1D or 2D signal using z-score or robust MAD-based normalization.

    Args:
        signal (np.ndarray): 1D or 2D array. If 2D, assumes shape (channels, time).
        method (str): Normalization method, either 'zscore' or 'robust'.

    Returns:
        np.ndarray: Normalized signal.

    Raises:
        ValueError: If an unsupported normalization method is provided.
    """
    if method == 'zscore':
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True)
        return (signal - mean) / std

    elif method == 'robust':
        median = np.median(signal, axis=-1, keepdims=True)
        mad = np.median(np.abs(signal - median), axis=-1, keepdims=True)
        robust_std = 1.4826 * mad
        return (signal - median) / robust_std

    else:
        raise ValueError("Unsupported method. Use 'zscore' or 'robust'.")