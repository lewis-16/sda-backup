# General libraries
import numpy as np
import os


def generate_beta_burst(fs, duration, freq, burst_prob, amplitude, freq_jitter=0.5):
    """
    Generate a beta burst signal.

    Parameters:
        fs (int): Sampling frequency in Hz.
        duration (float): Duration in seconds.
        freq (float): Central frequency of the beta burst.
        burst_prob (float): Probability of burst at each time window.
        amplitude (float): Base amplitude of the burst.
        freq_jitter (float): Random frequency jitter for each burst.

    Returns:
        np.ndarray: Generated beta burst signal.
    """
    n_samples = int(fs * duration)
    signal = np.zeros(n_samples)
    window_size = int(fs * 0.5)  # 0.5s windows for bursts

    for start in range(0, n_samples, window_size):
        if np.random.rand() < burst_prob:
            f = freq + np.random.uniform(-freq_jitter, freq_jitter)
            t = np.arange(0, window_size) / fs
            burst = amplitude * np.sin(2 * np.pi * f * t) * np.hanning(window_size)
            signal[start:start + window_size] += burst[:min(window_size, n_samples - start)]

    return signal


def generate_noise(fs, duration, noise_level):
    """
    Generate Gaussian white noise.

    Parameters:
        fs (int): Sampling frequency.
        duration (float): Duration in seconds.
        noise_level (float): Standard deviation of the noise.

    Returns:
        np.ndarray: Noise signal.
    """
    n_samples = int(fs * duration)
    return np.random.normal(0, noise_level, n_samples)

def generate_spindle_bursts(fs, duration, freq=14, burst_prob=0.3, amplitude=1.0, freq_jitter=0.5):
    """
    Generate spindle bursts (~12–16 Hz) commonly seen in thalamus.

    Returns:
        np.ndarray: Spindle burst signal.
    """
    n_samples = int(fs * duration)
    signal = np.zeros(n_samples)
    window_size = int(fs * 1.0)  # 1 second windows

    for start in range(0, n_samples, window_size):
        if np.random.rand() < burst_prob:
            f = freq + np.random.uniform(-freq_jitter, freq_jitter)
            t = np.arange(0, window_size) / fs
            burst = amplitude * np.sin(2 * np.pi * f * t) * np.hanning(window_size)
            signal[start:start + window_size] += burst[:min(window_size, n_samples - start)]

    return signal

def generate_gamma_activity(fs, duration, freq=40, amplitude=0.3):
    """
    Generate continuous gamma oscillations.

    Returns:
        np.ndarray: Gamma signal.
    """
    t = np.arange(0, duration, 1/fs)
    return amplitude * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))

def generate_slow_oscillations(fs, duration, freq=1.0, amplitude=1.0, phase_jitter=True):
    """
    Generate slow oscillations (~1 Hz), as seen in VIM or sensory areas.

    Parameters:
        fs (int): Sampling frequency.
        duration (float): Duration in seconds.
        freq (float): Base frequency of the oscillation.
        amplitude (float): Amplitude of the oscillation.
        phase_jitter (bool): Whether to randomize phase slightly.

    Returns:
        np.ndarray: Slow oscillation signal.
    """
    t = np.arange(0, duration, 1/fs)
    phase = np.random.uniform(0, 2*np.pi) if phase_jitter else 0
    return amplitude * np.sin(2 * np.pi * freq * t + phase)

def generate_gpi_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes=4):
    """
    Generate GPi region signal with beta bursts and additive noise.

    Parameters:
        subject_params (dict): Subject-specific parameters.
        session_params (dict): Session-specific parameters.
        electrode_params (list of dict): List of per-electrode parameters.
        duration (float): Recording duration in seconds.
        fs (int): Sampling frequency in Hz.
        n_electrodes (int): Number of electrodes.

    Returns:
        np.ndarray: Simulated GPi signal of shape (n_electrodes, n_samples).
    """
    signals = []
    for i in range(n_electrodes):
        beta = generate_beta_burst(
            fs=fs,
            duration=duration,
            freq=20 + subject_params.get("beta_freq_offset", 0),
            burst_prob=subject_params.get("beta_burst_prob", 0.4),
            amplitude=subject_params.get("beta_gain", 1.0) * electrode_params[i].get("gain", 1.0)
        )

        noise = generate_noise(
            fs=fs,
            duration=duration,
            noise_level=subject_params.get("noise_level", 0.2)
                      + session_params.get("session_noise", 0.1)
                      + electrode_params[i].get("electrode_noise", 0.1)
        )

        signal = beta + noise
        signals.append(signal)

    return np.stack(signals)


def generate_vo_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes=4):
    """
    Generate VO region signal with spindle bursts.

    Returns:
        np.ndarray: Simulated VO signal.
    """
    signals = []
    for i in range(n_electrodes):
        spindles = generate_spindle_bursts(
            fs=fs,
            duration=duration,
            freq=14 + subject_params.get("spindle_freq_offset", 0),
            burst_prob=subject_params.get("spindle_burst_prob", 0.3),
            amplitude=subject_params.get("spindle_gain", 1.0) * electrode_params[i].get("gain", 1.0)
        )
        noise = generate_noise(
            fs=fs,
            duration=duration,
            noise_level=subject_params.get("noise_level", 0.2)
                      + session_params.get("session_noise", 0.1)
                      + electrode_params[i].get("electrode_noise", 0.1)
        )
        signal = spindles + noise
        signals.append(signal)
    return np.stack(signals)

def generate_motor_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes=4):
    """
    Generate Motor Cortex signal with gamma activity and mu suppression.

    Returns:
        np.ndarray: Simulated Motor signal.
    """
    signals = []
    for i in range(n_electrodes):
        gamma = generate_gamma_activity(
            fs=fs,
            duration=duration,
            freq=40 + subject_params.get("gamma_freq_offset", 0),
            amplitude=subject_params.get("gamma_gain", 0.5) * electrode_params[i].get("gain", 1.0)
        )
        noise = generate_noise(
            fs=fs,
            duration=duration,
            noise_level=subject_params.get("noise_level", 0.2)
                      + session_params.get("session_noise", 0.1)
                      + electrode_params[i].get("electrode_noise", 0.1)
        )
        signal = gamma + noise
        signals.append(signal)
    return np.stack(signals)

def generate_stn_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes=4):
    """
    Generate STN region signal with beta and HFO.

    Returns:
        np.ndarray: Simulated STN signal.
    """
    signals = []
    for i in range(n_electrodes):
        beta = generate_beta_burst(
            fs=fs,
            duration=duration,
            freq=20 + subject_params.get("beta_freq_offset", 0),
            burst_prob=subject_params.get("beta_burst_prob", 0.4),
            amplitude=subject_params.get("beta_gain", 1.0) * electrode_params[i].get("gain", 1.0)
        )
        hfo = generate_gamma_activity(
            fs=fs,
            duration=duration,
            freq=150 + subject_params.get("hfo_freq_offset", 0),
            amplitude=subject_params.get("hfo_gain", 0.2) * electrode_params[i].get("gain", 1.0)
        )
        noise = generate_noise(
            fs=fs,
            duration=duration,
            noise_level=subject_params.get("noise_level", 0.2)
                      + session_params.get("session_noise", 0.1)
                      + electrode_params[i].get("electrode_noise", 0.1)
        )
        signal = beta + hfo + noise
        signals.append(signal)
    return np.stack(signals)

def generate_vim_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes=4):
    """
    Generate VIM region signal with slow oscillations.

    Returns:
        np.ndarray: Simulated VIM signal.
    """
    signals = []
    for i in range(n_electrodes):
        slow = generate_slow_oscillations(
            fs=fs,
            duration=duration,
            freq=1.0 + subject_params.get("slow_freq_offset", 0),
            amplitude=subject_params.get("slow_gain", 1.0) * electrode_params[i].get("gain", 1.0),
            phase_jitter=True
        )
        noise = generate_noise(
            fs=fs,
            duration=duration,
            noise_level=subject_params.get("noise_level", 0.2)
                      + session_params.get("session_noise", 0.1)
                      + electrode_params[i].get("electrode_noise", 0.1)
        )
        signal = slow + noise
        signals.append(signal)
    return np.stack(signals)

def simulate_full_dataset(n_subjects=10, n_sessions=2, regions=None, duration=10, fs=500, n_electrodes_per_region=4, output_dir="simulated_dataset"):
    """
    Updated version to simulate full dataset including multiple regions.

    Parameters:
        n_subjects (int): Number of subjects.
        n_sessions (int): Sessions per subject.
        regions (list): List of region names (e.g., ["GPi", "STN", "VO", "Motor", "VIM"]).
        duration (float): Duration of recording in seconds.
        fs (int): Sampling frequency.
        n_electrodes_per_region (int): Number of electrodes per region.
        output_dir (str): Directory to save the dataset.

    Returns:
        dict: Metadata about the simulation.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_metadata = []

    for subject_id in range(n_subjects):
        print("creating the dataset for: Subject", subject_id)
        subject_dir = os.path.join(output_dir, f"subject_{subject_id}")
        os.makedirs(subject_dir, exist_ok=True)

        # Subject-level variability
        subject_params = {
            "beta_gain": np.random.uniform(0.8, 1.5),
            "beta_freq_offset": np.random.uniform(-2.0, 2.0),
            "beta_burst_prob": np.random.uniform(0.3, 0.7),
            "spindle_gain": np.random.uniform(0.8, 1.3),
            "spindle_freq_offset": np.random.uniform(-1.0, 1.0),
            "spindle_burst_prob": np.random.uniform(0.2, 0.5),
            "gamma_gain": np.random.uniform(0.3, 0.7),
            "gamma_freq_offset": np.random.uniform(-5.0, 5.0),
            "hfo_gain": np.random.uniform(0.1, 0.3),
            "hfo_freq_offset": np.random.uniform(-10.0, 10.0),
            "slow_gain": np.random.uniform(0.8, 1.2),
            "slow_freq_offset": np.random.uniform(-0.3, 0.3),
            "noise_level": np.random.uniform(0.3, 0.4)
        }

        for session_id in range(n_sessions):
            session_data = {}
            session_params = {
                "session_noise": np.random.uniform(0.05, 0.15)
            }

            for region in regions:
                electrode_params = [
                    {"gain": np.random.uniform(0.9, 1.1), "electrode_noise": np.random.uniform(0.03, 0.1)}
                    for _ in range(n_electrodes_per_region)
                ]

                if region == "GPi":
                    signal = generate_gpi_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes_per_region)
                elif region == "STN":
                    signal = generate_stn_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes_per_region)
                elif region == "VO":
                    signal = generate_vo_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes_per_region)
                elif region == "Motor":
                    signal = generate_motor_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes_per_region)
                elif region == "VIM":
                    signal = generate_vim_signal(subject_params, session_params, electrode_params, duration, fs, n_electrodes_per_region)
                else:
                    signal = np.zeros((n_electrodes_per_region, int(duration * fs)))

                session_data[region] = signal

                dataset_metadata.append({
                    "subject": subject_id,
                    "session": session_id,
                    "region": region,
                    "signal_shape": signal.shape
                })

            save_path = os.path.join(subject_dir, f"session_{session_id}.npz")
            np.savez(save_path, **session_data)

    return dataset_metadata

__all__ = ['generate_beta_burst', 'generate_noise', 'generate_spindle_bursts', 'generate_gamma_activity', 'generate_slow_oscillations', 'generate_gpi_signal', 'generate_vo_signal', 'generate_motor_signal', 'generate_stn_signal', 'generate_vim_signal', 'simulate_full_dataset']
