"""
Nested array sensor topology and multi-source signal generation.
"""

import numpy as np


def create_nested_array(m1: int = 3, m2: int = 2) -> np.ndarray:
    """Generates 2-level nested array sensor positions in units of lambda/2."""
    level1 = np.arange(m1 + 1)
    level2 = np.arange(2, m2 + 1) * (m1 + 1) - 1
    return np.r_[level1, level2]


def simulate_array_data(
    sensor_positions: np.ndarray,
    angles_deg: list,
    snr_db: float = 10.0,
    snapshots: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Generates narrow-band array data with additive complex Gaussian noise."""
    rng = np.random.default_rng(seed)
    M = len(sensor_positions)
    K = len(angles_deg)
    T = snapshots

    # Sources
    s = (
        rng.standard_normal((K, T)) + 1j * rng.standard_normal((K, T))
    ) / np.sqrt(2)

    # Steering matrix
    theta_rad = np.deg2rad(angles_deg)
    A = np.exp(
        -1j * np.pi * sensor_positions[:, None] * np.sin(theta_rad)[None, :]
    )

    # Signal and noise
    signal = A @ s
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (
        (rng.standard_normal((M, T)) + 1j * rng.standard_normal((M, T)))
        * np.sqrt(noise_power / 2.0)
    )

    return signal + noise

