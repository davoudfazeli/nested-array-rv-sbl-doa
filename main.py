"""
Demo runner for Nested Array RV-SBL Direction of Arrival (DOA) Estimation.
"""

import numpy as np
from scipy.signal import find_peaks

from src.array_models import create_nested_array, simulate_array_data
from src.rv_sbl import bayesian_dsp2020


def main():
    print("=" * 65)
    print(" Direction of Arrival Estimation: Real-Valued SBL for Nested Arrays")
    print("=" * 65)

    # 1. Config
    m1, m2 = 3, 2
    sensors = create_nested_array(m1=m1, m2=m2)
    true_angles = [-30.0, 20.0]
    snr_db = 15.0
    snapshots = 200

    print(f"Array Geometry (Sensors): {sensors.tolist()} (M = {len(sensors)})")
    print(f"True Source Angles:       {true_angles}")
    print(f"SNR:                      {snr_db} dB, Snapshots: {snapshots}\n")

    # 2. Simulate
    X = simulate_array_data(
        sensor_positions=sensors,
        angles_deg=true_angles,
        snr_db=snr_db,
        snapshots=snapshots,
        seed=42,
    )

    # 3. Estimate DOA
    gamma, refined_grid, iters = bayesian_dsp2020(
        X=X,
        K_sources=len(true_angles),
        sensor_positions=sensors,
        resolution=1.0,
    )

    peaks, _ = find_peaks(gamma, distance=5)
    if len(peaks) >= len(true_angles):
        top_idx = peaks[np.argsort(gamma[peaks])[::-1][: len(true_angles)]]
    else:
        top_idx = np.argsort(gamma)[::-1][: len(true_angles)]

    estimated_angles = np.sort(refined_grid[top_idx])
    errors = np.abs(np.sort(true_angles) - estimated_angles)

    # 4. Results
    print(f"Algorithm converged in:   {iters} iterations")
    print(f"Estimated DOA Angles:     {np.round(estimated_angles, 2).tolist()}")
    print(f"Absolute Errors (deg):    {np.round(errors, 4).tolist()}")
    print(f"Mean Absolute Error:      {np.mean(errors):.4f} deg")
    print("=" * 65)


if __name__ == "__main__":
    main()

