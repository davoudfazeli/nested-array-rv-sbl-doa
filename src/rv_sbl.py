"""
Core Real-Valued Sparse Bayesian Learning (RV-SBL) with Off-Grid Refinement.
Faithful Python implementation of M.Sc. thesis research algorithms.
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.signal import find_peaks


def garddic(
    z: np.ndarray,
    grid_deg: np.ndarray,
    d_new: np.ndarray,
    Gamma: np.ndarray,
    wi: np.ndarray,
    n_iters: int = 10,
    lr: float = 0.01,
) -> np.ndarray:
    """Off-grid continuous refinement via gradient descent (garddic.m).

    Parameters
    ----------
    z : Virtual coarray measurement vector.
    grid_deg : Initial coarse peak angles in degrees.
    d_new : Sensor positions normalized by wavelength.
    Gamma : Prior variance matrix of the active sources.
    wi : Pre-whitening transform matrix (W_sq @ Q).
    n_iters : Number of gradient descent steps.
    lr : Learning rate.

    Returns
    -------
    Refined continuous angles in degrees.
    """
    grid = grid_deg.copy()

    for _ in range(n_iters):
        dteta = np.zeros(len(grid))
        theta_rad = np.deg2rad(grid)
        phase = -2j * np.pi * d_new[:, None] * np.sin(theta_rad)[None, :]
        B = wi @ np.exp(phase)

        dB = (
            -2j
            * np.pi
            * (d_new[:, None] * np.cos(theta_rad)[None, :])
            * (wi @ np.exp(phase))
        )

        Sigmaz = np.eye(B.shape[0]) + B @ Gamma @ B.conj().T
        Sigmaz_inv = np.linalg.pinv(Sigmaz)

        for c in range(len(grid)):
            dBi = np.zeros_like(B)
            dBi[:, c] = dB[:, c]
            dsig = dBi @ Gamma @ B.conj().T + B @ Gamma @ dBi.conj().T

            term1 = np.trace(Sigmaz_inv @ dsig)
            term2 = (z.conj().T @ Sigmaz_inv @ dsig @ Sigmaz_inv @ z).item()
            dteta[c] = np.real(term1 - term2)

        grid = grid - lr * dteta

    return grid


def bayesian_dsp2020(
    X: np.ndarray,
    K_sources: int,
    sensor_positions: np.ndarray,
    resolution: float = 1.0,
    search_range: tuple = (-90.0, 90.0),
    max_iter: int = 200,
    tol: float = 1e-3,
    etc: int = 4,
):
    """Real-Valued Sparse Bayesian Learning (RV-SBL) for nested arrays.

    Implements unitary transformation Q, covariance pre-whitening, and EM-SBL.
    """
    M, T = X.shape
    r = sensor_positions

    search_area = np.arange(
        search_range[0], search_range[1] + resolution, resolution
    )
    K_hat = len(search_area)

    # Sample covariance matrix
    Rx = (X @ X.conj().T) / T

    # Difference coarray mapping
    diffs = r[:, None] - r[None, :]
    diff_flat = diffs.flatten(order="F")
    unique_diffs = np.unique(diff_flat)
    M2 = len(unique_diffs)

    F_mat = np.zeros((M * M, M2))
    for idx, u in enumerate(unique_diffs):
        F_mat[diff_flat == u, idx] = 1.0

    finv = np.linalg.pinv(F_mat.T @ F_mat) @ F_mat.T
    pos_all = unique_diffs

    # Unitary transformation matrix Q
    n_half = (M2 - 1) // 2
    if 2 * n_half + 1 == M2:
        I_h = np.eye(n_half)
        J1 = np.fliplr(I_h)
        Z_col = np.zeros((n_half, 1))
        Q = (
            np.block([
                [I_h, Z_col, J1],
                [np.zeros((1, n_half)), np.sqrt(2), np.zeros((1, n_half))],
                [1j * J1, Z_col, -1j * I_h],
            ])
            / np.sqrt(2)
        )
    else:
        Q = np.eye(M2, dtype=complex)

    # Whitening transform
    Q_finv = Q @ finv
    kron_Rx = np.kron(Rx.T, Rx)
    W = np.real((Q_finv @ kron_Rx @ Q_finv.conj().T) / T)
    W_sq = np.real(sqrtm(np.linalg.pinv(W)))

    # Real-valued measurement vector Y and real dictionary Phi
    Rx_vec = Rx.flatten(order="F")
    Y = np.real(W_sq @ Q_finv @ Rx_vec)

    theta_rad = np.deg2rad(search_area)
    A_complex = np.exp(
        -1j * np.pi * pos_all[:, None] * np.sin(theta_rad)[None, :]
    )
    Phi = np.real(W_sq @ Q @ A_complex)

    # SBL Hyperparameter estimation loop
    gamma = np.ones(K_hat) * 0.1
    sigma2 = 1.0

    for it in range(max_iter):
        gamma_old = gamma.copy()

        Gamma = np.diag(gamma)
        Sigma_y = sigma2 * np.eye(len(Y)) + Phi @ Gamma @ Phi.T
        Sigma_y_inv = np.linalg.pinv(Sigma_y)

        mu = Gamma @ Phi.T @ (Sigma_y_inv @ Y)
        diag_Sigma = np.real(
            np.diag(Gamma - Gamma @ Phi.T @ Sigma_y_inv @ Phi @ Gamma)
        )

        gamma = (np.abs(mu) ** 2) + diag_Sigma

        if (
            np.linalg.norm(gamma - gamma_old) / (np.linalg.norm(gamma_old) + 1e-12)
            < tol
        ):
            break

    # Off-Grid Continuous Refinement
    dist_samples = max(1, int(round(5.0 / resolution)))
    peaks, _ = find_peaks(gamma, distance=dist_samples)

    if len(peaks) >= K_sources:
        top_candidates = peaks[np.argsort(gamma[peaks])[::-1][:etc]]
    else:
        top_candidates = np.argsort(gamma)[::-1][:etc]

    sub_grid = search_area[top_candidates]
    sub_Gamma = np.diag(gamma[top_candidates])
    d_pos = pos_all / 2.0
    wi = W_sq @ Q

    refined_sub = garddic(
        z=finv @ Rx_vec,
        grid_deg=sub_grid,
        d_new=d_pos,
        Gamma=sub_Gamma,
        wi=wi,
        n_iters=10,
    )

    refined_grid = search_area.copy()
    refined_grid[top_candidates] = refined_sub

    return gamma, refined_grid, it + 1

