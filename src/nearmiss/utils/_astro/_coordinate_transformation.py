import numpy as np


def RSW_to_ECI_covariance(
    r_eci: np.ndarray, v_eci: np.ndarray, cov_ric: np.ndarray
) -> np.ndarray:
    """
    Convert a covariance matrix from RSW to ECI frame.

    Parameters
    ----------
    r_eci : np.ndarray
        Position vector in ECI frame.
    v_eci : np.ndarray
        Velocity vector in ECI frame.
    cov_ric : np.ndarray
        Covariance matrix in RSW frame.

    Returns
    -------
    np.ndarray
        Covariance matrix in ECI frame.
    """
    r_hat = r_eci / np.linalg.norm(r_eci)
    h_vec = np.cross(r_eci, v_eci)
    w_hat = h_vec / np.linalg.norm(h_vec)
    s_hat = np.cross(w_hat, r_hat)
    M = np.vstack((r_hat, s_hat, w_hat)).T  # RSW -> ECI
    zero = np.zeros((3, 3))
    R_rot = np.block([[M, zero], [zero, M]])
    return R_rot @ cov_ric @ R_rot.T


def cov_matrix_from_ECI_to_NTW_frame_converter(
    r: np.ndarray, v: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """
    Convert a covariance matrix from ECI to NTW frame.

    Parameters
    ----------
    r : np.ndarray
        Position vector in ECI frame.
    v : np.ndarray
        Velocity vector in ECI frame.
    cov : np.ndarray
        Covariance matrix in ECI frame.

    Returns
    -------
    np.ndarray
        Covariance matrix in NTW frame.
    """
    h = np.cross(r, v)
    W_hat = h / np.linalg.norm(h)
    T_hat = v / np.linalg.norm(v)
    N_hat = np.cross(T_hat, W_hat)
    M = np.vstack((T_hat, W_hat, N_hat))
    zero = np.zeros((3, 3))
    R_rot = np.block([[M, zero], [zero, M]])
    cov_NTW = R_rot @ cov @ R_rot.T
    return cov_NTW


def semi_major_minor_axis_from_cov_NTW(cov_NTW: np.ndarray) -> tuple[float, float]:
    """
    Extract semi-major and semi-minor axes from a covariance matrix in NTW frame.

    Parameters
    ----------
    cov_NTW : np.ndarray
        Covariance matrix in NTW frame.

    Returns
    -------
    tuple[float, float]
        Semi-major and semi-minor axes.
    """
    cov_2d = cov_NTW[0:2, 0:2]
    eigvals = np.linalg.eigvals(cov_2d)
    semi_axes = np.sqrt(eigvals)
    return (np.max(semi_axes), np.min(semi_axes))
