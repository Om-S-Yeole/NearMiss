import numpy as np
from numpy.polynomial.polynomial import Polynomial


def cubic_spline_root_finder(
    f_n: float, f_n_1: float, f_dot_n: float, f_dot_n_1: float, del_t: float
):
    """
    Find the roots of a cubic spline polynomial.

    Parameters
    ----------
    f_n : float
        Function value at the initial point.
    f_n_1 : float
        Function value at the next point.
    f_dot_n : float
        Derivative at the initial point.
    f_dot_n_1 : float
        Derivative at the next point.
    del_t : float
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Roots of the cubic spline polynomial.
    """
    alpha_0 = f_n
    alpha_1 = f_dot_n * del_t
    alpha_2 = -3 * f_n - 2 * f_dot_n * del_t + 3 * f_n_1 - f_dot_n_1 * del_t
    alpha_3 = 2 * f_n + f_dot_n * del_t - 2 * f_n_1 + f_dot_n_1 * del_t

    p = Polynomial([alpha_0, alpha_1, alpha_2, alpha_3])
    roots: np.ndarray = p.roots()
    return roots


def four_point_cubic_spline_root_finder(tau_arr: np.ndarray, p_arr: np.ndarray):
    """
    Find the roots of a cubic spline using four points.

    Parameters
    ----------
    tau_arr : np.ndarray
        Array of normalized time values.
    p_arr : np.ndarray
        Array of function values at the corresponding time values.

    Returns
    -------
    np.ndarray
        Roots of the cubic spline polynomial.
    """
    _, t_1, t_2, _ = tau_arr
    p_1, p_2, p_3, p_4 = p_arr
    DET = (
        (t_1**3) * (t_2**2)
        + (t_1**2) * (t_2)
        + (t_1) * (t_2**3)
        - (t_1**3) * (t_2)
        - (t_1**2) * (t_2**3)
        - (t_1) * (t_2**2)
    )

    alpha_0 = p_1
    alpha_1 = (
        (t_2**3 - t_2**2) * (p_2 - p_1)
        + (t_1**2 - t_1**3) * (p_3 - p_1)
        + (((t_1**3) * (t_2**2) - (t_1**2) * (t_2**3)) * (p_4 - p_1))
    ) / (DET)
    alpha_2 = (
        (t_2 - t_2**3) * (p_2 - p_1)
        + (t_1**3 - t_1) * (p_3 - p_1)
        + ((t_1 * t_2**3 - t_1**3 * t_2) * (p_4 - p_1))
    ) / (DET)
    alpha_3 = (
        ((t_2**2 - t_2) * (p_2 - p_1))
        + (t_1 - t_1**2) * (p_3 - p_1)
        + ((t_2 * t_1**2 - t_1 * t_2**2) * (p_4 - p_1))
    ) / (DET)

    p = Polynomial([alpha_0, alpha_1, alpha_2, alpha_3])
    roots: np.ndarray = p.roots()
    return roots


def quintic_polynomial_maker(
    f_n: float,
    f_n_1: float,
    f_dot_n: float,
    f_dot_n_1: float,
    f_dou_dot_n: float,
    f_dou_dot_n_1: float,
    del_t: float,
) -> Polynomial:
    """
    Create a quintic polynomial based on boundary conditions.

    Parameters
    ----------
    f_n : float
        Function value at the initial point.
    f_n_1 : float
        Function value at the next point.
    f_dot_n : float
        First derivative at the initial point.
    f_dot_n_1 : float
        First derivative at the next point.
    f_dou_dot_n : float
        Second derivative at the initial point.
    f_dou_dot_n_1 : float
        Second derivative at the next point.
    del_t : float
        Time step in seconds.

    Returns
    -------
    Polynomial
        The quintic polynomial.
    """
    alpha_0 = f_n
    alpha_1 = f_dot_n * del_t
    alpha_2 = 0.5 * f_dou_dot_n * (del_t**2)
    alpha_3 = (
        -10 * f_n
        - 6 * f_dot_n * del_t
        - 1.5 * f_dou_dot_n * (del_t**2)
        + 10 * f_n_1
        - 4 * f_dot_n_1 * del_t
        + 0.5 * f_dou_dot_n_1 * (del_t**2)
    )
    alpha_4 = (
        15 * f_n
        + 8 * f_dot_n * del_t
        + 1.5 * f_dou_dot_n * (del_t**2)
        - 15 * f_n_1
        + 7 * f_dot_n_1 * del_t
        - f_dou_dot_n_1 * (del_t**2)
    )
    alpha_5 = (
        -6 * f_n
        - 3 * f_dot_n * del_t
        - 0.5 * f_dou_dot_n * (del_t**2)
        + 6 * f_n_1
        - 3 * f_dot_n_1 * del_t
        + 0.5 * f_dou_dot_n_1 * (del_t**2)
    )

    p = Polynomial([alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5])
    return p
