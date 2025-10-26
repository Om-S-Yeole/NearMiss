import numpy as np
from math import log, sqrt, erf
from datetime import datetime, timedelta
from poliastro.constants import GM_earth


def t_from_tau(t_n: datetime, tau: float, del_t: float) -> datetime:
    """
    Calculate the time corresponding to a given normalized time (tau).

    Parameters
    ----------
    t_n : datetime
        The initial time.
    tau : float
        The normalized time.
    del_t : float
        The time step in seconds.

    Returns
    -------
    datetime
        The calculated time.
    """
    return t_n + timedelta(seconds=tau * del_t)


def acc_vec_calculator(r: np.ndarray) -> np.ndarray:
    """
    Calculate the acceleration vector due to Earth's gravity.

    Parameters
    ----------
    r : np.ndarray
        Position vector in ECI frame.

    Returns
    -------
    np.ndarray
        Acceleration vector in ECI frame.
    """

    r_norm = np.linalg.norm(r)

    # Multiplication to GM_earth by 1e-9 is to convert units m^3/s^2 to km^3/s^2
    # This took me a 1 full day to find out why my algorithm was not working.
    return ((-GM_earth.value * 1e-9) / (r_norm**3)) * r


def ellipsoidal_function(r_d: np.ndarray, v_p: np.ndarray, a: float, b: float) -> float:
    """
    Calculate the ellipsoidal function value.

    Parameters
    ----------
    r_d : np.ndarray
        Relative position vector.
    v_p : np.ndarray
        Primary object's velocity vector.
    a : float
        Semi-major axis.
    b : float
        Semi-minor axis.

    Returns
    -------
    float
        Value of the ellipsoidal function.
    """
    return (
        (((r_d @ v_p) ** 2 / (v_p @ v_p)) / (a**2))
        + (((r_d @ r_d) - (r_d @ v_p) ** 2 / (v_p @ v_p)) / (b**2))
        - 1
    )


def max_prob_function(r_obj_1: float, r_obj_2: float, d_close: float) -> float:
    """
    Calculate the maximum probability of collision.

    Parameters
    ----------
    r_obj_1 : float
        Radius of the first object.
    r_obj_2 : float
        Radius of the second object.
    d_close : float
        Closest approach distance.

    Returns
    -------
    float
        Maximum probability of collision.
    """

    if r_obj_1 <= 0 or r_obj_2 <= 0:
        raise ValueError("Object radii must be positive non-zero values.")
    if d_close <= 0:
        return 1.0

    r = (r_obj_1 + r_obj_2) / d_close
    if r >= 1.0:
        return 1.0

    term = sqrt(-log((1 - r) / (1 + r)))
    return 0.5 * (
        erf(((r + 1) * term) / (2 * sqrt(r))) + erf(((r - 1) * term) / (2 * sqrt(r)))
    )
