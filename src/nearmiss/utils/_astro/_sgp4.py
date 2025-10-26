import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta


def propagate_sgp4(
    sat: Satrec, dt: float, epoch: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a satellite's state vectors using SGP4.

    Parameters
    ----------
    sat : Satrec
        Satellite object.
    dt : float
        Time delta in seconds from the epoch.
    epoch : datetime
        Epoch time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Propagated position and velocity vectors.

    Raises
    ------
    TypeError
        If `dt` is not a float or int.
    RuntimeError
        If SGP4 propagation fails.
    """
    if not isinstance(dt, (int, float)):
        raise TypeError(f"Expected dt as float or int, got {type(dt)}")

    t = epoch + timedelta(seconds=dt)
    jd, fr = jday(
        t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6
    )
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 propagation error code: {e}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def distance_squared(
    t_sec: float, sat1: Satrec, sat2: Satrec, epoch: datetime
) -> float:
    """
    Calculate the squared distance between two satellites at a given time.

    Parameters
    ----------
    t_sec : float
        Time in seconds from the epoch.
    sat1 : Satrec
        First satellite object.
    sat2 : Satrec
        Second satellite object.
    epoch : datetime
        Epoch time.

    Returns
    -------
    float
        Squared distance between the two satellites.
    """
    r1, v1 = propagate_sgp4(sat1, t_sec, epoch)
    r2, v2 = propagate_sgp4(sat2, t_sec, epoch)
    return float(np.sum((r2 - r1) ** 2))
