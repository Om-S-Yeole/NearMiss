"""
Module for SGP4-based satellite propagation and distance calculations.

This module provides functions to propagate satellite state vectors using SGP4 and calculate distances between satellites.
"""

import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta


class SGP4Exception(Exception):
    """
    Custom exception for SGP4 propagation errors.

    Attributes
    ----------
    message : str
        Error message describing the SGP4 propagation failure.
    """

    def __init__(self, message="SGP4 propagation failed"):
        self.message = message
        super().__init__(self.message)


def propagate_sgp4(
    sat: Satrec, dt: float, epoch: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a satellite's state vectors using SGP4.

    Parameters
    ----------
    sat : Satrec
        Satellite object from the SGP4 library.
    dt : float
        Time delta in seconds from the epoch.
    epoch : datetime
        Epoch time.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Propagated position vector (in km) and velocity vector (in km/s).

    Raises
    ------
    TypeError
        If `dt` is not a float or int.
    SGP4Exception
        If SGP4 propagation fails.

    Notes
    -----
    - The function calculates the Julian date and fraction for the given time delta and propagates the satellite's state vectors.
    - If the SGP4 propagation returns a non-zero error code, an `SGP4Exception` is raised.
    """
    if not isinstance(dt, (int, float)):
        raise TypeError(f"Expected dt as float or int, got {type(dt)}")

    t = epoch + timedelta(seconds=dt)
    jd, fr = jday(
        t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6
    )
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise SGP4Exception(f"SGP4 propagation error code: {e}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def distance_squared(
    t_sec: float, sat1: Satrec, sat2: Satrec, D_start: datetime
) -> float:
    """
    Calculate the squared distance between two satellites at a given time.

    Parameters
    ----------
    t_sec : float
        Time in seconds from the epoch.
    sat1 : Satrec
        First satellite object from the SGP4 library.
    sat2 : Satrec
        Second satellite object from the SGP4 library.
    D_start : datetime
        Start time of the analysis window.

    Returns
    -------
    float
        Squared distance between the two satellites in km^2.

    Notes
    -----
    - The function propagates both satellites to the specified time and calculates the squared Euclidean distance between their positions.
    - If propagation fails for either satellite, an `SGP4Exception` is raised.
    """
    r1, _ = propagate_sgp4(sat1, t_sec, D_start)
    r2, _ = propagate_sgp4(sat2, t_sec, D_start)

    diff = r2 - r1
    dist2 = np.dot(diff, diff)
    return dist2
