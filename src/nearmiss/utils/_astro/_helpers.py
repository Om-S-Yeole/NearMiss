"""
Module for helper functions related to satellite attributes and comparisons.

This module provides utility functions to extract satellite attributes from `Satrec` objects and to check if two satellites are physically identical.
"""

import numpy as np
from datetime import datetime
from sgp4.api import Satrec
from nearmiss.utils._astro._constants import (
    EARTH_RADII,
    EARTH_SURFACE_VELOCITY,
)
from nearmiss.utils._astro._dataclasses import SingleSatInputAttributes


def satellite_attributes_from_Satrec_obj(
    sat: Satrec, D_start: datetime, tle_epoch: datetime, sat_rad: float
) -> SingleSatInputAttributes:
    """
    Extract satellite attributes from a `Satrec` object.

    Parameters
    ----------
    sat : Satrec
        Satellite object from the SGP4 library.
    D_start : datetime
        Start time of the analysis window.
    tle_epoch : datetime
        Epoch time of the TLE data.
    sat_rad: float
        Radius of satellite in m.

    Returns
    -------
    SingleSatInputAttributes
        A dataclass containing the extracted satellite attributes.

    Notes
    -----
    - The function calculates position and velocity vectors in the ECI frame and normalizes them.
    - The TLE age is calculated as the difference between `D_start` and `tle_epoch` in hours.
    """
    ndot = sat.ndot
    nddot = sat.nddot
    bstar = sat.bstar
    inclo = sat.inclo
    nodeo = sat.nodeo
    ecco = sat.ecco
    argpo = sat.argpo
    mo = sat.mo
    no_kozai = sat.no_kozai
    a = sat.a
    altp = sat.altp
    alta = sat.alta
    argpdot = sat.argpdot
    mdot = sat.mdot
    nodedot = sat.nodedot
    am = sat.am
    em = sat.em
    im = sat.im
    Om = sat.Om
    om = sat.om
    mm = sat.mm
    nm = sat.nm

    jd = np.atleast_1d(sat.jdsatepoch)
    fr = np.atleast_1d(sat.jdsatepochF)
    _, r, v = sat.sgp4_array(jd, fr)

    r_x = r[0][0] / EARTH_RADII
    r_y = r[0][1] / EARTH_RADII
    r_z = r[0][2] / EARTH_RADII

    v_x = v[0][0] / EARTH_SURFACE_VELOCITY
    v_y = v[0][1] / EARTH_SURFACE_VELOCITY
    v_z = v[0][2] / EARTH_SURFACE_VELOCITY

    tle_age = ((D_start - tle_epoch).total_seconds()) / 3600

    MLInputAttri = SingleSatInputAttributes(
        ndot,
        nddot,
        bstar,
        inclo,
        nodeo,
        ecco,
        argpo,
        mo,
        no_kozai,
        a,
        altp,
        alta,
        argpdot,
        mdot,
        nodedot,
        am,
        em,
        im,
        Om,
        om,
        mm,
        nm,
        r_x,
        r_y,
        r_z,
        v_x,
        v_y,
        v_z,
        tle_age,
        sat_rad,
    )
    return MLInputAttri


def sats_are_physically_identical(
    sat1: Satrec, sat2: Satrec, tol_km: float = 0.1
) -> bool:
    """
    Check if two satellites are physically identical based on their positions.

    Parameters
    ----------
    sat1 : Satrec
        First satellite object from the SGP4 library.
    sat2 : Satrec
        Second satellite object from the SGP4 library.
    tol_km : float, optional
        Tolerance in kilometers for position comparison. Default is 0.1 km.

    Returns
    -------
    bool
        True if the satellites are physically identical within the specified tolerance, False otherwise.

    Notes
    -----
    - The function propagates both satellites to their epoch time and compares their positions.
    - If either satellite fails propagation, the function returns False.
    """
    dt = sat1.jdsatepoch
    jd = int(dt)
    fr = dt - jd
    e1, r1, _ = sat1.sgp4(jd, fr)
    e2, r2, _ = sat2.sgp4(jd, fr)
    if e1 != 0 or e2 != 0:
        return False
    return np.linalg.norm(np.array(r1) - np.array(r2)) < tol_km
