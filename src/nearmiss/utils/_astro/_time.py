"""
Module for time-related utilities.

This module provides functions to convert between Julian Date and datetime objects, and to calculate Julian Date relative to the J2000 epoch.
"""

import numpy as np
from datetime import datetime, timedelta, timezone
from sgp4.api import jday


def jd_to_datetime(jd: float) -> datetime:
    """
    Convert Julian Date to a datetime object.

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    datetime
        Corresponding datetime object in UTC.

    Notes
    -----
    - The function accounts for fractional days and converts them to hours, minutes, and seconds.
    - The returned datetime object is timezone-aware and set to UTC.
    """
    jd += 0.5
    F, I = np.modf(jd)
    I = int(I)
    A = int((I - 1867216.25) / 36524.25)
    B = I + 1 + A - int(A / 4)
    C = B + 1524
    D = int((C - 122.1) / 365.25)
    E = int(365.25 * D)
    G = int((C - E) / 30.6001)
    day = C - E - int(30.6001 * G) + F
    month = G - 1 if G < 13.5 else G - 13
    year = D - 4716 if month > 2.5 else D - 4715
    return datetime(year, month, int(day), tzinfo=timezone.utc) + timedelta(
        days=day % 1
    )


def datetime_to_jd(dt: datetime) -> tuple[float, float, float]:
    """
    Convert a datetime object to Julian Date.

    Parameters
    ----------
    dt : datetime
        Datetime object to convert.

    Returns
    -------
    tuple[float, float, float]
        - Julian Date (float).
        - Whole part of the Julian Date (float).
        - Fractional part of the Julian Date (float).

    Notes
    -----
    - The function uses the SGP4 library's `jday` function for conversion.
    - The input datetime object must be timezone-aware.
    """
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day = whole_part + frac_part
    return julian_day, whole_part, frac_part


def datetime_to_jd_2000(dt: datetime) -> float:
    """
    Convert a datetime object to Julian Date relative to the J2000 epoch.

    Parameters
    ----------
    dt : datetime
        Datetime object to convert.

    Returns
    -------
    float
        Julian Date relative to the J2000 epoch.

    Notes
    -----
    - The J2000 epoch corresponds to Julian Date 2451545.0.
    - The function calculates the difference between the input datetime's Julian Date and the J2000 epoch.
    """
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day_2000 = whole_part + frac_part - 2451545.0
    return julian_day_2000
