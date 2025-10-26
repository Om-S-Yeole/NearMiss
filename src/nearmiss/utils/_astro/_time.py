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
        Corresponding datetime object.
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


def datetime_to_jd_2000(dt: datetime) -> float:
    whole_part, frac_part = jday(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )
    julian_day_2000 = whole_part + frac_part - 2451545.0
    return julian_day_2000
