import numpy as np
from math import sqrt
from sgp4.api import Satrec
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from typing import Tuple
from nearmiss.utils import (
    jd_to_datetime,
    propagate_sgp4,
    distance_squared,
    max_prob_function,
    apoapsis_periapsis_filter,
    bounding_box_filter,
    relative_motion_filter,
    inclination_raan_filter,
)


def close_approach_physical_algorithm_sgp4(
    tle1: Tuple[str, str],
    tle2: Tuple[str, str],
    D_start: datetime,
    D_stop: datetime,
    r_obj_1: float = 1.5,
    r_obj_2: float = 1.5,
    Dist: float = 10.0,
    inc_tol: float = 10.0,
    raan_tol: float = 15.0,
    threshold_km: float = 8.0,
    safety_margin: float = 0.0,
) -> Tuple[datetime | None, float | None, float]:
    """
    Determine the closest approach and collision probability between two satellites using SGP4.

    Parameters
    ----------
    tle1 : tuple[str, str]
        Two-line element set for the first satellite.
    tle2 : tuple[str, str]
        Two-line element set for the second satellite.
    D_start : datetime
        Start time of the analysis window.
    D_stop : datetime
        End time of the analysis window.
    r_obj_1 : float, optional
        Radius of the first object. Default is 1.5.
    r_obj_2 : float, optional
        Radius of the second object. Default is 1.5.
    Dist : float, optional
        Minimum distance threshold for collision. Default is 10.0 km.
    inc_tol: float, optional
        Inclination tolarence in deg for Inclination RAAN Filter. Default is 10 deg.
    raan_tol: float, optional
        Right Ascension of Ascending Node tolarence in deg for Inclination RAAN Filter. Default is 15 deg.
    threshold_km: float, optional
        Threshold distance in km for Relative Motion Filter. Default is 8.0 km.
    safety_margin: float, optional
        Safety margin distance in km for Relative Motion Filter. Default is 0.0 km.

    Returns
    -------
    tuple[datetime|None, float|None, float]
        Time of closest approach, closest distance, and maximum collision probability.

    Raises
    ------
    TypeError
        If input parameters are of incorrect types.
    ValueError
        If input parameters are invalid or inconsistent.
    RuntimeError
        If the minimization process fails.
    """
    # --- Type Validation ---
    if not isinstance(tle1, tuple) or not isinstance(tle2, tuple):
        raise TypeError("tle1 and tle2 must be instances of tuple.")
    if not isinstance(D_start, datetime) or not isinstance(D_stop, datetime):
        raise TypeError("D_start and D_stop must be datetime objects.")
    if not isinstance(r_obj_1, (float, int)) or not isinstance(r_obj_2, (float, int)):
        raise TypeError("r_obj_1 and r_obj_2 must be instances of float.")
    if not isinstance(Dist, (float, int)):
        raise TypeError("Dist must be an instance of float.")

    # --- Value Validation ---
    if D_start >= D_stop:
        raise ValueError(
            f"D_start cannot be >= D_stop. Got D_start={D_start}, D_stop={D_stop}"
        )
    if (D_stop - D_start) > timedelta(days=7):
        raise ValueError("Time window must not exceed 7 days.")
    if r_obj_1 <= 0 or r_obj_2 <= 0:
        raise ValueError("Object radii must be positive non-zero values.")
    if Dist < 0:
        raise ValueError("Dist value must be positive.")

    # --- Initialize satellites ---
    sat1 = Satrec.twoline2rv(*tle1)
    sat2 = Satrec.twoline2rv(*tle2)

    # --- Get epochs and sync to common ---
    epoch1 = jd_to_datetime(sat1.jdsatepoch + sat1.jdsatepochF)
    epoch2 = jd_to_datetime(sat2.jdsatepoch + sat2.jdsatepochF)
    common_epoch = max(epoch1, epoch2)

    # --- Propagate both to common epoch ---
    r_1, v_1 = propagate_sgp4(sat1, (common_epoch - epoch1).total_seconds(), epoch1)
    r_2, v_2 = propagate_sgp4(sat2, (common_epoch - epoch2).total_seconds(), epoch2)

    # ---------------
    # --- Filters ---
    # ---------------

    # --- Inclination-RAAN Filter ---
    sat_1_inc = np.rad2deg(sat1.inclo)
    sat_1_raan = np.rad2deg(sat1.nodeo)
    sat_2_inc = np.rad2deg(sat2.inclo)
    sat_2_raan = np.rad2deg(sat2.nodeo)
    if not inclination_raan_filter(
        sat_1_inc, sat_2_inc, sat_1_raan, sat_2_raan, inc_tol, raan_tol
    ):
        print(
            "Inclination-RAAN filter activated for this pair. No possibility of collision."
        )
        return (None, None, 0)

    # --- Apoapsis-Periapsis Filter ---
    r_p1 = (sat1.a * (1 - sat1.ecco)) * 6378.135 / sat1.radiusearthkm  # km
    r_a1 = (sat1.a * (1 + sat1.ecco)) * 6378.135 / sat1.radiusearthkm  # km
    r_p2 = (sat2.a * (1 - sat2.ecco)) * 6378.135 / sat2.radiusearthkm  # km
    r_a2 = (sat2.a * (1 + sat2.ecco)) * 6378.135 / sat2.radiusearthkm  # km
    if apoapsis_periapsis_filter(r_p1, r_p2, r_a1, r_a2, Dist):
        print(
            f"Apoapsis-Periapsis Filter activated for this pair. No possibility of collision. (Delta={max(r_p1, r_p2)-min(r_a1, r_a2):.2f} km)"
        )
        return (None, None, 0.0)

    # --- Bounding-Box Filter ---
    if not bounding_box_filter(sat1, sat2, common_epoch, D_start, D_stop):
        print(
            f"Bounding Box Filter activated for this pair. No possibility of collision for given time window."
        )
        return (None, None, 0.0)

    # --- Relative Motion Filter ---
    if not relative_motion_filter(
        r_1,
        v_1,
        r_2,
        v_2,
        window_seconds=(D_stop - D_start).total_seconds(),
        threshold_km=threshold_km,
        safety_margin_km=safety_margin,
    ):
        print(
            f"Relative motion filter activated for this pair. No possibility of collision for given time window."
        )
        return (None, None, 0.0)

    # --- Minimize distance squared ---
    start_offset = (D_start - common_epoch).total_seconds()
    bound_sec = (D_stop - D_start).total_seconds()

    res = minimize_scalar(
        distance_squared,
        bounds=[start_offset, start_offset + bound_sec],
        method="bounded",
        args=(sat1, sat2, common_epoch),
    )

    if not res.success:
        raise RuntimeError("Failed to find minimum distance.")

    t_min = common_epoch + timedelta(seconds=res.x)
    d_min = sqrt(distance_squared(res.x, sat1, sat2, common_epoch))
    P_max = max_prob_function(r_obj_1 / 1000, r_obj_2 / 1000, d_min)

    return (t_min, d_min, P_max)
