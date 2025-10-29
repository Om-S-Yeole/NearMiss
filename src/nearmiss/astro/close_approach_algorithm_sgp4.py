"""
Module for determining the closest approach and collision probability between satellites using SGP4.

This module provides a function to calculate the closest approach and collision probability between two satellites based on their TLE data and a specified time window.
"""

from math import sqrt, log
from sgp4.api import Satrec
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from typing import Tuple
from sgp4.conveniences import sat_epoch_datetime
from nearmiss.utils import (
    datetime_to_jd_2000,
    distance_squared,
    max_prob_function,
    apoapsis_periapsis_filter,
    satellite_attributes_from_Satrec_obj,
    SingleSatInputAttributes,
    MLOutputAttributes,
    SatPairAttributes,
)


def close_approach_physical_algorithm_sgp4(
    tle1: Tuple[str, str],
    tle2: Tuple[str, str],
    D_start: datetime,
    D_stop: datetime,
    r_obj_1: float = 5,
    r_obj_2: float = 5,
    Dist: float = 10.0,
) -> SatPairAttributes:
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
        Radius of the first object in meters. Default is 5 m.
    r_obj_2 : float, optional
        Radius of the second object in meters. Default is 5 m.
    Dist : float, optional
        Minimum distance threshold for collision in kilometers. Default is 10.0 km.

    Returns
    -------
    SatPairAttributes
        A dataclass containing:
        - Input attributes for both satellites (`SingleSatInputAttributes`).
        - Output attributes (`MLOutputAttributes`) including:
            - Filter rejection code (int).
            - Time of closest approach in the J2000 system (float, scaled by 1e5).
            - Natural logarithm of (1 + closest distance) (float).
            - Collision probability at the closest approach (float).

    Raises
    ------
    TypeError
        If input parameters are of incorrect types.
    ValueError
        If input parameters are invalid or inconsistent.
    RuntimeError
        If the minimization process fails.

    Notes
    -----
    - The function applies multiple filters to reduce computational overhead.
    - The SGP4 propagation is used to calculate the positions and velocities of the satellites.
    - The collision probability is calculated based on the closest approach distance.
    - If a filter rejects the pair, the function returns early with the corresponding filter rejection code.
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
    sat_1_inp_attri: SingleSatInputAttributes = satellite_attributes_from_Satrec_obj(
        sat1, D_start, sat_epoch_datetime(sat1)
    )
    sat_2_inp_attri: SingleSatInputAttributes = satellite_attributes_from_Satrec_obj(
        sat2, D_start, sat_epoch_datetime(sat2)
    )

    # --- Filters ---

    filter_rejection_code = 0

    # --- 1: Apoapsis-Periapsis Filter ---
    r_p1 = (sat1.a * (1 - sat1.ecco)) * 6378.135 / sat1.radiusearthkm  # km
    r_a1 = (sat1.a * (1 + sat1.ecco)) * 6378.135 / sat1.radiusearthkm  # km
    r_p2 = (sat2.a * (1 - sat2.ecco)) * 6378.135 / sat2.radiusearthkm  # km
    r_a2 = (sat2.a * (1 + sat2.ecco)) * 6378.135 / sat2.radiusearthkm  # km
    if apoapsis_periapsis_filter(r_p1, r_p2, r_a1, r_a2, Dist):
        filter_rejection_code = 1
        outputs: MLOutputAttributes = MLOutputAttributes(filter_rejection_code, 0, 0, 0)

        sat_pair_attri_and_outputs: SatPairAttributes = SatPairAttributes(
            sat_1_inp_attri, sat_2_inp_attri, outputs
        )

        return sat_pair_attri_and_outputs

    # --- Main Algorithm ---

    bound_sec = (D_stop - D_start).total_seconds()

    res = minimize_scalar(
        distance_squared,
        bounds=[0, bound_sec],
        method="bounded",
        args=(sat1, sat2, D_start),
    )

    if not res.success:
        raise RuntimeError("Failed to find minimum distance.")

    t_min = D_start + timedelta(seconds=res.x)
    d_min = sqrt(res.fun)
    P_max = max_prob_function(r_obj_1 / 1000, r_obj_2 / 1000, d_min)

    outputs: MLOutputAttributes = MLOutputAttributes(
        filter_rejection_code, datetime_to_jd_2000(t_min) / 1e5, log(1 + d_min), P_max
    )

    sat_pair_attri_and_outputs: SatPairAttributes = SatPairAttributes(
        sat_1_inp_attri, sat_2_inp_attri, outputs
    )

    return sat_pair_attri_and_outputs
