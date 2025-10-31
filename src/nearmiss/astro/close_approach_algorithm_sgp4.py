"""
Module for determining the closest approach and collision probability between satellites using SGP4.

This module provides a function to calculate the closest approach and collision probability between two satellites based on their TLE data and a specified time window.
"""

import numpy as np
from datetime import datetime, timedelta
from math import log, sqrt
from typing import Tuple

from scipy.optimize import minimize_scalar
from sgp4.api import Satrec
from sgp4.conveniences import sat_epoch_datetime

from nearmiss.utils import (
    MLOutputAttributes,
    SatPairAttributes,
    SingleSatInputAttributes,
    apoapsis_periapsis_filter,
    datetime_to_jd_2000,
    distance_squared,
    max_prob_function,
    satellite_attributes_from_Satrec_obj,
)


def close_approach_physical_algorithm_sgp4(
    tle1: Tuple[str, str],
    tle2: Tuple[str, str],
    D_start: datetime,
    D_stop: datetime,
    random_sat_radii: bool = True,
    r_obj_1: float | None = None,
    r_obj_2: float | None = None,
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
    random_sat_radii: bool
        Whether to take radii of both satellites randomly from range 1 m to 10 m. Default is True. Ignored if both r_obj_1 and r_obj_2 are provided.
    r_obj_1 : float, None, optional
        Radius of the first object in meters. Default is None. If both r_obj_1 and r_obj_2 are not given then random_sat_radii must be set to `True`, otherwise raises `ValueError`.
    r_obj_2 : float, None, optional
        Radius of the second object in meters. Default is None. If both r_obj_1 and r_obj_2 are not given then random_sat_radii must be set to `True`, otherwise raises `ValueError`.
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
    if not isinstance(random_sat_radii, bool):
        raise TypeError("random_sat_radii must be of type bool.")
    if not (isinstance(r_obj_1, float) or r_obj_1 is None):
        raise TypeError(
            f"Expected type of r_obj_1 is float or None. Got {type(r_obj_1)}"
        )
    if not (isinstance(r_obj_2, float) or r_obj_2 is None):
        raise TypeError(
            f"Expected type of r_obj_2 is float or None. Got {type(r_obj_2)}"
        )
    if not isinstance(Dist, (float, int)):
        raise TypeError("Dist must be an instance of float.")

    # --- Value Validation ---
    if D_start >= D_stop:
        raise ValueError(
            f"D_start cannot be >= D_stop. Got D_start={D_start}, D_stop={D_stop}"
        )
    if (D_stop - D_start) > timedelta(days=7):
        raise ValueError("Time window must not exceed 7 days.")
    if (
        (r_obj_1 is not None)
        and (r_obj_2 is not None)
        and (r_obj_1 <= 0 or r_obj_2 <= 0)
    ):
        raise ValueError("Object radii must be positive non-zero values.")
    if Dist < 0:
        raise ValueError("Dist value must be positive.")

    # --- Initialize satellites ---
    sat1 = Satrec.twoline2rv(*tle1)
    sat2 = Satrec.twoline2rv(*tle2)

    if r_obj_1 and r_obj_2:
        pass
    elif random_sat_radii:
        r_obj_1, r_obj_2 = np.random.rand(2) * 9 + 1
    else:
        raise ValueError(
            "At least provide values of both r_obj_1 and r_obj_2 or set random_sat_radii to True."
        )

    sat_1_inp_attri: SingleSatInputAttributes = satellite_attributes_from_Satrec_obj(
        sat1, D_start, sat_epoch_datetime(sat1), r_obj_1
    )
    sat_2_inp_attri: SingleSatInputAttributes = satellite_attributes_from_Satrec_obj(
        sat2, D_start, sat_epoch_datetime(sat2), r_obj_2
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
