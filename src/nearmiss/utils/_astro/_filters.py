"""
Module for filtering satellite pairs based on orbital parameters.

This module provides a function to filter satellite pairs using their apoapsis and periapsis distances.
"""


def apoapsis_periapsis_filter(
    r_1_p: float, r_2_p: float, r_1_a: float, r_2_a: float, Dist: float
) -> bool:
    """
    Filter satellite pairs based on their apoapsis and periapsis distances.

    Parameters
    ----------
    r_1_p : float
        Periapsis distance of the first satellite in kilometers.
    r_2_p : float
        Periapsis distance of the second satellite in kilometers.
    r_1_a : float
        Apoapsis distance of the first satellite in kilometers.
    r_2_a : float
        Apoapsis distance of the second satellite in kilometers.
    Dist : float
        Minimum distance threshold in kilometers.

    Returns
    -------
    bool
        True if the maximum periapsis distance is greater than the minimum apoapsis distance by more than `Dist`, indicating no collision risk. False otherwise.
    """
    if (max(r_1_p, r_2_p) - min(r_1_a, r_2_a)) > Dist:
        return True
    return False
