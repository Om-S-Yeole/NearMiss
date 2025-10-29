"""
Unit tests for the `_filters` module.

This module tests the filtering function for satellite pairs.
"""

from nearmiss.utils._astro._filters import apoapsis_periapsis_filter


def test_apoapsis_periapsis_filter():
    """
    Test the `apoapsis_periapsis_filter` function for filtering satellite pairs.
    """
    r_1_p = 7000
    r_2_p = 7100
    r_1_a = 8000
    r_2_a = 8100
    Dist = 50
    assert apoapsis_periapsis_filter(r_1_p, r_2_p, r_1_a, r_2_a, Dist) is False

    Dist = 500
    assert apoapsis_periapsis_filter(r_1_p, r_2_p, r_1_a, r_2_a, Dist) is True
