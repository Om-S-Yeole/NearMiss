"""
Unit tests for the `_sgp4` module.

This module tests the SGP4-based propagation and distance calculation functions.
"""

import pytest
from datetime import datetime
from sgp4.api import Satrec
from nearmiss.utils._astro._sgp4 import propagate_sgp4, distance_squared, SGP4Exception


def test_propagate_sgp4():
    """
    Test the `propagate_sgp4` function for propagating satellite state vectors.
    """
    sat = Satrec.twoline2rv(
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    epoch = datetime(2021, 10, 2, 14, 0, 0)
    dt = 60.0
    r, v = propagate_sgp4(sat, dt, epoch)
    assert len(r) == 3
    assert len(v) == 3


def test_propagate_sgp4_exception():
    """
    Test the `propagate_sgp4` function for raising an exception on propagation failure.
    """
    sat = Satrec()
    epoch = datetime(2021, 10, 2, 14, 0, 0)
    dt = 60.0
    with pytest.raises(SGP4Exception):
        propagate_sgp4(sat, dt, epoch)


def test_distance_squared():
    """
    Test the `distance_squared` function for calculating squared distance between satellites.
    """
    sat1 = Satrec.twoline2rv(
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    sat2 = Satrec.twoline2rv(
        "1 43210U 98067B   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 43210  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    D_start = datetime(2021, 10, 2, 14, 0, 0)
    t_sec = 60.0
    dist2 = distance_squared(t_sec, sat1, sat2, D_start)
    assert dist2 > 0
