"""
Unit tests for the `close_approach_algorithm_sgp4` module.

This module tests the function for determining the closest approach and collision probability between satellites using SGP4.
"""

import pytest
from datetime import datetime, timedelta
from sgp4.api import Satrec
from nearmiss.astro.close_approach_algorithm_sgp4 import (
    close_approach_physical_algorithm_sgp4,
)
from nearmiss.utils._astro._dataclasses import SatPairAttributes


def test_close_approach_physical_algorithm_sgp4():
    """
    Test the `close_approach_physical_algorithm_sgp4` function for calculating closest approach and collision probability.
    """
    tle1 = (
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    tle2 = (
        "1 43210U 98067B   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 43210  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    D_start = datetime(2021, 10, 2, 14, 0, 0)
    D_stop = D_start + timedelta(days=1)

    result = close_approach_physical_algorithm_sgp4(
        tle1, tle2, D_start, D_stop, r_obj_1=5, r_obj_2=5, Dist=10.0
    )

    assert isinstance(result, SatPairAttributes)
    assert result.output.filter_rej_code == 0
    assert result.output.t_close > 0
    assert result.output.ln_d_min > 0
    assert 0 <= result.output.probab <= 1


def test_close_approach_physical_algorithm_sgp4_filter_rejection():
    """
    Test the `close_approach_physical_algorithm_sgp4` function for filter rejection.
    """
    tle1 = (
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    tle2 = (
        "1 43210U 98067B   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 43210  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    D_start = datetime(2021, 10, 2, 14, 0, 0)
    D_stop = D_start + timedelta(days=1)

    # Use a large distance threshold to trigger filter rejection
    result = close_approach_physical_algorithm_sgp4(
        tle1, tle2, D_start, D_stop, r_obj_1=5, r_obj_2=5, Dist=1000.0
    )

    assert isinstance(result, SatPairAttributes)
    assert result.output.filter_rej_code == 1
    assert result.output.t_close == 0
    assert result.output.ln_d_min == 0
    assert result.output.probab == 0


def test_close_approach_physical_algorithm_sgp4_invalid_inputs():
    """
    Test the `close_approach_physical_algorithm_sgp4` function for invalid inputs.
    """
    tle1 = (
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    tle2 = (
        "1 43210U 98067B   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 43210  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    D_start = datetime(2021, 10, 2, 14, 0, 0)
    D_stop = D_start + timedelta(days=8)  # Exceeding the 7-day limit

    with pytest.raises(ValueError, match="Time window must not exceed 7 days."):
        close_approach_physical_algorithm_sgp4(
            tle1, tle2, D_start, D_stop, r_obj_1=5, r_obj_2=5, Dist=10.0
        )
