"""
Unit tests for the `_helpers` module.

This module tests the helper functions for satellite attribute extraction and comparison.
"""

import pytest
from datetime import datetime
from sgp4.api import Satrec
from nearmiss.utils._astro._helpers import (
    satellite_attributes_from_Satrec_obj,
    sats_are_physically_identical,
)
from nearmiss.utils._astro._dataclasses import SingleSatInputAttributes


def test_satellite_attributes_from_Satrec_obj():
    """
    Test the `satellite_attributes_from_Satrec_obj` function for extracting satellite attributes.
    """
    sat = Satrec.twoline2rv(
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    D_start = datetime(2021, 10, 2, 14, 0, 0)
    tle_epoch = datetime(2021, 10, 2, 14, 10, 0)
    result = satellite_attributes_from_Satrec_obj(sat, D_start, tle_epoch)
    assert isinstance(result, SingleSatInputAttributes)
    assert result.satnum == 25544


def test_sats_are_physically_identical():
    """
    Test the `sats_are_physically_identical` function for comparing two satellites.
    """
    sat1 = Satrec.twoline2rv(
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    sat2 = Satrec.twoline2rv(
        "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993",
        "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556",
    )
    assert sats_are_physically_identical(sat1, sat2) is True
