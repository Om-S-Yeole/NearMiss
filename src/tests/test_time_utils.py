"""
Unit tests for the `_time` module.

This module tests the conversion functions between Julian Date and datetime objects.
"""

import pytest
from datetime import datetime, timezone
from nearmiss.utils._astro._time import (
    jd_to_datetime,
    datetime_to_jd,
    datetime_to_jd_2000,
)


def test_jd_to_datetime():
    """
    Test the `jd_to_datetime` function for converting Julian Date to datetime.
    """
    jd = 2451545.0  # J2000 epoch
    expected = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = jd_to_datetime(jd)
    assert result == expected


def test_datetime_to_jd():
    """
    Test the `datetime_to_jd` function for converting datetime to Julian Date.
    """
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd, whole, frac = datetime_to_jd(dt)
    assert jd == pytest.approx(2451545.0, rel=1e-6)
    assert whole == 2451545.0
    assert frac == 0.0


def test_datetime_to_jd_2000():
    """
    Test the `datetime_to_jd_2000` function for converting datetime to Julian Date relative to J2000.
    """
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd_2000 = datetime_to_jd_2000(dt)
    assert jd_2000 == pytest.approx(0.0, rel=1e-6)
