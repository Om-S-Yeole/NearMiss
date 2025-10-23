import os
import pytest
from unittest.mock import patch, mock_open
from datetime import datetime
from nearmiss.astro.tle_parser import (
    tle_to_r_v,
    raw_tle_file_to_r_v_processer,
    process_data_to_r_v,
)


def test_tle_to_r_v():
    line_1 = "1 25544U 98067A   21275.59097222  .00002182  00000-0  50300-4 0  9993"
    line_2 = "2 25544  51.6442  21.5553 0005545  45.1234 315.6789 15.48815347249556"
    dt, r, v = tle_to_r_v(line_1, line_2)

    assert isinstance(dt, datetime)
    assert len(r) == 3
    assert len(v) == 3


@patch("os.path.isfile", return_value=True)
@patch("os.path.getsize", return_value=100)
@patch("builtins.open", new_callable=mock_open, read_data="test_file.txt")
def test_raw_tle_file_to_r_v_processer(mock_open, mock_getsize, mock_isfile):
    with patch(
        "nearmiss.astro.tle_parser.tle_to_r_v",
        return_value=(datetime.now(), [1, 2, 3], [4, 5, 6]),
    ):
        raw_tle_file_to_r_v_processer(from_latest_raw_data_file=True)

    mock_isfile.assert_called()
    mock_getsize.assert_called()
    mock_open.assert_called()


@patch("nearmiss.astro.tle_parser.reterive_data_from_api")
@patch("nearmiss.astro.tle_parser.raw_tle_file_to_r_v_processer")
def test_process_data_to_r_v(mock_raw_tle, mock_retrieve_data):
    process_data_to_r_v(from_api=True)
    mock_retrieve_data.assert_called_once()
    mock_raw_tle.assert_called_once()
