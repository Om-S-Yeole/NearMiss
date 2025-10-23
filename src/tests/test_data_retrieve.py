import os
import pytest
from unittest.mock import patch, mock_open
from nearmiss.data.data_retrieve import reterive_data_from_api


@patch("nearmiss.data.data_retrieve._read_yaml")
@patch("nearmiss.data.data_retrieve.requests.get")
@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_reterive_data_from_api(
    mock_open, mock_makedirs, mock_requests_get, mock_read_yaml
):
    # Mock configuration
    mock_read_yaml.return_value = {
        "CELESTRACK": {
            "base_url": "https://celestrak.org/NORAD/elements/gp.php",
            "query": "GROUP",
            "format": "TLE",
            "groups": ["group1", "group2"],
        }
    }

    # Mock API responses
    mock_requests_get.return_value.ok = True
    mock_requests_get.return_value.text = "TLE data"

    # Call the function
    reterive_data_from_api()

    # Assertions
    mock_read_yaml.assert_called_once()
    assert mock_requests_get.call_count == 2
    mock_makedirs.assert_called()
    mock_open.assert_called()
    assert mock_open().write.call_count == 2


def test_reterive_data_from_api_no_groups():
    with patch(
        "nearmiss.data.data_retrieve._read_yaml",
        return_value={"CELESTRACK": {"groups": []}},
    ):
        with pytest.raises(
            Exception, match="Can not find groups to request in config file."
        ):
            reterive_data_from_api()
