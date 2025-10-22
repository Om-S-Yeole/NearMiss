import os
import time
import requests
from nearmiss.utils import _read_yaml


def reterive_data_from_api():
    """
    Retrieve data from the CelesTrak API and save it to a file.

    This function reads configuration details from a YAML file, sends requests to the CelesTrak API for specified groups, and retrieves TLE data. The data is saved to a timestamped file in the raw data directory, and the name of the latest file is updated in a separate file.

    Raises
    ------
    Exception
        If no groups are found in the configuration file.

    Notes
    -----
    - The configuration file path is hardcoded as "nearmiss/data/configs/celestrak.yaml".
    - The function assumes the presence of a utility function `_read_yaml` to read the YAML file.
    - The retrieved data is stored in the "data/raw" directory relative to the project structure. This data directory is at the same level where the src directory is.
    """
    celestrak_config_path: str = "nearmiss/data/configs/celestrak.yaml"
    cfg: dict = _read_yaml(path=celestrak_config_path)
    celestrack_cfg: dict = cfg["CELESTRACK"]

    request_base_url: str = celestrack_cfg.get(
        "base_url", "https://celestrak.org/NORAD/elements/gp.php"
    )
    request_query_type: str = celestrack_cfg.get("query", "GROUP")
    request_format_type: str = celestrack_cfg.get("format", "TLE")
    groups: list = celestrack_cfg.get("groups", [])

    if groups:
        response_txt = ""
        total_groups = len(groups)
        for idx, group in enumerate(groups, start=1):
            request_url = f"{request_base_url}?{request_query_type}={group}&FORMAT={request_format_type}"
            res = requests.get(request_url)
            if res.ok:
                if res.text:
                    response_txt += f"{"\n" if response_txt else ""}{res.text}"
                else:
                    print(
                        f"Response for group '{group}' do not contain TLE data string in text."
                    )
            else:
                print(
                    f"Invalid response from the server. Status code {res.status_code}.For group '{group}'"
                )
            print(
                f"[{idx}/{total_groups}] Work for group '{group}' done successfully. Data retrieved: {True if res.text else False}"
            )

        t = int(time.time())
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "..", "..", "data"
        )
        raw_data_dir = f"{data_dir}/raw"
        os.makedirs(raw_data_dir, exist_ok=True)
        new_file = f"celestrak_data_{t}.txt"
        txt_file_name = os.path.join(raw_data_dir, new_file)
        latest_file = os.path.join(data_dir, "latest_raw_data_file.txt")

        with open(txt_file_name, "w") as file:
            file.write(response_txt)
        with open(latest_file, "w") as file:
            file.write(new_file)
        print("New data retrieved and stored successfully.")
    else:
        raise Exception("Can not find groups to request in config file.")
