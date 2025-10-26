import os
from datetime import datetime, timedelta
from sgp4.model import Satrec
from sgp4.api import days2mdhms
from sgp4.propagation import sgp4
from sgp4.conveniences import sat_epoch_datetime
from nearmiss.data import reterive_data_from_api


def tle_to_r_v(
    line_1: str, line_2: str, whichconst: int = 2
) -> tuple[datetime, list, list]:
    """
    Convert TLE data to position and velocity vectors.

    Parameters
    ----------
    line_1 : str
        The first line of the TLE data.
    line_2 : str
        The second line of the TLE data.
    whichconst : int, optional
        The gravitational constant set to use (0, 1, or 2). Default is 2.

    Returns
    -------
    tuple[datetime, list, list]
        A tuple containing the datetime of calculation of r and v, position vector (r) and velocity vector (v).

    Raises
    ------
    TypeError
        If `line_1`, `line_2`, or `whichconst` are of incorrect types.
    ValueError
        If `whichconst` is not one of [0, 1, 2].

    Notes
    -----
    - The position and velocity vectors are calculated for the same day at 00:00:00 UTC.
    - Returned position vector has units km.
    - Returned velocity vector has units km/s.
    """

    if not (whichconst in (0, 1, 2)):
        raise ValueError(
            f"Expected value of whichconst is from [0, 1, 2]. Got {whichconst}."
        )

    sat = Satrec.twoline2rv(line_1, line_2, whichconst)
    _, _, hour, minute, second = days2mdhms(sat.epochyr, sat.epochdays)
    minutes_ahead = hour * 60 + minute + (second / 60)

    sat_current_datetime = sat_epoch_datetime(sat)
    r_v_processed_datetime = sat_current_datetime - timedelta(minutes=minutes_ahead)

    # Finding r and v on for the same day but at 00:00:00 UTC
    r, v = sgp4(sat, -minutes_ahead)

    return (r_v_processed_datetime, list(r), list(v))


def raw_tle_file_to_r_v_processer(
    from_latest_raw_data_file: bool = True, raw_file_name: str | None = None
):
    """
    Process a raw TLE file and convert TLE data to position and velocity vectors.

    Parameters
    ----------
    from_latest_raw_data_file : bool, optional
        Whether to process the latest raw data file. Default is True.
    raw_file_name : str or None, optional
        The name of the raw file to process. If None, the latest file is used.

    Raises
    ------
    TypeError
        If `from_latest_raw_data_file` or `raw_file_name` are of incorrect types.
    ValueError
        If `from_latest_raw_data_file` is False and `raw_file_name` is not provided.
    FileNotFoundError
        If the specified file does not exist.
    NotImplementedError
        If the latest raw data file exists but is empty.

    Notes
    -----
    - The processed data is saved in a CSV file in the "data/processed" directory.
    - The name of the latest processed file is updated in "latest_processed_data_file.txt".
    """

    if (not from_latest_raw_data_file) and (not raw_file_name):
        raise ValueError(
            f"If the latest file data must not be processed (by setting from_latest_raw_data_file argument as False), then it is expected to provide the value for argument raw_file_name."
        )
    if from_latest_raw_data_file and raw_file_name:
        print(
            f"Warning: Function initiated by setting to process latest file as well as by providing raw_file_name. Using the raw_file_name file for further processing."
        )

    file_path = ""

    current_file_dir = os.path.dirname(__file__)
    data_dir_path = os.path.join(current_file_dir, "..", "..", "..", "data")
    raw_data_dir_path = os.path.join(data_dir_path, "raw")
    processed_file_name = None

    if from_latest_raw_data_file and (not raw_file_name):
        latest_data_file_path = os.path.join(data_dir_path, "latest_raw_data_file.txt")
        print(latest_data_file_path)
        if not os.path.isfile(latest_data_file_path):
            raise FileNotFoundError(
                "Error: latest_raw_data_file.txt file do not exist."
            )
        if os.path.getsize(latest_data_file_path) == 0:
            raise NotImplementedError(
                "Error: latest_raw_data_file.txt exist but is empty."
            )
        with open(latest_data_file_path, "r") as file:
            latest_file = file.read()
            file_path = os.path.join(raw_data_dir_path, latest_file)
            root, _ = os.path.splitext(latest_file)
            processed_file_name = f"{root}_processed.csv"
    else:
        file_path = os.path.join(raw_data_dir_path, raw_file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"Error: file named {raw_file_name} do not exist in data/raw directory."
            )
        root, _ = os.path.splitext(raw_file_name)
        processed_file_name = f"{root}_processed.csv"

    processed_dir_path = os.path.join(data_dir_path, "processed")
    os.makedirs(processed_dir_path, exist_ok=True)
    processed_file_path = os.path.join(processed_dir_path, processed_file_name)

    with (
        open(file_path, "r") as read_file,
        open(processed_file_path, "a") as write_file,
    ):
        while True:
            name = read_file.readline()
            if not name:
                break  # EOF
            name = name.strip()

            line_1 = read_file.readline().strip()
            line_2 = read_file.readline().strip()

            cat_no = line_1[2:7]
            try:
                dt, r, v = tle_to_r_v(line_1, line_2)
                write_file.write(
                    f"{dt},{name},{cat_no},{r[0]},{r[1]},{r[2]},{v[0]},{v[1]},{v[2]}\n"
                )
            except ValueError as e:
                print(
                    f"TLE can not be processed for {name} with catalog number {cat_no}"
                )

    with open(
        os.path.join(data_dir_path, "latest_processed_data_file.txt"), "w"
    ) as file:
        file.write(processed_file_name)

    print(f"TLEs in file processed successfully in file {processed_file_name}.")


def process_data_to_r_v(
    from_api: bool = False,
    from_latest_raw_data_file: bool = True,
    raw_file_name: str | None = None,
):
    """
    Process TLE data to position and velocity vectors, optionally retrieving data from the API.

    Parameters
    ----------
    from_api : bool, optional
        Whether to retrieve data from the API before processing. Default is False.
    from_latest_raw_data_file : bool, optional
        Whether to process the latest raw data file. Default is True.
    raw_file_name : str or None, optional
        The name of the raw file to process. If None, the latest file is used.

    Raises
    ------
    TypeError
        If `from_api`, `from_latest_raw_data_file`, or `raw_file_name` are of incorrect types.
    ValueError
        If `from_latest_raw_data_file` is False and `raw_file_name` is not provided.

    Notes
    -----
    - If `from_api` is True, it overrides all other parameters and retrieves data from the API.
    - The processed data is saved in a CSV file in the "data/processed" directory.
    """

    if (not from_latest_raw_data_file) and (not raw_file_name):
        raise ValueError(
            f"If the latest file data must not be processed (by setting from_latest_raw_data_file argument as False), then it is expected to provide the value for argument raw_file_name."
        )
    if from_latest_raw_data_file and raw_file_name:
        print(
            f"Warning: Function initiated by setting to process latest file as well as by providing raw_file_name. Using the raw_file_name file for further processing."
        )
        from_latest_raw_data_file = False

    if from_api:
        from_latest_raw_data_file = False
        raw_file_name = None
        reterive_data_from_api()
        raw_tle_file_to_r_v_processer(True)
    elif from_latest_raw_data_file:
        from_api = False
        raw_file_name = None
        raw_tle_file_to_r_v_processer(True)
    elif raw_file_name:
        from_api = False
        from_latest_raw_data_file = False
        raw_tle_file_to_r_v_processer(False, raw_file_name)
