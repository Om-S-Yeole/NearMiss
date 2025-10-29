"""
Module for processing raw TLE data to generate training datasets.

This module provides functionality to process raw TLE data, identify satellite pairs, and calculate their closest approach and collision probabilities.
"""

import os
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from scipy.spatial import KDTree
from sgp4.api import Satrec
from nearmiss.astro import close_approach_physical_algorithm_sgp4
from nearmiss.utils import (
    datetime_to_jd,
    sats_are_physically_identical,
    SatPairAttributes,
    SGP4Exception,
)


def training_data_maker_from_physical_algorithm(
    D_start: datetime,
    D_stop: datetime,
    t_interval: timedelta = timedelta(seconds=900),
    r_threshold_KDtree: float = 12,
    from_latest_raw_data_file: bool = True,
    raw_file_name: str | None = None,
    optional_args: dict | None = None,
):
    """
    Generate training data using a physical algorithm for satellite collision analysis.

    Parameters
    ----------
    D_start : datetime
        Start time of the analysis window.
    D_stop : datetime
        End time of the analysis window.
    t_interval : timedelta, optional
        Time interval for discretizing orbit propagation. Default is 900 seconds.
    r_threshold_KDtree : float, optional
        Distance threshold in kilometers for identifying potential satellite pairs using KDTree. Default is 12 km.
    from_latest_raw_data_file : bool, optional
        Whether to process the latest raw data file. Default is True.
    raw_file_name : str or None, optional
        Name of the raw TLE file to process. If None, the latest file is used. Default is None.
    optional_args : dict or None, optional
        Additional arguments for the physical algorithm. Default is None.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If input parameters are of incorrect types.
    ValueError
        If `from_latest_raw_data_file` is False and `raw_file_name` is not provided.
    FileNotFoundError
        If the specified raw TLE file does not exist.
    NotImplementedError
        If the latest raw data file exists but is empty.

    Notes
    -----
    - The function reads TLE data, identifies potential satellite pairs using KDTree, and calculates their closest approach and collision probabilities.
    - The processed data is saved in a CSV file in the "data/processed" directory.
    - The name of the latest processed file is updated in "latest_processed_data_file.txt".
    """

    if not isinstance(D_start, datetime):
        raise TypeError(f"Expected type of D_start is datetime. Got {type(D_start)}.")
    if not isinstance(D_stop, datetime):
        raise TypeError(f"Expected type of D_stop is datetime. Got {type(D_stop)}.")
    if not isinstance(from_latest_raw_data_file, bool):
        raise TypeError(
            f"Expected type of from_latest_raw_data_file is bool. Got {type(from_latest_raw_data_file)}"
        )
    if not (isinstance(raw_file_name, str) or raw_file_name is None):
        raise TypeError(
            f"Expected type of raw_file_name is str or None. Got {type(raw_file_name)}"
        )
    if not (isinstance(optional_args, dict) or optional_args is None):
        raise TypeError(
            f"Expected type of optional_args is dict or None. Got {type(optional_args)}"
        )

    if not (D_start.tzinfo == timezone.utc):
        D_start = D_start.replace(tzinfo=timezone.utc)
    if not (D_stop.tzinfo == timezone.utc):
        D_stop = D_stop.replace(tzinfo=timezone.utc)

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

    # Read all TLEs into memory first
    with open(file_path, "r") as read_file:
        print(f"Starting the task. Reading all TLEs. Please wait...")
        lines = [line.strip() for line in read_file if line.strip()]
        print(f"Reading TLE file finished. Proceeding to next step.")

    # Group lines into (name, line1, line2)
    sats = [(lines[i], lines[i + 1], lines[i + 2]) for i in range(0, len(lines), 3)]

    sat_obj_list = []
    for sat in sats:
        sat_obj = Satrec.twoline2rv(sat[1], sat[2])
        sat_obj_list.append(sat_obj)

    current_time = D_start
    pairs_to_check = set()

    while current_time < D_stop:
        sat_pos_at_curr_time = np.empty((len(sat_obj_list), 3), dtype=np.float64)

        _, curr_time_wp, curr_time_fp = datetime_to_jd(current_time)
        for i, sat_obj in enumerate(sat_obj_list):
            error_code, r, _ = sat_obj.sgp4(curr_time_wp, curr_time_fp)
            if error_code == 0:
                sat_pos_at_curr_time[i] = r
            else:
                sat_pos_at_curr_time[i] = np.nan  # handle propagation failure

        # Filter valid positions (non-NaN)
        valid_mask = np.isfinite(sat_pos_at_curr_time).all(axis=1)
        valid_positions = sat_pos_at_curr_time[valid_mask]
        valid_indices = np.nonzero(valid_mask)[0]

        # Skip if no valid data
        if len(valid_positions) == 0:
            current_time += t_interval
            continue

        # Build KDTree only on valid data
        tree = KDTree(valid_positions)
        pairs = tree.query_pairs(r=r_threshold_KDtree, output_type="ndarray")

        # Map back to original satellite indices
        if pairs.size > 0:
            for pair in pairs:
                if pair[0] == pair[1]:
                    print(f"Repeted {pair[0]} and {pair[1]}")
                    continue
                global_pair = (valid_indices[pair[0]], valid_indices[pair[1]])
                pairs_to_check.add(global_pair)

        current_time = current_time + t_interval

    pairs_to_check = np.array(list(pairs_to_check))

    with open(processed_file_path, "w") as write_file:
        print(f"Opened the writing file. Starting to write. Please wait...")
        print(f"Satellite pairs algorithm has to check: {len(pairs_to_check)}")
        one_fourth_pairs = int(len(pairs_to_check) / 4)
        half_pairs = 2 * one_fourth_pairs
        three_fourth_pairs = 3 * one_fourth_pairs

        for idx, pair_to_check in enumerate(pairs_to_check):
            if idx == one_fourth_pairs:
                print(f"[1/4] Processing done, processing further...")
            if idx == half_pairs:
                print(f"[2/4] Processing done, processing further...")
            if idx == three_fourth_pairs:
                print(f"[3/4] Processing done, processing further...")

            sat_idx_1 = pair_to_check[0]
            sat_idx_2 = pair_to_check[1]

            if sat_idx_1 == sat_idx_2:
                print("Skipping identical index pair:", sat_idx_1)
                continue

            sat_1_line_1 = sats[sat_idx_1][1]
            sat_1_line_2 = sats[sat_idx_1][2]

            sat_2_line_1 = sats[sat_idx_2][1]
            sat_2_line_2 = sats[sat_idx_2][2]

            sat1 = Satrec.twoline2rv(sat_1_line_1, sat_1_line_2)
            sat2 = Satrec.twoline2rv(sat_2_line_1, sat_2_line_2)

            if sat_1_line_2[2:7] == sat_2_line_2[2:7]:
                # print(f"Same satellites. Catalog number {sat_1_line_2[2:7]}")
                continue

            if sats_are_physically_identical(sat1, sat2):
                # print(f"Two TLEs form identical orbits (likely docked). Catalogs {sat_1_line_2[2:7]} and {sat_2_line_2[2:7]}")
                continue

            args = {
                "tle1": (sat_1_line_1, sat_1_line_2),
                "tle2": (sat_2_line_1, sat_2_line_2),
                "D_start": D_start,
                "D_stop": D_stop,
            }
            if optional_args:
                args.update(optional_args)

            try:
                sat_pair_attri: SatPairAttributes = (
                    close_approach_physical_algorithm_sgp4(**args)
                )
            except SGP4Exception as sgpexp:
                print(f"[SGP4 Fail] {sat_idx_1}-{sat_idx_2}: {sgpexp}")
                continue
            except Exception as e:
                print(
                    f"[ERROR] Unexpected failure for pair {sat_idx_1}-{sat_idx_2}: {type(e).__name__} -> {e}"
                )
                raise

            sat_1_inp = asdict(sat_pair_attri.sat_1_inp)
            sat_2_inp = asdict(sat_pair_attri.sat_2_inp)
            output = list(asdict(sat_pair_attri.output).values())

            # Convert all values to strings and write in one go
            row = [
                *map(str, sat_1_inp.values()),
                *map(str, sat_2_inp.values()),
                *map(str, output),
            ]
            write_file.write(",".join(row) + "\n")

    with open(
        os.path.join(data_dir_path, "latest_processed_data_file.txt"), "w"
    ) as file:
        file.write(processed_file_name)

    print(f"TLEs in file processed successfully in file {processed_file_name}.")
