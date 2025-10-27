import os
from datetime import datetime, timezone
from dataclasses import asdict
from nearmiss.astro import close_approach_physical_algorithm_sgp4
from nearmiss.utils import (
    SatPairAttributes,
)


def training_data_maker_from_physical_algorithm(
    D_start: datetime,
    D_stop: datetime,
    from_latest_raw_data_file: bool = True,
    raw_file_name: str | None = None,
    optional_args: dict | None = None,
):

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

    # Read all TLEs into memory first
    with open(file_path, "r") as read_file:
        print(f"Starting the task. Please wait...")
        lines = [line.strip() for line in read_file if line.strip()]

    # Group lines into (name, line1, line2)
    sats = [(lines[i], lines[i + 1], lines[i + 2]) for i in range(0, len(lines), 3)]

    with open(processed_file_path, "w") as write_file:
        print(f"Working on the task. Please wait...")
        done = 0
        for i, (_, sat_1_line_1, sat_1_line_2) in enumerate(sats):
            tle_1 = (sat_1_line_1, sat_1_line_2)
            for _, sat_2_line_1, sat_2_line_2 in sats[i + 1 :]:
                if done % 100000 == 0:
                    print(f"[{int(done/100000)}] Working...")
                tle_2 = (sat_2_line_1, sat_2_line_2)

                args = {
                    "tle1": tle_1,
                    "tle2": tle_2,
                    "D_start": D_start,
                    "D_stop": D_stop,
                }
                if optional_args:
                    args.update(optional_args)

                sat_pair_attri: SatPairAttributes = (
                    close_approach_physical_algorithm_sgp4(**args)
                )

                sat_1_inp = asdict(sat_pair_attri.sat_1_inp)
                sat_2_inp = asdict(sat_pair_attri.sat_2_inp)
                output = list(asdict(sat_pair_attri.output).values())

                # If any combination is rejected by any filter, then do not write that data
                if output[0] != 0:
                    continue

                # Convert all values to strings and write in one go
                row = [
                    *map(str, sat_1_inp.values()),
                    *map(str, sat_2_inp.values()),
                    *map(str, output),
                ]
                write_file.write(",".join(row) + "\n")
                done = done + 1

    with open(
        os.path.join(data_dir_path, "latest_processed_data_file.txt"), "w"
    ) as file:
        file.write(processed_file_name)

    print(f"TLEs in file processed successfully in file {processed_file_name}.")


if __name__ == "__main__":
    training_data_maker_from_physical_algorithm(
        D_start=datetime(2025, 10, 22, 2, 0, 0),
        D_stop=datetime(2025, 10, 26, 0, 0, 0),
        from_latest_raw_data_file=False,
        raw_file_name="celestrak_data_2000.txt",
    )
