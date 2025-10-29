"""
Command-line interface for data creation.

This script provides a CLI to fetch data from an API and create training datasets for machine learning models.
"""

import argparse
from argparse import Namespace
from datetime import datetime, timedelta
from nearmiss import reterive_data_from_api, training_data_maker_from_physical_algorithm


def fetch_data_from_api_and_make_training_dataset(
    D_start: datetime,
    D_stop: datetime,
    retrieve_from_api: bool = True,
    from_latest_raw_data_file: bool = True,
    raw_data_file_name: str | None = None,
    optional_args_for_data_creation: dict | None = None,
    optional_args_for_physical_algorithm: dict | None = None,
):
    """
    Fetch data from the API and create a training dataset.

    Parameters
    ----------
    D_start : datetime
        Start time of the data creation window.
    D_stop : datetime
        End time of the data creation window.
    retrieve_from_api : bool, optional
        Whether to retrieve data from the API. Default is True.
    from_latest_raw_data_file : bool, optional
        Whether to use the latest raw data file for data creation. Default is True.
    raw_data_file_name : str or None, optional
        Name of the raw data file to use. Default is None.
    optional_args_for_data_creation : dict or None, optional
        Additional arguments for data creation. Default is None.
    optional_args_for_physical_algorithm : dict or None, optional
        Additional arguments for the physical algorithm. Default is None.

    Returns
    -------
    None
    """

    # --- Retrieve data from the api ---
    if retrieve_from_api:
        reterive_data_from_api()

    # --- Create the data ---
    args = {
        "D_start": D_start,
        "D_stop": D_stop,
        "from_latest_raw_data_file": from_latest_raw_data_file,
        "raw_file_name": raw_data_file_name,
    }

    if optional_args_for_data_creation:
        args.update(optional_args_for_data_creation)
    if optional_args_for_physical_algorithm:
        args.update(optional_args_for_physical_algorithm)

    training_data_maker_from_physical_algorithm(**args)


def arg_parser():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Data fetching and making training model for ML algorithm."
    )

    parser.add_argument(
        "D_start",
        type=str,  # This string must be converted to datetime object
        help="String representation of the date from which time window to be started. Format is %%Y-%%m-%%d %%H:%%M:%%S. For example, 2025-10-29 15:40:28. Assumed timezone of the given time is UTC.",
    )

    parser.add_argument(
        "D_stop",
        type=str,  # This string must be converted to datetime object
        help="String representation of the date from which time window to be ended. Format is %%Y-%%m-%%d %%H:%%M:%%S. For example, 2025-11-03 06:16:57. Assumed timezone of the given time is UTC.",
    )

    parser.add_argument(
        "--retrieve_from_api", action="store_true", help="Whether to data from API"
    )

    parser.add_argument(
        "--from_latest_raw_data_file",
        action="store_true",
        help="Whether to create data from latest raw data file",
    )

    parser.add_argument(
        "--raw_data_file_name",
        type=str,
        help="Name of the raw data file in 'data/raw/' directory from which the training data to be created.",
    )

    parser.add_argument(
        "--t_interval",
        type=float,  # This float must be converted to timedelta object
        help="Time interval in seconds using which to discretize the orbit propagation using KDTree.",
    )

    parser.add_argument(
        "--r_threshold_KDtree",
        type=float,
        help="Threshold r (in km) in order to decide pairs from KDTree. More r means more potential pairs that fall in spherical region where r is radius.",
    )

    parser.add_argument(
        "--r_obj_1", type=float, help="Radius of primary satellites in meters."
    )

    parser.add_argument(
        "--r_obj_2", type=float, help="Radius of secondary satellites in meters."
    )

    parser.add_argument(
        "--Dist",
        type=float,
        help="Threshold distance in km for Apoapsis Periapsis Filter",
    )

    args: Namespace = parser.parse_args()
    return args


def main():
    """
    Main function to execute the data creation process.

    Parses command-line arguments, prepares optional arguments, and calls the data creation function.

    Returns
    -------
    None
    """

    args: Namespace = arg_parser()

    format_string: str = "%Y-%m-%d %H:%M:%S"
    D_start: datetime = datetime.strptime(args.D_start, format_string)
    D_stop: datetime = datetime.strptime(args.D_stop, format_string)

    optional_args_for_data_creation: dict = {}

    if args.t_interval:
        optional_args_for_data_creation["t_interval"] = timedelta(
            seconds=args.t_interval
        )
    if args.r_threshold_KDtree:
        optional_args_for_data_creation["r_threshold_KDtree"] = args.r_threshold_KDtree

    if not optional_args_for_data_creation:
        optional_args_for_data_creation = None

    optional_args_for_physical_algorithm: dict = {}

    if args.r_obj_1:
        optional_args_for_physical_algorithm["r_obj_1"] = args.r_obj_1
    if args.r_obj_2:
        optional_args_for_physical_algorithm["r_obj_2"] = args.r_obj_2
    if args.Dist:
        optional_args_for_physical_algorithm["Dist"] = args.Dist

    if not optional_args_for_physical_algorithm:
        optional_args_for_physical_algorithm = None

    fetch_data_from_api_and_make_training_dataset(
        D_start=D_start,
        D_stop=D_stop,
        retrieve_from_api=args.retrieve_from_api,
        from_latest_raw_data_file=args.from_latest_raw_data_file,
        raw_data_file_name=args.raw_data_file_name,
        optional_args_for_data_creation=optional_args_for_data_creation,
        optional_args_for_physical_algorithm=optional_args_for_physical_algorithm,
    )


if __name__ == "__main__":
    main()
