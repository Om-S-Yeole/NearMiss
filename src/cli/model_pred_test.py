import argparse
from argparse import Namespace

from ml import full_model_prediction_or_test


def arg_parser():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - file_name : str
            File name from which to make predictions or testing.
        - mode : str
            Mode of operation, either 'test' or 'predict'.
        - batch_size : int
            Batch size for dataloaders. Default is 512.
    """

    parser = argparse.ArgumentParser(
        description="Test or make Predictions from the pretrained 3 stage neural network model."
    )

    parser.add_argument(
        "file_name",
        type=str,
        help="File name from which to make predictions or testing.",
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["test", "predict"],
        help="Wether to do model testing or predictions. Only one of the valid argument can be passed 'test' or 'predict'.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for dataloaders. Default is 512.",
    )

    args: Namespace = parser.parse_args()
    return args


def main():
    """
    Main function to execute the model prediction or testing process.

    Parses command-line arguments and calls the `full_model_prediction_or_test` function.

    Returns
    -------
    None
        This function does not return any value.
    """

    args: Namespace = arg_parser()
    full_model_prediction_or_test(
        file_name=args.file_name, mode=args.mode, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
