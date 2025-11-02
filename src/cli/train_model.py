import argparse
from argparse import Namespace

from ml import train_full_model


def arg_parser():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing:
        - from_latest_processed_file : bool
            Whether to use the latest processed file. Default is True.
        - processed_file_name : str or None
            The name of the processed file to use. Required if `from_latest_processed_file` is False.
        - save_training_evaluations : bool
            Whether to save the training evaluations to a file. Default is True.
        - filter_stage_lr : float or None
            Learning rate for the filter stage.
        - filter_stage_epochs : int or None
            Number of epochs for the filter stage.
        - approach_stage_lr : float or None
            Learning rate for the approach stage.
        - approach_stage_epochs : int or None
            Number of epochs for the approach stage.
        - likelihood_stage_lr : float or None
            Learning rate for the likelihood stage.
        - likelihood_stage_epochs : int or None
            Number of epochs for the likelihood stage.
        - filter_rej_code_threshold : float
            Threshold for the filter rejection code. Must be between 0 and 1. Default is 0.5.
    """

    parser = argparse.ArgumentParser(
        description="Train the full 3 stage neural network"
    )

    parser.add_argument(
        "--from_latest_processed_file",
        type=bool,
        default=True,
        help="Whether to use the latest processed file. Default is True.",
    )

    parser.add_argument(
        "--processed_file_name",
        type=str,
        help="The name of the processed file to use. Required if `from_latest_processed_file` is False.",
    )

    parser.add_argument(
        "--save_training_evaluations",
        type=bool,
        default=True,
        help="Whether to save the training evaluations to a file. Default is True.",
    )

    parser.add_argument(
        "--filter_stage_lr", type=float, help="Learning rate for the filter stage."
    )

    parser.add_argument(
        "--filter_stage_epochs", type=int, help="Number of epochs for the filter stage."
    )

    parser.add_argument(
        "--approach_stage_lr", type=float, help="Learning rate for the approach stage."
    )

    parser.add_argument(
        "--approach_stage_epochs",
        type=int,
        help="Number of epochs for the approach stage.",
    )

    parser.add_argument(
        "--likelihood_stage_lr",
        type=float,
        help="Learning rate for the likelihood stage.",
    )

    parser.add_argument(
        "--likelihood_stage_epochs",
        type=int,
        help="Number of epochs for the likelihood stage.",
    )

    parser.add_argument(
        "--filter_rej_code_threshold",
        type=float,
        default=0.5,
        help="Threshold for the filter rejection code. Must be between 0 and 1. Default is 0.5",
    )

    args: Namespace = parser.parse_args()
    return args


def main():
    """
    Main function to execute the model training process.

    Parses command-line arguments, prepares arguments for the training function, and calls `train_full_model`.

    Returns
    -------
    None
        This function does not return any value.
    """

    args: Namespace = arg_parser()

    train_model_args = {}

    if args.from_latest_processed_file:
        train_model_args.update(
            {"from_latest_processed_file": args.from_latest_processed_file}
        )
    if args.processed_file_name:
        train_model_args.update({"processed_file_name": args.processed_file_name})
    if args.save_training_evaluations:
        train_model_args.update(
            {"save_training_evaluations": args.save_training_evaluations}
        )
    if args.filter_stage_lr:
        train_model_args.update({"filter_stage_lr": args.filter_stage_lr})
    if args.filter_stage_epochs:
        train_model_args.update({"filter_stage_epochs": args.filter_stage_epochs})
    if args.approach_stage_lr:
        train_model_args.update({"approach_stage_lr": args.approach_stage_lr})
    if args.approach_stage_epochs:
        train_model_args.update({"approach_stage_epochs": args.approach_stage_epochs})
    if args.likelihood_stage_lr:
        train_model_args.update({"likelihood_stage_lr": args.likelihood_stage_lr})
    if args.likelihood_stage_epochs:
        train_model_args.update(
            {"likelihood_stage_epochs": args.likelihood_stage_epochs}
        )
    if args.filter_rej_code_threshold:
        train_model_args.update(
            {"filter_rej_code_threshold": args.filter_rej_code_threshold}
        )

    train_full_model(**train_model_args)


if __name__ == "__main__":
    main()
