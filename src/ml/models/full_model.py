import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ml.models.stage import load_stage


def train_full_model(
    from_latest_processed_file: bool = True,
    processed_file_name: str | None = None,
    save_training_evaluations: bool = True,
    filter_stage_lr: float | int = 1e-3,
    filter_stage_epochs: int = 15,
    approach_stage_lr: float | int = 1e-3,
    approach_stage_epochs: int = 15,
    likelihood_stage_lr: float | int = 1e-3,
    likelihood_stage_epochs: int = 15,
    filter_rej_code_threshold: float = 0.5,
) -> None:
    """
    Train the full model across three stages: filter, approach, and likelihood.

    Parameters
    ----------
    from_latest_processed_file : bool, optional
        Whether to use the latest processed file. Default is True.
    processed_file_name : str, optional
        The name of the processed file to use. Required if `from_latest_processed_file` is False. Default is None.
    save_training_evaluations : bool, optional
        Whether to save the training evaluations to a file. Default is True.
    filter_stage_lr : float or int, optional
        Learning rate for the filter stage. Default is 1e-3.
    filter_stage_epochs : int, optional
        Number of epochs for the filter stage. Default is 15.
    approach_stage_lr : float or int, optional
        Learning rate for the approach stage. Default is 1e-3.
    approach_stage_epochs : int, optional
        Number of epochs for the approach stage. Default is 15.
    likelihood_stage_lr : float or int, optional
        Learning rate for the likelihood stage. Default is 1e-3.
    likelihood_stage_epochs : int, optional
        Number of epochs for the likelihood stage. Default is 15.
    filter_rej_code_threshold : float, optional
        Threshold for the filter rejection code. Must be between 0 and 1. Default is 0.5.

    Raises
    ------
    TypeError
        If any parameter is of an incorrect type.
    ValueError
        If `filter_rej_code_threshold` is not between 0 and 1.

    Returns
    -------
    None
    """
    if not isinstance(from_latest_processed_file, bool):
        raise TypeError(
            f"Expected type of from_latest_processed_file is bool. Got {type(from_latest_processed_file)}."
        )
    if not (isinstance(processed_file_name, str) or processed_file_name is None):
        raise TypeError(
            f"Expected type of from_latest_processed_file is str or None. Got {type(processed_file_name)}"
        )
    if not isinstance(save_training_evaluations, bool):
        raise TypeError(
            f"Expected type of save_training_evaluations is bool. Got {type(save_training_evaluations)}."
        )
    if not isinstance(filter_stage_lr, (float, int)):
        raise TypeError(
            f"Expected type of filter_stage_lr is float or int. Got {type(filter_stage_lr)}"
        )
    if not isinstance(filter_stage_epochs, (float, int)):
        raise TypeError(
            f"Expected type of filter_stage_epochs is int. Got {type(filter_stage_epochs)}"
        )
    if not isinstance(approach_stage_lr, (float, int)):
        raise TypeError(
            f"Expected type of approach_stage_lr is float or int. Got {type(approach_stage_lr)}"
        )
    if not isinstance(approach_stage_epochs, (float, int)):
        raise TypeError(
            f"Expected type of approach_stage_epochs is int. Got {type(approach_stage_epochs)}"
        )
    if not isinstance(likelihood_stage_lr, (float, int)):
        raise TypeError(
            f"Expected type of likelihood_stage_lr is float or int. Got {type(likelihood_stage_lr)}"
        )
    if not isinstance(likelihood_stage_epochs, (float, int)):
        raise TypeError(
            f"Expected type of likelihood_stage_epochs is int. Got {type(likelihood_stage_epochs)}"
        )
    if not isinstance(filter_rej_code_threshold, float):
        raise TypeError(
            f"Expected type of filter_rej_code_threshold is float. Got {type(filter_rej_code_threshold)}"
        )

    if not (filter_rej_code_threshold > 0 and filter_rej_code_threshold < 1):
        raise ValueError(
            f"filter_rej_code_threshold must be between 0 and 1. Got {filter_rej_code_threshold}"
        )

    device = None
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = torch.device("cuda")
    else:
        torch.set_default_device("cpu")
        device = torch.device("cpu")

    print(f"Implementation is working on device: {device}")

    data_file_name = None
    data_file_path = None
    if from_latest_processed_file:
        latest_processed_data_file_path = os.path.join(
            "..", "data", "latest_processed_data_file.txt"
        )
        with open(latest_processed_data_file_path, "r") as f:
            data_file_name = f.readline().strip()

        data_file_path = os.path.join("..", "data", "processed", data_file_name)
    else:
        data_file_name = processed_file_name
        data_file_path = os.path.join("..", "data", "processed", processed_file_name)

    df: pd.DataFrame = pd.read_csv(data_file_path)

    filter_stage_model_state_path = (
        "./ml/models/checkpoints/filter_stage/filter_stage.pth"
    )
    approach_stage_model_state_path = (
        "./ml/models/checkpoints/approach_stage/approach_stage.pth"
    )
    likelihood_stage_model_state_path = (
        "./ml/models/checkpoints/likelihood_stage/likelihood_stage.pth"
    )

    generator = torch.Generator(device=device).manual_seed(42)

    # ------------------------- Stage 1: Filter -------------------------

    print("-----------------------------------------------------")
    print("Starting of the Stage 1: Filter")
    print("-----------------------------------------------------")

    filter_stage_evaluation_results = load_stage(
        df=df,
        stage="filter",
        loss_fn=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam,
        stage_model_state_path=filter_stage_model_state_path,
        generator=generator,
        device=device,
        stage_lr=filter_stage_lr,
        stage_epochs=filter_stage_epochs,
        val_dataset_ratio=0.2,
        train_dataloader_batch_size=1028,
        train_dataloader_to_shuffle=True,
        val_dataloader_batch_size=256,
        val_dataloader_to_shuffle=True,
        evaluation_dataloader_batch_size=1028,
        validation_loss_threshold=0.01,
        patience=2,
        filter_rej_code_threshold=filter_rej_code_threshold,
    )

    predicted_filter_rej_code_mask = (
        torch.sigmoid(filter_stage_evaluation_results).view(-1)
        > filter_rej_code_threshold
    )  # Here True means rejected and False means not rejected

    print("-----------------------------------------------------")
    print("End of the Stage 1: Filter")
    print("-----------------------------------------------------")

    # ------------------------- Stage 2: Approach -------------------------

    print("-----------------------------------------------------")
    print("Starting of the Stage 2: Approach")
    print("-----------------------------------------------------")

    # Take negation of each element in the tensor in order to take only those who have predicted filter rejection code as 0
    mask_np = (~predicted_filter_rej_code_mask.cpu().numpy()).astype(bool)
    df_approach = df.loc[mask_np]

    predicted_t_close_ln_d_min = load_stage(
        df=df_approach,
        stage="approach",
        loss_fn=nn.MSELoss,
        optimizer=optim.Adam,
        stage_model_state_path=approach_stage_model_state_path,
        generator=generator,
        device=device,
        stage_lr=approach_stage_lr,
        stage_epochs=approach_stage_epochs,
        val_dataset_ratio=0.2,
        train_dataloader_batch_size=1028,
        train_dataloader_to_shuffle=True,
        val_dataloader_batch_size=256,
        val_dataloader_to_shuffle=True,
        evaluation_dataloader_batch_size=1028,
        validation_loss_threshold=0.01,
        patience=2,
        filter_rej_code_threshold=None,
    )

    predicted_t_close_ln_d_min = predicted_t_close_ln_d_min.reshape(-1, 2)

    print("-----------------------------------------------------")
    print("End of the Stage 2: Approach")
    print("-----------------------------------------------------")

    # ------------------------- Stage 3: Likelihood -------------------------

    print("-----------------------------------------------------")
    print("Starting of the Stage 3: Likelihood")
    print("-----------------------------------------------------")

    df_approach.loc[:, ["t_close", "ln_d_min"]] = (
        predicted_t_close_ln_d_min.cpu().numpy()
    )
    df_likelihood = df_approach

    predicted_probab = load_stage(
        df=df_likelihood,
        stage="likelihood",
        loss_fn=nn.BCEWithLogitsLoss,
        optimizer=optim.Adam,
        stage_model_state_path=likelihood_stage_model_state_path,
        generator=generator,
        device=device,
        stage_lr=likelihood_stage_lr,
        stage_epochs=likelihood_stage_epochs,
        val_dataset_ratio=0.2,
        train_dataloader_batch_size=1028,
        train_dataloader_to_shuffle=True,
        val_dataloader_batch_size=256,
        val_dataloader_to_shuffle=True,
        evaluation_dataloader_batch_size=1028,
        validation_loss_threshold=0.01,
        patience=2,
        filter_rej_code_threshold=None,
    )

    predicted_probab = torch.sigmoid(predicted_probab)

    print("-----------------------------------------------------")
    print("End of the Stage 3: Likelihood")
    print("-----------------------------------------------------")

    # ------------------------- Save Evaluations -------------------------

    if save_training_evaluations:
        root, _ = os.path.splitext(data_file_name)
        training_eval_dir = os.path.join("..", "data", "training_eval")
        os.makedirs(training_eval_dir, exist_ok=True)
        evaluations_file_path = os.path.join(
            "..", "data", "training_eval", f"{root}_train_eval.csv"
        )

        with open(evaluations_file_path, "w") as write_file:
            print("Saving the training evaluations to file. Please wait...")

            write_file.write("filter_rej_code,t_close,ln_d_min,probab\n")
            i = 0
            for bool_rej_code in predicted_filter_rej_code_mask:
                if bool_rej_code:  # Means if the pair is rejected to collide by filter
                    write_file.write("1,0.0,0.0,0.0\n")
                else:  # Means if the pair is not rejected by filter
                    write_file.write(
                        f"0,{predicted_t_close_ln_d_min[i][0]},{predicted_t_close_ln_d_min[i][1]},{predicted_probab[i]}\n"
                    )
                    i += 1

            print(f"Training evaluations are saved at location {evaluations_file_path}")

    return None
