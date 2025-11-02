import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from ml.data import TestFeaturesDataset
from ml.models import (
    APPROACH_STAGE_MODEL_STATE_PATH,
    FILTER_STAGE_MODEL_STATE_PATH,
    LIKELIHOOD_STAGE_MODEL_STATE_PATH,
    ApproachStageNN,
    FilterStageNN,
    LikelihoodStageNN,
)
from ml.utils import evaluation_loop


def full_model_prediction_or_test(
    file_name: str, mode: str = "predict", batch_size: int = 512
):
    """
    Perform predictions or testing using the full 3-stage neural network model.

    Parameters
    ----------
    file_name : str
        Name of the file containing input data for predictions or testing.
    mode : str, optional
        Mode of operation, either 'predict' or 'test'. Default is 'predict'.
    batch_size : int, optional
        Batch size for the DataLoader. Default is 512.

    Raises
    ------
    TypeError
        If `file_name`, `mode`, or `batch_size` is of an incorrect type.
    ValueError
        If `mode` is not 'test' or 'predict', or if `batch_size` is less than or equal to 0.

    Returns
    -------
    None
        This function does not return any value. Results are either printed or saved to a file.
    """
    if not isinstance(file_name, str):
        raise TypeError(f"Expected type of file_name is str. Got {type(file_name)}.")
    if not isinstance(mode, str):
        raise TypeError(f"Expected type of mode is str. Got {type(mode)}.")

    if not isinstance(batch_size, int):
        raise TypeError(f"Expected type of batch_size is int. Got {type(batch_size)}.")

    if mode not in ["test", "predict"]:
        raise ValueError(
            f"Valid arguments for mode are 'test' and 'predict' only. Got '{mode}'."
        )

    if batch_size <= 0:
        raise ValueError(
            f"Value of batch_size must be integer greater than 0. Got {batch_size}"
        )

    device = None
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = torch.device("cuda")
    else:
        torch.set_default_device("cpu")
        device = torch.device("cpu")

    print(f"Implementation is working on device: {device}")

    test_file_path = None
    if mode == "test":
        processed_dir_path = os.path.join("..", "data", "processed")
        test_file_path = os.path.join(processed_dir_path, file_name)
    else:
        to_predict_dir_path = os.path.join("..", "data", "to_predict")
        test_file_path = os.path.join(to_predict_dir_path, file_name)

    df: pd.DataFrame = pd.read_csv(test_file_path)
    df_features = None
    df_true_targets = None

    if mode == "predict":
        df_features = df
    else:
        df_features = df.drop(columns=["filter_rej_code", "ln_d_min", "probab"])
        df_true_targets = df[["filter_rej_code", "ln_d_min", "probab"]]

    # ------------------------- Stage 1: Filter -------------------------
    print("Starting of the Stage 1: Filter")

    filter_stage_checkpoint = torch.load(
        FILTER_STAGE_MODEL_STATE_PATH, map_location=device
    )

    filter_stage_dataset = TestFeaturesDataset(
        df_features.copy(),
        device,
    )
    filter_stage_dataloader = DataLoader(
        filter_stage_dataset, batch_size=batch_size, shuffle=False
    )

    filter_stage_model = FilterStageNN.load_trained_model(
        FILTER_STAGE_MODEL_STATE_PATH, device
    )

    filter_stage_eval_results = evaluation_loop(
        filter_stage_model, filter_stage_dataloader, device, True
    )

    filter_stage_mask = (
        torch.sigmoid(filter_stage_eval_results).view(-1)
        > filter_stage_checkpoint["filter_rej_code_threshold"]
    )  # If value is True, then pair is rejected. If value is False, then pair is not rejected.

    filter_stage_mask_np = filter_stage_mask.cpu().numpy()

    print("End of the Stage 1: Filter")

    # ------------------------- Stage 2: Approach -------------------------

    print("Starting of the Stage 2: Approach")

    mask_np = ((~filter_stage_mask).cpu().numpy()).astype(bool)
    df_approach_temp: pd.DataFrame = df_features.loc[mask_np]

    approach_stage_dataset = TestFeaturesDataset(
        df_approach_temp.copy(),
        device,
    )
    approach_stage_dataloader = DataLoader(
        approach_stage_dataset, batch_size=batch_size, shuffle=False
    )

    approach_stage_model = ApproachStageNN.load_trained_model(
        APPROACH_STAGE_MODEL_STATE_PATH, device
    )

    approach_stage_eval_results = (
        evaluation_loop(approach_stage_model, approach_stage_dataloader, device, True)
        .cpu()
        .numpy()
    )

    print("End of the Stage 2: Approach")

    # ------------------------- Stage 3: Likelihood -------------------------

    print("Starting of the Stage 3: Likelihood")

    df_likelihood_temp = df_approach_temp
    df_likelihood_temp.loc[:, "ln_d_min"] = approach_stage_eval_results

    likelihood_stage_dataset = TestFeaturesDataset(
        df_likelihood_temp.copy(),
        device,
    )
    likelihood_stage_dataloader = DataLoader(
        likelihood_stage_dataset, batch_size=batch_size, shuffle=False
    )

    likelihood_stage_model = LikelihoodStageNN.load_trained_model(
        LIKELIHOOD_STAGE_MODEL_STATE_PATH, device
    )

    likelihood_stage_eval_results = (
        torch.sigmoid(
            evaluation_loop(
                likelihood_stage_model, likelihood_stage_dataloader, device, True
            )
        )
        .cpu()
        .numpy()
    )

    print("End of the Stage 3: Likelihood")

    # ------------------------- Testing / Prediction -------------------------

    complete_predictions = []
    i = 0
    for pred_filter_bool in filter_stage_mask_np:
        if pred_filter_bool:  # Means the pair is rejected by the filter
            complete_predictions.append([1.0, 0.0, 0.0, 0.0])
        else:
            complete_predictions.append(
                [
                    0.0,
                    approach_stage_eval_results[i],
                    likelihood_stage_eval_results[i],
                ]
            )
            i += 1

    complete_predictions = np.array(complete_predictions)

    if mode == "test":
        # Give the classification accuracy for filter stage
        print("Calculating test metrics. Please wait...")
        acc = np.array(
            complete_predictions[:, 0] == df_true_targets["filter_rej_code"].to_numpy(),
            dtype=np.float32,
        ).mean()
        # Give the R2 score for rest of the targets
        r2_ln_d_min = r2_score(df_true_targets["ln_d_min"], complete_predictions[:, 1])
        r2_probab = r2_score(df_true_targets["probab"], complete_predictions[:, 2])

        print("-------------- Test Metrics -----------------")
        print(f"Accuracy for Filter Rejection Code: {acc:.3f}")
        print(f"R2 Score for ln_d_min: {r2_ln_d_min:.3f}")
        print(f"R2 Score for probab: {r2_probab:.3f}")
        print("---------------------------------------------")
    else:
        predictions_dir_path = os.path.join("..", "data", "predictions")
        os.makedirs(predictions_dir_path, exist_ok=True)
        predictions_file_name = f"{os.path.splitext(file_name)[0]}_predictions.csv"

        predictions_file_path = os.path.join(
            predictions_dir_path, predictions_file_name
        )

        df_to_csv = pd.DataFrame(
            complete_predictions,
            columns=["filter_rej_code", "ln_d_min", "probab"],
        )

        print("Writing the predictions to the file. Please wait...")
        df_to_csv.to_csv(predictions_file_path, index=False)
        print(
            f"Predictions written successfully in the file with path {predictions_file_path}"
        )
