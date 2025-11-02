import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ml.data import TrainDataset
from ml.models.approach_stage import ApproachStageNN
from ml.models.filter_stage import FilterStageNN
from ml.models.likelihood_stage import LikelihoodStageNN
from ml.utils import evaluation_loop, train_validation_loop


def load_stage(
    df: pd.DataFrame,
    stage: str,
    loss_fn: nn.Module,  # Just pass the class of the loss criterion not an instance
    optimizer: torch.optim.Optimizer,  # Just pass the class of the optimizer not an instance
    stage_model_state_path: str,
    generator: torch.Generator,
    device: torch.device,
    stage_lr: float | int = 0.01,
    stage_epochs: int = 15,
    val_dataset_ratio: float = 0.2,
    train_dataloader_batch_size: int = 1028,
    train_dataloader_to_shuffle: bool = True,
    val_dataloader_batch_size: int = 256,
    val_dataloader_to_shuffle: bool = True,
    evaluation_dataloader_batch_size: int = 1028,
    validation_loss_threshold: float = 0.01,
    patience: int = 2,
    filter_rej_code_threshold: float | None = None,
) -> torch.Tensor:
    """
    Load and train a specific stage of the model.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data for the stage.
    stage : str
        The stage to load, one of 'filter', 'approach', or 'likelihood'.
    loss_fn : nn.Module
        The class of the loss function to use.
    optimizer : torch.optim.Optimizer
        The class of the optimizer to use.
    stage_model_state_path : str
        The path to save the model state after training.
    generator : torch.Generator
        The random generator for reproducibility.
    device : torch.device
        The device to perform computations on (e.g., 'cpu' or 'cuda').
    stage_lr : float or int, optional
        The learning rate for the stage. Default is 0.01.
    stage_epochs : int, optional
        The number of epochs for training. Default is 15.
    val_dataset_ratio : float, optional
        The ratio of the validation dataset. Default is 0.2.
    train_dataloader_batch_size : int, optional
        The batch size for the training DataLoader. Default is 1028.
    train_dataloader_to_shuffle : bool, optional
        Whether to shuffle the training DataLoader. Default is True.
    val_dataloader_batch_size : int, optional
        The batch size for the validation DataLoader. Default is 256.
    val_dataloader_to_shuffle : bool, optional
        Whether to shuffle the validation DataLoader. Default is True.
    evaluation_dataloader_batch_size : int, optional
        The batch size for the evaluation DataLoader. Default is 1028.
    validation_loss_threshold : float, optional
        The threshold for validation loss to determine worsening. Default is 0.01.
    patience : int, optional
        The number of consecutive epochs allowed to worsen before stopping. Default is 2.
    filter_rej_code_threshold : float or None, optional
        The threshold for rejection code filtering, required if stage is 'filter'. Default is None.

    Returns
    -------
    torch.Tensor
        A tensor containing the evaluation results for the stage.

    Raises
    ------
    ValueError
        If `filter_rej_code_threshold` is not provided when the stage is 'filter'.
    """
    full_dataset = TrainDataset(df, stage)
    train_dataset, validation_dataset = random_split(
        full_dataset, [1.0 - val_dataset_ratio, val_dataset_ratio], generator=generator
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_dataloader_batch_size,
        shuffle=train_dataloader_to_shuffle,
        generator=generator,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=val_dataloader_batch_size,
        shuffle=val_dataloader_to_shuffle,
        generator=generator,
    )
    evaluation_dataloader = DataLoader(
        full_dataset, batch_size=evaluation_dataloader_batch_size
    )

    stage_model = None
    if stage == "filter":
        stage_model = FilterStageNN(
            mean=full_dataset.features_mean,
            std=full_dataset.features_std,
            filter_rej_code_threshold=filter_rej_code_threshold,
        ).to(device)
    elif stage == "approach":
        stage_model = ApproachStageNN(
            mean=full_dataset.features_mean, std=full_dataset.features_std
        ).to(device)
    else:
        stage_model = LikelihoodStageNN(
            mean=full_dataset.features_mean, std=full_dataset.features_std
        ).to(device)

    stage_loss_fn: nn.Module = loss_fn()
    stage_optimizer: torch.optim.Optimizer = optimizer(
        params=stage_model.parameters(), lr=stage_lr
    )

    args = {
        "model": stage_model,
        "loss_fn": stage_loss_fn,
        "optimizer": stage_optimizer,
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "stage": stage,
        "device": device,
        "epochs": stage_epochs,
        "validation_loss_threshold": validation_loss_threshold,
        "patience": patience,
    }

    if stage == "filter":
        if filter_rej_code_threshold is not None:
            args.update({"filter_rej_code_threshold": filter_rej_code_threshold})
        else:
            raise ValueError(
                "Value of filter_rej_code_threshold is expected when the stage is set as 'filter'."
            )

    train_validation_loop(**args)

    checkpoint = {
        "model_state": stage_model.state_dict(),
        "model_class": stage_model.__class__.__name__,
        "mean": stage_model.mean.tolist(),
        "std": stage_model.std.tolist(),
    }

    if stage == "filter":
        checkpoint.update(
            {
                "filter_rej_code_threshold": stage_model.filter_rej_code_threshold.item(),
            }
        )

    torch.save(checkpoint, stage_model_state_path)

    evaluation_results = evaluation_loop(stage_model, evaluation_dataloader, device)

    return evaluation_results
