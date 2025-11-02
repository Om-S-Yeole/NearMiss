import numpy as np
import torch
import torch.nn as nn


class FilterStageNN(nn.Module):
    """
    A neural network model for the 'filter' stage.

    Parameters
    ----------
    mean : list or np.ndarray
        The mean value for feature normalization.
    std : list or np.ndarray
        The standard deviation value for feature normalization.
    filter_rej_code_threshold : float
        Threshold for the filter rejection code.

    Attributes
    ----------
    flatten : nn.Flatten
        A layer to flatten the input tensor.
    mode : str
        The current mode of the model ('train', 'val', or 'test').
    mean : torch.Tensor
        A buffer to store the mean value for normalization.
    std : torch.Tensor
        A buffer to store the standard deviation value for normalization.
    linear_relu_stack : nn.Sequential
        A sequential stack of linear layers, ReLU activations, and dropout layers.
    filter_rej_code_threshold : torch.Tensor
        Threshold for the filter rejection code.
    """

    def __init__(
        self,
        mean: list | np.ndarray,
        std: list | np.ndarray,
        filter_rej_code_threshold: float,
    ):
        """
        Initialize the FilterStageNN model.

        Parameters
        ----------
        mean : list or np.ndarray
            The mean value for feature normalization.
        std : list or np.ndarray
            The standard deviation value for feature normalization.
        filter_rej_code_threshold : float
            Threshold for the filter rejection code.
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.mode = "train"  # 'train', 'val', or 'test'

        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer(
            "filter_rej_code_threshold", torch.tensor(filter_rej_code_threshold)
        )

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 1),
        )

    def set_mode(self, mode: str):
        """
        Set the mode of the model.

        Parameters
        ----------
        mode : str
            The mode to set, one of ['train', 'val', 'test'].

        Raises
        ------
        ValueError
            If the mode is not one of ['train', 'val', 'test'].
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode must be one of ['train', 'val', 'test']")
        self.mode = mode
        if mode == "train":
            super().train(True)
        else:
            super().train(False)

    def forward(self, x: torch.Tensor):
        """
        Perform a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the model.
        """
        x = self.flatten(x)
        x = (x - self.mean) / (self.std)
        logits = self.linear_relu_stack(x)
        return logits

    @classmethod
    def load_trained_model(cls, path: str, device: torch.device = "cpu"):
        """
        Load a trained model from a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        device : torch.device, optional
            The device to load the model onto (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

        Returns
        -------
        FilterStageNN
            The loaded model set to 'test' mode.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
            filter_rej_code_threshold=checkpoint["filter_rej_code_threshold"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.set_mode("test")
        return model
