import numpy as np
import torch
import torch.nn as nn


class LikelihoodStageNN(nn.Module):
    """
    A neural network model for the 'likelihood' stage.

    Parameters
    ----------
    mean : list or np.ndarray
        The mean value for feature normalization.
    std : list or np.ndarray
        The standard deviation value for feature normalization.

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
    linear_gelu_stack : nn.Sequential
        A sequential stack of linear layers, GELU activations, batch normalization, and dropout layers.
    """

    def __init__(self, mean: list | np.ndarray, std: list | np.ndarray):
        """
        Initialize the LikelihoodStageNN model.

        Parameters
        ----------
        mean : list or np.ndarray
            The mean value for feature normalization.
        std : list or np.ndarray
            The standard deviation value for feature normalization.
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.mode = "train"  # 'train', 'val', or 'test'

        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(61, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
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
        outputs = self.linear_gelu_stack(x)
        return outputs

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
        LikelihoodStageNN
            The loaded model set to 'test' mode.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.set_mode("test")
        return model
