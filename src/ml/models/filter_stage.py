import torch
import torch.nn as nn


class FilterStageNN(nn.Module):
    """
    A neural network model for the 'filter' stage.

    Parameters
    ----------
    mean : float
        The mean value for feature normalization.
    std : float
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
    linear_relu_stack : nn.Sequential
        A sequential stack of linear layers, ReLU activations, and dropout layers.
    """

    def __init__(self, mean: float, std: float):
        """
        Initialize the FilterStageNN model.

        Parameters
        ----------
        mean : float
            The mean value for feature normalization.
        std : float
            The standard deviation value for feature normalization.
        """
        super().__init__()

        self.flatten = nn.Flatten()
        self.mode = "train"  # 'train', 'val', or 'test'

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

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
        logits = self.linear_relu_stack(x)
        return logits
