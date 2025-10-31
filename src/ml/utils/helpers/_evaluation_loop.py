import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluation_loop(
    model: nn.Module, evaluation_dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    """
    Perform an evaluation loop on the given model and dataloader.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to evaluate.
    evaluation_dataloader : DataLoader
        The DataLoader providing the evaluation data.
    device : torch.device
        The device to perform computations on (e.g., 'cpu' or 'cuda').

    Returns
    -------
    torch.Tensor
        A tensor containing the concatenated predictions from the model.
    """
    model.set_mode("test")
    output_list = []

    with torch.no_grad():
        for _, (X, _) in enumerate(evaluation_dataloader):
            X = X.to(device)
            pred = torch.flatten(model(X))
            output_list.append(pred)

    return torch.cat(output_list, dim=0)
