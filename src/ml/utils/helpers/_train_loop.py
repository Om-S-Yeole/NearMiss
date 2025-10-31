from math import isclose

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def train_validation_loop(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    stage: str,
    filter_rej_code_threshold: float | None = None,
    device: torch.device | None = None,
    epochs: int = 10,
    validation_loss_threshold: float = 0.01,
    patience: int = 2,
):
    """
    Perform the training and validation loop for the given model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train and validate.
    loss_fn : nn.Module
        The loss function to optimize.
    optimizer : optim.Optimizer
        The optimizer to use for training.
    train_dataloader : DataLoader
        The DataLoader providing the training data.
    validation_dataloader : DataLoader
        The DataLoader providing the validation data.
    stage : str
        The stage of the training, one of 'filter', 'approach', or 'likelihood'.
    filter_rej_code_threshold : float, optional
        The threshold for rejection code filtering, required if stage is 'filter'. Default is None.
    device : torch.device, optional
        The device to perform computations on (e.g., 'cpu' or 'cuda'). Default is None.
    epochs : int, optional
        The number of epochs to train for. Default is 10.
    validation_loss_threshold : float, optional
        The threshold for validation loss to determine worsening. Default is 0.01.
    patience : int, optional
        The number of consecutive epochs allowed to worsen before stopping. Default is 2.

    Returns
    -------
    None
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_threshold = validation_loss_threshold
    validation_loss = None
    train_dataloader_len = len(train_dataloader)
    worsen_epochs = 0

    model.set_mode("train")

    for epoch_no in range(1, epochs + 1):
        model.train(mode=True)
        print(f"---------- Epoch: {epoch_no} ------------")
        for mini_batch_no, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred.view(-1, 1), y.view(-1, 1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(
                f"mini batch no. {mini_batch_no + 1} finished. [{mini_batch_no + 1}/{train_dataloader_len}]"
            )

        model.set_mode("val")
        validation_mini_batches_loss_sum = 0
        validation_dataloader_len = len(validation_dataloader)
        validation_y_true = []
        validation_y_pred = []

        with torch.no_grad():
            for _, (X, y) in enumerate(validation_dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred.view(-1, 1), y.view(-1, 1)).item()
                validation_mini_batches_loss_sum += loss

                # Collect for metric computation
                validation_y_true.extend(y.view(-1).cpu().tolist())
                validation_y_pred.extend(pred.view(-1).cpu().tolist())

        avg_validation_loss = (
            validation_mini_batches_loss_sum / validation_dataloader_len
        )

        print("--------------")
        if stage == "filter":
            if not filter_rej_code_threshold:
                raise ValueError(
                    "Expected value of filter_rej_code_threshold if value of argument 'stage' is set as 'filter'"
                )
            y_probabs = torch.sigmoid(torch.tensor(validation_y_pred))
            y_preds = (y_probabs > filter_rej_code_threshold).float()
            y_true = torch.tensor(validation_y_true).float()
            acc = (y_preds == y_true).float().mean().item()
            print(f"Validation Accuracy for epoch {epoch_no}: {acc * 100:.2f}%")

        elif stage == "approach":
            y_true = torch.tensor(validation_y_true)
            y_pred = torch.tensor(validation_y_pred)
            r_2 = r2_score(validation_y_true, validation_y_pred)
            mae = torch.mean(torch.abs(y_true - y_pred)).item()
            print(f"Validation R2 for epoch {epoch_no}: {r_2:.4f}")
            print(f"Validation MAE for epoch {epoch_no}: {mae:.6f}")

        else:
            y_true = torch.tensor(validation_y_true)
            y_pred = torch.sigmoid(torch.tensor(validation_y_pred))
            r_2 = r2_score(validation_y_true, y_pred.cpu().tolist())
            mae = torch.mean(torch.abs(y_true - y_pred)).item()
            print(f"Validation R2 for epoch {epoch_no}: {r_2:.4f}")
            print(f"Validation MAE for epoch {epoch_no}: {mae:.6f}")

        print(
            f"Average validation loss for epoch no. {epoch_no}: {avg_validation_loss}"
        )

        if isclose(avg_validation_loss, 0.0, abs_tol=1e-5):
            print(
                f"Average validation loss reached {avg_validation_loss} which is very close to 0.0. Hence terminating model training in order to avoid overfitting."
            )
            return

        if validation_loss:
            if (avg_validation_loss - validation_loss) > loss_threshold:
                worsen_epochs += 1
                if worsen_epochs > patience:
                    print(f"Validation loss of previous epoch: {validation_loss}")
                    print(f"Validation loss of this epoch: {avg_validation_loss}")
                    print(
                        f"Current loss is greater than previous loss than required threshold {loss_threshold}. Hence terminating model training."
                    )
                    return
            else:
                validation_loss = avg_validation_loss
                worsen_epochs = 0
        else:
            validation_loss = avg_validation_loss
            worsen_epochs = 0

        print("-----------------------------------------")
