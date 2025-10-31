import pandas as pd
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    A PyTorch Dataset for training models with processed data.

    Parameters
    ----------
    processed_df : pd.DataFrame
        The processed DataFrame containing input features and target attributes.
    stage : str, optional
        The stage of the dataset, one of 'filter', 'approach', or 'likelihood'. Default is 'filter'.
    mean : float, optional
        The mean value for feature normalization. Default is None.
    std : float, optional
        The standard deviation value for feature normalization. Default is None.
    device : torch.device, optional
        The device to store tensors (e.g., 'cpu' or 'cuda'). Default is None.

    Attributes
    ----------
    df : pd.DataFrame
        The processed DataFrame.
    stage : str
        The stage of the dataset.
    input_attri : list of str
        The list of input feature column names.
    target_attri : list of str
        The list of target attribute column names.
    features : torch.Tensor
        The tensor containing input features.
    targets : torch.Tensor
        The tensor containing target values.
    features_mean : torch.Tensor
        The mean of features (computed if mean is None).
    features_std : torch.Tensor
        The standard deviation of features (computed if std is None).
    """

    def __init__(
        self,
        processed_df: pd.DataFrame,
        stage: str = "filter",
        mean: float | None = None,
        std: float | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize the TrainDataset.

        Parameters
        ----------
        processed_df : pd.DataFrame
            The processed DataFrame containing input features and target attributes.
        stage : str, optional
            The stage of the dataset, one of 'filter', 'approach', or 'likelihood'. Default is 'filter'.
        mean : float, optional
            The mean value for feature normalization. Default is None.
        std : float, optional
            The standard deviation value for feature normalization. Default is None.
        device : torch.device, optional
            The device to store tensors (e.g., 'cpu' or 'cuda'). Default is None.
        """
        if not isinstance(stage, str):
            raise TypeError(f"Expected type of stage is str. Got {type(stage)}")
        if stage not in ["filter", "approach", "likelihood"]:
            raise ValueError(
                f"Expected value of stage is either 'filter', 'approach', or 'likelihood'. Got '{stage}'"
            )

        self.df: pd.DataFrame = processed_df
        self.stage = stage

        self.input_attri = [
            "ndot_1",
            "nddot_1",
            "bstar_1",
            "inclo_1",
            "nodeo_1",
            "ecco_1",
            "argpo_1",
            "mo_1",
            "no_kozai_1",
            "a_1",
            "altp_1",
            "alta_1",
            "argpdot_1",
            "mdot_1",
            "nodedot_1",
            "am_1",
            "em_1",
            "im_1",
            "Om_1",
            "om_1",
            "mm_1",
            "nm_1",
            "r_x_1",
            "r_y_1",
            "r_z_1",
            "v_x_1",
            "v_y_1",
            "v_z_1",
            "tle_age_1",
            "sat_radius_1",
            "ndot_2",
            "nddot_2",
            "bstar_2",
            "inclo_2",
            "nodeo_2",
            "ecco_2",
            "argpo_2",
            "mo_2",
            "no_kozai_2",
            "a_2",
            "altp_2",
            "alta_2",
            "argpdot_2",
            "mdot_2",
            "nodedot_2",
            "am_2",
            "em_2",
            "im_2",
            "Om_2",
            "om_2",
            "mm_2",
            "nm_2",
            "r_x_2",
            "r_y_2",
            "r_z_2",
            "v_x_2",
            "v_y_2",
            "v_z_2",
            "tle_age_2",
            "sat_radius_2",
        ]

        self.target_attri = ["filter_rej_code", "t_close", "ln_d_min", "probab"]

        self.features = None
        self.targets = None

        if self.stage == "filter":
            self.features = torch.tensor(
                self.df[self.input_attri].values, dtype=torch.float32, device=device
            )
            self.targets = torch.tensor(
                self.df[self.target_attri[0]].values, dtype=torch.float32, device=device
            ).unsqueeze(1)
        elif self.stage == "approach":
            self.features = torch.tensor(
                self.df[self.input_attri].values, dtype=torch.float32, device=device
            )
            self.targets = torch.tensor(
                self.df[self.target_attri[1:3]].values,
                dtype=torch.float32,
                device=device,
            )
        else:
            self.features = torch.tensor(
                self.df[self.input_attri + self.target_attri[1:3]].values,
                dtype=torch.float32,
                device=device,
            )
            self.targets = torch.tensor(
                self.df[self.target_attri[-1]].values,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1)

        # Normalize features
        if mean is not None and std is not None:
            self.features = (self.features - mean) / (std + 1e-8)
        else:
            self.features_mean = self.features.mean(dim=0)
            self.features_std = self.features.std(dim=0) + 1e-8
            self.features = (self.features - self.features_mean) / self.features_std

    def __getitem__(self, index):
        """
        Get a single data point from the dataset.

        Parameters
        ----------
        index : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple
            A tuple containing the features and target for the given index.
        """
        return self.features[index], self.targets[index]

    def __len__(self):
        """
        Get the number of data points in the dataset.

        Returns
        -------
        int
            The number of data points in the dataset.
        """
        return len(self.features)
