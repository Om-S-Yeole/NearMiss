import yaml


def _read_yaml(path: str) -> dict:
    """
    Read the YAML file for the given specified path and return its content.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed content of the YAML file.

    Raises
    ------
    TypeError
        If the provided path is not a string.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    FileNotFoundError
        If the specified file does not exist.
    """
    if not isinstance(path, str):
        raise TypeError(f"Expected type of path is str. Got {type(path)}.")

    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAMLError for file path {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{path}' not found")
