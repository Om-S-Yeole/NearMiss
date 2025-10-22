import yaml


def _read_yaml(path: str) -> dict:
    """
    Read the yaml file for given specified path and returns the content.
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
