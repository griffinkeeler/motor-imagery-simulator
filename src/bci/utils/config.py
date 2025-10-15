from __future__ import annotations
import os, json, random
from pathlib import Path
import numpy as np
import yaml


# load YAML; seed control; paths
def load_config(path: str | os.PathLike[str]) -> dict:
    """
    Opens a YAML file and returns its plain contents as a Python dict.

    Args:
        path: Path to YAML file.

    Returns:
        Dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int | None) -> None:
    """ Make randomness deterministic."""
    # If seed is not determined in YAML, everything is skipped
    if seed is None:
        return
    # Ensures that Python’s internal hashing (used in dicts, sets, etc.) is deterministic.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Controls Python's built-in random module
    random.seed(seed)
    # Controls NumPy's random number generator
    np.random.seed(seed)


def ensure_dir(path: str) -> Path:
    """Safely create result folders."""
    # Converts path to Path object for consistent behavior
    p = Path(path)
    # Creates directory and missing parent directories
    p.mkdir(parents=True, exist_ok=True)
    # Returns the path
    return p


def save_json(obj:dict, path:str) -> None:
    """Save metrics and outputs consistently."""
    with open(path, "w") as f:
        # Converts dict to JSON and writes it to the file
        json.dump(obj, f, indent=2)


def dump_effective_config(cfg: dict, out_dir:str | os.PathLike[str]) -> None:
    """Stores the exact config used for this run alongside results."""
    ensure_dir(out_dir)
    with open(Path(out_dir) / "effective_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def format_path_template(template: str, context: dict) -> str:
    ctx = {"subject_id": "unknown", "dataset_type": "unspecified", **context}
    return str(Path(template.format(**ctx)))