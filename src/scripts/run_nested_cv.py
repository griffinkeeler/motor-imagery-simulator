import numpy as np

from pathlib import Path

from src.bci.evaluation.cv.nested import run_nested

from src.bci.utils.config import (
    load_config,
    set_all_seeds,
    format_path_template,
)

def main():
    # 1) Load configuration and seed
    here = Path(__file__).parents[2]
    cfg_path = Path(f"{here}/configs/base.yaml")
    # Loads configs as a dictionary
    cfg = load_config(cfg_path)
    seed = cfg["experiment"].get("seed")
    set_all_seeds(seed)
    subject_id = cfg["data"]["subject_id"]
    dataset_type = cfg["data"]["dataset_type"]
    cv_name = cfg["cv"]["rskf"]["name"]
    cv_type = cfg["cv"]["cv_type"]
    run_name = str(f"sub_{subject_id}-{dataset_type}-{cv_type}-{cv_name}-seed{seed}")

    # 3) Load data
    X = np.load(
        format_path_template(
            str(Path(cfg_path.parents[1] / cfg["data"]["X_path"])),
            {"subject_id": subject_id, "dataset_type": dataset_type},
        )
    )
    y = np.load(
        format_path_template(
            str(Path(cfg_path.parents[1] / cfg["data"]["y_path"])),
            {"subject_id": subject_id, "dataset_type": dataset_type},
        )
    )

    run_nested(X, y, cfg, run_name)

if __name__ == "__main__":
    main()