import json

import numpy as np
import mlflow
from pathlib import Path


from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from src.bci.utils.config import (
    load_config,
    set_all_seeds,
    ensure_dir,
    dump_effective_config,
    save_json,
    format_path_template,
)


def make_pipeline(cfg):
    """Creates sklearn pipeline."""
    return Pipeline(
        [
            ("csp", CSP(**cfg["csp"])),
            ("lda", LDA(**cfg["lda"])),
        ]
    )


def make_param_grid(cfg):
    """Returns a dictionary of the hyperparameter grid."""
    return {
        "csp__n_components": cfg["param_grid"]["n_components"],
        "lda__shrinkage": cfg["param_grid"]["lda_shrinkage"],
        "lda__solver": cfg["param_grid"]["lda_solver"],
    }


def run_nested(X, y, cfg, run_name, results_dir):
    """Run nested cross-validation."""
    # Outer CV for model evaluation
    outer = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["rskf"]["n_splits"],
        n_repeats=cfg["cv"]["rskf"]["n_repeats"],
        random_state=cfg["cv"]["rskf"]["random_state"],
    )
    # Inner CV for model selection (tuning hyperparameters)
    inner = StratifiedKFold(
        n_splits=cfg["cv"]["skf"]["n_splits"],
        shuffle=cfg["cv"]["skf"]["shuffle"],
        random_state=cfg["cv"]["skf"]["random_state"],
    )
    # Create pipeline
    pipeline = make_pipeline(cfg)
    # Define hyperparameter grid
    param_grid = make_param_grid(cfg)
    # Define grid search (inner loop)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1,
    )

    # Outer fold scores
    outer_scores = []
    # Best parameters
    outer_params = []

    mlflow.set_tracking_uri(
        "/Users/griffinkeeler/PycharmProjects/motor-imagery-simulator/mlruns"
    )
    mlflow.set_experiment(cfg["experiment"]["name"])
    mlflow.autolog()
    with mlflow.start_run(run_name=run_name):
        for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            print(f"Outer fold {fold_idx}")

            # Inner loop tuning happen automatically here inside GridSearchCV
            search.fit(X[train_idx], y[train_idx])

            # Evaluate the best model on the outer test set
            best_model = search.best_estimator_
            test_score = best_model.score(X[test_idx], y[test_idx])

            outer_scores.append(test_score)
            outer_params.append(search.best_params_)

        print(f"Mean BA: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
        print("Best params for outer fold:")
        for params in outer_params:
            print(params)


def run_rskf(X, y, cfg, run_name, results_dir):
    # 1) Initialize models
    # **: {'a': 1} -> a=1
    csp = CSP(**cfg["csp"])
    lda = LDA(**cfg["lda"])

    seed = cfg["experiment"].get("seed")

    # 2) Cross-validation setup
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg["cv"]["rskf"]["n_splits"],
        n_repeats=cfg["cv"]["rskf"]["n_repeats"],
        random_state=cfg["cv"]["rskf"]["random_state"],
    )

    fold_scores = []

    # 3) MLflow logging context
    mlflow.set_tracking_uri(
        "/Users/griffinkeeler/PycharmProjects/motor-imagery-simulator/mlruns"
    )
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "seed": seed,
                **cfg["csp"],
                **cfg["lda"],
                **cfg["cv"]["rskf"],
            }
        )

        # 7) Perform RSKF
        for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_csp = csp.fit_transform(X_train, y_train)
            X_test_csp = csp.transform(X_test)

            lda.fit(X_train_csp, y_train)
            y_pred = lda.predict(X_test_csp)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            fold_scores.append(bal_acc)
            mlflow.log_metric("balanced_accuracy_fold", bal_acc, step=fold_idx)

        # 4) Aggregate and save
        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        ci95_lo = float(np.percentile(fold_scores, 2.5))
        ci95_hi = float(np.percentile(fold_scores, 97.5))

        summary = {"mean BA": mean_acc, "std": std_acc}

        save_json(summary, results_dir / cfg["output"]["metrics_file"])
        np.savetxt(
            results_dir / cfg["output"]["folds_scores_file"],
            np.array(fold_scores),
            delimiter=",",
            fmt="%.4f",
        )

        mlflow.log_metric("balanced_accuracy_mean", mean_acc)
        mlflow.log_metric("balanced_accuracy_std", std_acc)
        mlflow.log_metric("balanced_accuracy_ci95_lo", ci95_lo)
        mlflow.log_metric("balanced_accuracy_ci95_hi", ci95_hi)
        mlflow.log_text(json.dumps(cfg, indent=2), "config_used.json")
        mlflow.log_artifacts(str(results_dir))

    print(f"Run complete. Mean accuracy: {mean_acc:.3f}")


def main():
    # 1) Load configuration and seed
    here = Path(__file__).parents[2]
    cfg_path = Path(f"{here}/experiments/models/mi_csp_lda/config.yaml")
    # Loads configs as a dictionary
    cfg = load_config(cfg_path)
    seed = cfg["experiment"].get("seed")
    set_all_seeds(seed)
    subject_id = cfg["data"]["subject_id"]
    dataset_type = cfg["data"]["dataset_type"]
    cv_name = cfg["cv"]["rskf"]["name"]
    run_name = str(f"sub_{subject_id}-{dataset_type}-{cv_name}-seed{seed}")

    # 2) Prepare results directory
    results_dir_str = format_path_template(
        str(Path(cfg_path).parent / cfg["output"]["dir"]),
        {"subject_id": subject_id, "dataset_type": dataset_type},
    )
    results_dir = ensure_dir(results_dir_str)
    dump_effective_config(cfg, results_dir)

    # 3) Load data
    X = np.load(
        format_path_template(
            str(Path(cfg_path.parents[3] / cfg["data"]["X_path"])),
            {"subject_id": subject_id, "dataset_type": dataset_type},
        )
    )
    y = np.load(
        format_path_template(
            str(Path(cfg_path.parents[3] / cfg["data"]["y_path"])),
            {"subject_id": subject_id, "dataset_type": dataset_type},
        )
    )

    run_nested(X, y, cfg, run_name, results_dir)


if __name__ == "__main__":
    main()
