import json, tempfile, os

import numpy as np
import pandas as pd
import mlflow
from pathlib import Path


from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline


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


def run_nested(X, y, cfg, run_name):
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

    # Outer fold scores
    outer_scores = []
    # Best parameters
    outer_params = []

    mlflow.set_tracking_uri(
        "/Users/griffinkeeler/PycharmProjects/motor-imagery-simulator/mlruns"
    )
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name=run_name) as parent_run:
        # Log the effective config (JSON) once
        mlflow.log_text(json.dumps(cfg, indent=2), "config_effective.json")

        for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            print(f"Outer fold {fold_idx}")

            # Define grid search (inner loop)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=-1,
                return_train_score=True,
                refit=True,
            )

            with mlflow.start_run(run_name=f"outer_fold_{fold_idx:02d}", nested=True):

                # Inner loop tuning happen automatically here inside GridSearchCV
                search.fit(X[train_idx], y[train_idx])

                # Evaluate the best model on the outer test set
                y_pred = search.best_estimator_.predict(X[test_idx])
                fold_score = balanced_accuracy_score(y[test_idx], y_pred)
                mlflow.log_metric("outer_bal_acc", float(fold_score))

                best_params = search.best_params_
                mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})

                results_df = pd.DataFrame(search.cv_results_)
                # FIXME: Results not loading in mlflow UI
                with tempfile.TemporaryDirectory() as tmp:
                    csv_path = Path(tmp) / f"cv_results_fold_{fold_idx:02d}.csv"
                    results_df.to_csv(csv_path, index=False)
                    mlflow.log_artifacts(str(csv_path), "cv_results")

                outer_scores.append(fold_score)
                outer_params.append(best_params)

        # Aggregate and save
        mean_acc = float(np.mean(outer_scores))
        std_acc = float(np.std(outer_scores))
        ci95_low = float(np.percentile(outer_scores, 2.5))
        ci95_hi = float(np.percentile(outer_scores, 97.5))

        mlflow.log_metric("balanced_accuracy_mean", mean_acc)
        mlflow.log_metric("balanced_accuracy_std", std_acc)
        mlflow.log_metric("balanced_accuracy_ci95_low", ci95_low)
        mlflow.log_metric("balanced_accuracy_ci95_hi", ci95_hi)

        summary = {
            "outer_fold_scores": [float(s) for s in outer_scores],
            "outer_mean_bal_acc": mean_acc,
            "outer_std_bal_acc": std_acc,
            "outer_ci95": [ci95_low, ci95_hi],
            "best_params_per_fold": outer_params,
        }
        mlflow.log_dict(summary, "metrics.json")

        print(f"Mean BA: {mean_acc:.3f} ± {std_acc:.3f}")
        print("Best params for outer fold:")
        for params in outer_params:
            print(params)