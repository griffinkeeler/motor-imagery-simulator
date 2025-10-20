import json, tempfile, os

import numpy as np
import pandas as pd
import mlflow

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


from src.bci.utils.config import save_json


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