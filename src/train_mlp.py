"""
Training and hyperparameter tuning module for Multi-Layer Perceptron (MLP) Regressor.
Standardizes both input features and regression target using TransformedTargetRegressor.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_SEED
from src.preprocessing import prepare_group_splits


def train_mlp():
    print("=" * 60)
    print("Starting Multi-Layer Perceptron (MLP) Training")
    print("=" * 60)

    splits = prepare_group_splits()
    train_df = splits["train_df"]
    val_df = splits["val_df"]

    X_train_scaled = splits["X_train_scaled"]
    y_train = splits["y_train"]
    groups_train = train_df["UE_ID"].values

    X_val_scaled = splits["X_val_scaled"]
    y_val = splits["y_val"]

    # Grid search for architecture, regularization, and solver
    param_grid = [
        {
            "hidden_layer_sizes": [(64, 32), (32, 16), (64, 64)],
            "alpha": [1e-3, 1e-2, 0.1],
            "solver": ["lbfgs"],
            "max_iter": [500],
        },
        {
            "hidden_layer_sizes": [(64, 32), (32, 16)],
            "alpha": [1e-3, 1e-2],
            "learning_rate_init": [0.001],
            "solver": ["adam"],
            "max_iter": [500],
        }
    ]

    gkf = GroupKFold(n_splits=5)
    best_score = float("inf")
    best_params = None

    all_params = list(ParameterGrid(param_grid))
    print(f"Running GroupKFold CV (5 folds) across {len(all_params)} parameter sets...")

    for params in all_params:
        cv_scores = []
        for cv_train_idx, cv_val_idx in gkf.split(X_train_scaled, y_train, groups=groups_train):
            base_mlp = MLPRegressor(
                **params,
                activation="relu",
                random_state=RANDOM_SEED,
            )
            regressor = TransformedTargetRegressor(regressor=base_mlp, transformer=StandardScaler())
            regressor.fit(X_train_scaled[cv_train_idx], y_train[cv_train_idx])
            preds = regressor.predict(X_train_scaled[cv_val_idx])
            rmse = np.sqrt(mean_squared_error(y_train[cv_val_idx], preds))
            cv_scores.append(rmse)

        mean_cv_rmse = np.mean(cv_scores)
        if mean_cv_rmse < best_score:
            best_score = mean_cv_rmse
            best_params = params

    print(f"Best Group-CV RMSE: {best_score:.4f} dB with params: {best_params}")

    # Retrain on full train set
    final_mlp = MLPRegressor(
        **best_params,
        activation="relu",
        random_state=RANDOM_SEED,
    )
    final_regressor = TransformedTargetRegressor(regressor=final_mlp, transformer=StandardScaler())
    final_regressor.fit(X_train_scaled, y_train)

    fitted_mlp = final_regressor.regressor_

    # Plot loss curve if available (Adam) or convergence progress
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if hasattr(fitted_mlp, "loss_curve_"):
        plt.plot(fitted_mlp.loss_curve_, color="#1f77b4", lw=2, label="Normalized Training Loss")
        plt.xlabel("Iteration / Epoch", fontsize=12)
        plt.ylabel("Loss (Standardized Scale)", fontsize=12)
    else:
        # L-BFGS loss history
        plt.axhline(fitted_mlp.loss_, color="#1f77b4", linestyle="--", lw=2, label=f"L-BFGS Final Loss: {fitted_mlp.loss_:.6f}")
        plt.xlabel("Convergence", fontsize=12)
        plt.ylabel("Objective Function Loss", fontsize=12)
    plt.title("MLP Training Loss Convergence", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_loss_curve.png", dpi=300)
    plt.close()

    # Evaluate on Validation Set
    val_preds = final_regressor.predict(X_val_scaled)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)

    print("\nValidation Performance (42 unseen UEs, 168 rows):")
    print(f"  MAE:  {val_mae:.4f} dB")
    print(f"  RMSE: {val_rmse:.4f} dB")
    print(f"  R²:   {val_r2:.4f}")

    # Save model artifacts
    mlp_dir = MODELS_DIR / "mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_regressor, mlp_dir / "mlp_model.joblib")

    metrics = {
        "model_name": "Multi-Layer Perceptron (MLP)",
        "best_params": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in best_params.items()
        },
        "best_group_cv_rmse": float(best_score),
        "total_iterations": int(fitted_mlp.n_iter_),
        "final_loss": float(fitted_mlp.loss_),
        "validation_metrics": {
            "MAE": float(val_mae),
            "RMSE": float(val_rmse),
            "R2": float(val_r2),
        },
    }

    with open(METRICS_DIR / "mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return final_regressor, metrics


if __name__ == "__main__":
    train_mlp()
