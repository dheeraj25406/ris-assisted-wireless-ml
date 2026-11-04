"""
Training and hyperparameter tuning module for XGBoost Regressor.
Employs group-aware cross-validation on the training set to prevent UE leakage.
Extracts feature importances and logs validation metrics.
"""

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import METRICS_DIR, MODELS_DIR, RANDOM_SEED
from src.preprocessing import prepare_group_splits


def train_xgboost():
    print("=" * 60)
    print("Starting XGBoost Regressor Training with Group-Aware CV")
    print("=" * 60)

    splits = prepare_group_splits()
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    feature_names = splits["feature_names"]

    X_train = splits["X_train_raw"]
    y_train = splits["y_train"]
    groups_train = train_df["UE_ID"].values

    X_val = splits["X_val_raw"]
    y_val = splits["y_val"]

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    gkf = GroupKFold(n_splits=5)
    best_score = float("inf")
    best_params = None

    print(f"Running GroupKFold CV (5 folds) across {len(list(ParameterGrid(param_grid)))} parameter sets...")

    for params in ParameterGrid(param_grid):
        cv_scores = []
        for cv_train_idx, cv_val_idx in gkf.split(X_train, y_train, groups=groups_train):
            model = xgb.XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(X_train[cv_train_idx], y_train[cv_train_idx])
            preds = model.predict(X_train[cv_val_idx])
            rmse = np.sqrt(mean_squared_error(y_train[cv_val_idx], preds))
            cv_scores.append(rmse)

        mean_cv_rmse = np.mean(cv_scores)
        if mean_cv_rmse < best_score:
            best_score = mean_cv_rmse
            best_params = params

    print(f"Best Group-CV RMSE: {best_score:.4f} dB with params: {best_params}")

    # Retrain on full train set
    best_xgb = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    best_xgb.fit(X_train, y_train)

    # Evaluate on Validation Set
    val_preds = best_xgb.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)

    print("\nValidation Performance (42 unseen UEs, 168 rows):")
    print(f"  MAE:  {val_mae:.4f} dB")
    print(f"  RMSE: {val_rmse:.4f} dB")
    print(f"  R²:   {val_r2:.4f}")

    # Feature Importances
    importances = dict(zip(feature_names, [float(x) for x in best_xgb.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\nFeature Importances (Gain/Weight):")
    for feat, imp in sorted_importances.items():
        print(f"  {feat:22s}: {imp:.4f}")

    # Save model artifacts
    xgb_dir = MODELS_DIR / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_xgb, xgb_dir / "xgb_model.joblib")
    best_xgb.save_model(str(xgb_dir / "xgb_model.json"))

    metrics = {
        "model_name": "XGBoost Regressor",
        "best_params": best_params,
        "best_group_cv_rmse": float(best_score),
        "validation_metrics": {
            "MAE": float(val_mae),
            "RMSE": float(val_rmse),
            "R2": float(val_r2),
        },
        "feature_importances": sorted_importances,
    }

    with open(METRICS_DIR / "xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return best_xgb, metrics


if __name__ == "__main__":
    train_xgboost()
