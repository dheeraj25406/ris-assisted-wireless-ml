"""
Comprehensive Model Evaluation and Comparison Module.
Evaluates Random Forest, MLP, XGBoost, and a Mean Baseline on the strictly untouched Test Set (42 UEs, 168 rows).
Computes MAE, RMSE, R2, and breakdowns across RIS element configurations (16, 64, 256, 1024).
Generates publication-quality diagnostic plots.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_TARGET,
)
from src.preprocessing import prepare_group_splits


def evaluate_all():
    print("=" * 70)
    print("EVALUATING ALL MODELS ON STRICTLY UNTOUCHED TEST SET")
    print("=" * 70)

    splits = prepare_group_splits()
    train_df = splits["train_df"]
    test_df = splits["test_df"].copy()
    feature_names = splits["feature_names"]

    X_train_raw = splits["X_train_raw"]
    y_train = splits["y_train"]

    X_test_raw = splits["X_test_raw"]
    X_test_scaled = splits["X_test_scaled"]
    y_test = splits["y_test"]

    print(f"Test Set: {test_df['UE_ID'].nunique()} UEs, {len(test_df)} observations.")

    # 1. Baseline Model (Predicts Mean of Training Set)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_raw, y_train)
    dummy_preds = dummy.predict(X_test_raw)

    # 2. Load Trained Models
    rf_model = joblib.load(MODELS_DIR / "random_forest" / "rf_model.joblib")
    mlp_model = joblib.load(MODELS_DIR / "mlp" / "mlp_model.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost" / "xgb_model.joblib")

    # Generate Predictions
    rf_preds = rf_model.predict(X_test_raw)
    mlp_preds = mlp_model.predict(X_test_scaled)
    xgb_preds = xgb_model.predict(X_test_raw)

    models_dict = {
        "Baseline (Mean)": dummy_preds,
        "Random Forest": rf_preds,
        "Multi-Layer Perceptron (MLP)": mlp_preds,
        "XGBoost Regressor": xgb_preds,
    }

    # Record overall performance
    overall_metrics = {}
    for name, preds in models_dict.items():
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        overall_metrics[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
        }
        print(f"\n{name:30s} -> MAE: {mae:.4f} dB | RMSE: {rmse:.4f} dB | R²: {r2:.4f}")

    # Breakdown by RIS Elements (16, 64, 256, 1024)
    elements_breakdown = {}
    for elem in [16, 64, 256, 1024]:
        mask = test_df["RIS_Elements"] == elem
        y_sub = y_test[mask]
        elements_breakdown[f"N_{elem}"] = {}
        for name, preds in models_dict.items():
            p_sub = preds[mask]
            mae = mean_absolute_error(y_sub, p_sub)
            rmse = np.sqrt(mean_squared_error(y_sub, p_sub))
            elements_breakdown[f"N_{elem}"][name] = {
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
            }

    # Save Predictions DataFrame
    test_df["Pred_Baseline"] = dummy_preds
    test_df["Pred_RandomForest"] = rf_preds
    test_df["Pred_MLP"] = mlp_preds
    test_df["Pred_XGBoost"] = xgb_preds
    test_df["Residual_RF"] = test_df[PRIMARY_TARGET] - rf_preds
    test_df["Residual_MLP"] = test_df[PRIMARY_TARGET] - mlp_preds
    test_df["Residual_XGB"] = test_df[PRIMARY_TARGET] - xgb_preds

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)
    print(f"\nTest predictions saved to {PREDICTIONS_DIR / 'test_predictions.csv'}")

    # Save Metrics JSON
    comparison_results = {
        "target": PRIMARY_TARGET,
        "test_sample_count": len(test_df),
        "test_ue_count": test_df["UE_ID"].nunique(),
        "overall_test_metrics": overall_metrics,
        "metrics_by_ris_elements": elements_breakdown,
        "best_model_overall": "Random Forest" if overall_metrics["Random Forest"]["RMSE"] <= overall_metrics["XGBoost Regressor"]["RMSE"] else "XGBoost Regressor"
    }

    with open(METRICS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    # =========================================================================
    # DIAGNOSTIC PLOTS
    # =========================================================================
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # Plot 1: Actual vs Predicted Scatter Plots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"]

    min_val = min(y_test.min(), min(rf_preds.min(), mlp_preds.min(), xgb_preds.min())) - 1.0
    max_val = max(y_test.max(), max(rf_preds.max(), mlp_preds.max(), xgb_preds.max())) + 1.0

    for idx, (name, preds) in enumerate(models_dict.items()):
        ax = axes[idx]
        ax.scatter(y_test, preds, color=colors[idx], alpha=0.6, edgecolors="none", s=40)
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Ideal 1:1 Line")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"{name}\nMAE: {overall_metrics[name]['MAE']:.3f} dB | R²: {overall_metrics[name]['R2']:.3f}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Actual Path Loss With RIS (dB)", fontsize=11)
        ax.set_ylabel("Predicted Path Loss (dB)", fontsize=11)
        ax.legend(loc="upper left")

    plt.suptitle("Model Evaluation: Actual vs. Predicted Path Loss on Unseen Test UEs", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Residual Distributions
    plt.figure(figsize=(10, 6))
    residuals_df = pd.DataFrame({
        "Random Forest": test_df["Residual_RF"],
        "MLP": test_df["Residual_MLP"],
        "XGBoost": test_df["Residual_XGB"],
    })
    for col, col_color in zip(residuals_df.columns, ["#2ca02c", "#1f77b4", "#ff7f0e"]):
        sns.kdeplot(residuals_df[col], fill=True, label=f"{col} (Mean: {residuals_df[col].mean():.3f}, Std: {residuals_df[col].std():.3f})", color=col_color, alpha=0.3, lw=2)
    plt.axvline(0, color="black", linestyle="--", lw=1.5)
    plt.title("Error Residual Distributions on Unseen Test UEs (y_true - y_pred)", fontsize=13, fontweight="bold")
    plt.xlabel("Residual Error (dB)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_residual_distributions.png", dpi=300)
    plt.close()

    # Plot 3: Residuals vs Predicted (Homoscedasticity Check)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_data = [
        ("Random Forest", test_df["Pred_RandomForest"], test_df["Residual_RF"], "#2ca02c"),
        ("MLP", test_df["Pred_MLP"], test_df["Residual_MLP"], "#1f77b4"),
        ("XGBoost", test_df["Pred_XGBoost"], test_df["Residual_XGB"], "#ff7f0e"),
    ]
    for ax, (title, preds, res, color) in zip(axes, plot_data):
        ax.scatter(preds, res, color=color, alpha=0.6, s=35)
        ax.axhline(0, color="black", linestyle="--", lw=1.5)
        ax.set_title(f"{title} Residuals vs. Fitted", fontweight="bold")
        ax.set_xlabel("Predicted Path Loss (dB)", fontsize=11)
        ax.set_ylabel("Residual (Actual - Pred) [dB]", fontsize=11)
        ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_residuals_vs_predicted.png", dpi=300)
    plt.close()

    # Plot 4: Error by RIS Element Configuration
    configs = ["N_16", "N_64", "N_256", "N_1024"]
    rf_rmses = [elements_breakdown[c]["Random Forest"]["RMSE"] for c in configs]
    mlp_rmses = [elements_breakdown[c]["Multi-Layer Perceptron (MLP)"]["RMSE"] for c in configs]
    xgb_rmses = [elements_breakdown[c]["XGBoost Regressor"]["RMSE"] for c in configs]

    bar_df = pd.DataFrame({
        "Configuration": ["16 Elements", "64 Elements", "256 Elements", "1024 Elements"],
        "Random Forest": rf_rmses,
        "MLP": mlp_rmses,
        "XGBoost": xgb_rmses,
    }).melt(id_vars="Configuration", var_name="Model", value_name="RMSE_dB")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=bar_df, x="Configuration", y="RMSE_dB", hue="Model", palette=["#2ca02c", "#1f77b4", "#ff7f0e"])
    plt.title("Test RMSE by RIS Element Configuration", fontsize=13, fontweight="bold")
    plt.ylabel("RMSE (dB)", fontsize=12)
    plt.xlabel("RIS Reflector Size", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Model", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_error_by_ris_elements.png", dpi=300)
    plt.close()

    # Plot 5: Feature Importance Comparison (RF vs XGBoost)
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names)
    imp_df = pd.DataFrame({"Random Forest": rf_imp, "XGBoost": xgb_imp}).reset_index().melt(
        id_vars="index", var_name="Model", value_name="Importance"
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, y="index", x="Importance", hue="Model", palette=["#2ca02c", "#ff7f0e"])
    plt.title("Feature Importance Comparison: RF vs. XGBoost", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score (Gini / Gain)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eval_feature_importance_comparison.png", dpi=300)
    plt.close()

    print("\nAll evaluation figures generated successfully.")
    return comparison_results


if __name__ == "__main__":
    evaluate_all()
