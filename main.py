"""
Main Pipeline Runner for ML-Based Performance Prediction of RIS-Assisted Wireless Systems.
Orchestrates:
1. Data Validation & Scientific Checks
2. Exploratory Data Analysis (EDA)
3. Feature Engineering & Leakage-Free Preprocessing
4. Model Training (Random Forest, MLP, XGBoost with GroupKFold)
5. Model Evaluation & Comparison on Unseen Test UEs
6. Candidate Targets Study
7. Sample Prediction Verification
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.candidate_targets_study import study_candidate_targets
from src.data_validation import validate_dataset
from src.evaluate_models import evaluate_all
from src.exploratory_analysis import run_eda
from src.predict import RISPredictor, compare_all_configurations
from src.preprocessing import prepare_group_splits
from src.train_mlp import train_mlp
from src.train_random_forest import train_random_forest
from src.train_xgboost import train_xgboost


def run_pipeline(stage: str = "all"):
    print("=" * 80)
    print("  ML-BASED PERFORMANCE PREDICTION OF RIS-ASSISTED WIRELESS SYSTEMS")
    print("  Major Project Pipeline Orchestrator")
    print("=" * 80)

    if stage in ["all", "validate"]:
        print("\n[STEP 1/6] Running Dataset Inspection & Scientific Validation...")
        validate_dataset()

    if stage in ["all", "eda"]:
        print("\n[STEP 2/6] Running Exploratory Data Analysis & Plot Generation...")
        run_eda()

    if stage in ["all", "preprocess"]:
        print("\n[STEP 3/6] Running Feature Engineering & Group-Based Splitting...")
        prepare_group_splits()

    if stage in ["all", "train"]:
        print("\n[STEP 4/6] Training Models with Group-Aware Cross-Validation...")
        print(" -> 4a. Random Forest Regressor")
        train_random_forest()
        print(" -> 4b. Multi-Layer Perceptron (MLP)")
        train_mlp()
        print(" -> 4c. XGBoost Regressor")
        train_xgboost()

    if stage in ["all", "evaluate"]:
        print("\n[STEP 5/6] Evaluating All Models on Strictly Untouched Test UEs...")
        evaluate_all()

    if stage in ["all", "study-targets"]:
        print("\n[STEP 6/6] Studying Secondary Candidate Targets (No-RIS PL & RIS Gain)...")
        study_candidate_targets()

    # Final Demo Prediction
    print("\n[DEMO] Testing Prediction Interface with Sample UE Location...")
    predictor = RISPredictor("random_forest")
    res = predictor.predict_point(rx_x=12.0, rx_y=18.0, ris_elements=1024)
    print(f"UE at (12.0, 18.0, 1.5), N=1024 -> Predicted Path Loss: {res['Predicted_Path_Loss_With_RIS_dB']:.4f} dB")
    print("\nFour-Configuration Sweep:")
    print(compare_all_configurations(12.0, 18.0))

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Figures:     {PROJECT_ROOT / 'results' / 'figures'}")
    print(f"  Metrics:     {PROJECT_ROOT / 'results' / 'metrics'}")
    print(f"  Predictions: {PROJECT_ROOT / 'results' / 'predictions'}")
    print(f"  Models:      {PROJECT_ROOT / 'models'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RIS ML Project Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="evaluate",
        choices=["all", "validate", "eda", "preprocess", "train", "evaluate", "study-targets"],
        help="Pipeline stage to execute (default: evaluate)",
    )
    args = parser.parse_args()
    run_pipeline(stage=args.stage)
