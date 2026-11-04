"""
Automated Pytest Test Suite for RIS ML Project Pipeline.
Verifies dataset integrity, mathematical consistency, zero group leakage,
and inference interface validity.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    CANDIDATE_TARGETS,
    CORE_FEATURES,
    ENGINEERED_FEATURES,
    FULL_FEATURE_SET,
    MODELS_DIR,
    PRIMARY_TARGET,
    RAW_DATA_PATH,
)
from src.data_validation import validate_dataset
from src.feature_engineering import audit_feature_leakage, engineer_features
from src.predict import RISPredictor, compare_all_configurations
from src.preprocessing import prepare_group_splits


def test_raw_dataset_structure():
    """Verifies file existence, row count, column count, and unique UEs."""
    assert RAW_DATA_PATH.exists(), f"Missing dataset file at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert df.shape == (1120, 24), f"Expected shape (1120, 24), got {df.shape}"
    assert df["UE_ID"].nunique() == 280, f"Expected 280 unique UEs, got {df['UE_ID'].nunique()}"
    assert set(df["RIS_Elements"].unique()) == {16, 64, 256, 1024}
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected missing values"


def test_mathematical_consistency_and_provisional_status():
    """Checks positive energies, identity RIS_Gain = PL_No_RIS - PL_With_RIS, and target status."""
    report = validate_dataset(RAW_DATA_PATH)
    assert report["total_rows"] == 1120
    assert report["total_columns"] == 24
    assert report["composite_energy_positive_finite"] is True
    assert report["max_diff_ris_gain_identity_dB"] < 1e-5, "Mathematical identity violation!"
    assert report["is_provisional_unverified"] is True
    assert report["target_status_values"] == ["PROVISIONAL_UNVERIFIED"]


def test_feature_engineering_and_leakage():
    """Checks feature computation and leakage audit."""
    df = pd.read_csv(RAW_DATA_PATH)
    eng_df = engineer_features(df)
    for col in ENGINEERED_FEATURES:
        assert col in eng_df.columns, f"Engineered column {col} missing"

    audit = audit_feature_leakage(FULL_FEATURE_SET, PRIMARY_TARGET)
    assert audit["is_clean"] is True
    assert len(audit["violations"]) == 0


def test_group_split_zero_leakage():
    """Validates that GroupShuffleSplit isolates UEs with 0 train-val-test overlap."""
    splits = prepare_group_splits()
    train_ues = set(splits["train_df"]["UE_ID"].unique())
    val_ues = set(splits["val_df"]["UE_ID"].unique())
    test_ues = set(splits["test_df"]["UE_ID"].unique())

    assert len(train_ues) == 196, f"Expected 196 Train UEs, got {len(train_ues)}"
    assert len(val_ues) == 42, f"Expected 42 Val UEs, got {len(val_ues)}"
    assert len(test_ues) == 42, f"Expected 42 Test UEs, got {len(test_ues)}"

    assert len(train_ues.intersection(val_ues)) == 0, "Leakage between Train and Val UEs!"
    assert len(train_ues.intersection(test_ues)) == 0, "Leakage between Train and Test UEs!"
    assert len(val_ues.intersection(test_ues)) == 0, "Leakage between Val and Test UEs!"

    assert len(splits["train_df"]) == 196 * 4
    assert len(splits["val_df"]) == 42 * 4
    assert len(splits["test_df"]) == 42 * 4


def test_model_artifacts_and_prediction_pipeline():
    """Checks predictor instantiation and plausible output ranges."""
    for model_name in ["random_forest", "mlp", "xgboost"]:
        predictor = RISPredictor(model_name)
        pred = predictor.predict_point(rx_x=5.0, rx_y=15.0, ris_elements=256)
        val = pred["Predicted_Path_Loss_With_RIS_dB"]
        assert isinstance(val, (float, np.floating))
        assert 60.0 <= val <= 110.0, f"Predicted path loss {val} dB out of realistic bounds!"

    # Test sweep
    sweep_df = compare_all_configurations(rx_x=10.0, rx_y=25.0)
    assert len(sweep_df) == 4
    assert list(sweep_df["RIS_Elements"]) == [16, 64, 256, 1024]
