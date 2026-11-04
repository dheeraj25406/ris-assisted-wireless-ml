"""
Data preprocessing and leakage-safe group splitting module.
Ensures that all 4 RIS configurations for any UE stay in the same split.
Scaler is fitted strictly on the training set.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FULL_FEATURE_SET,
    METRICS_DIR,
    MODELS_DIR,
    PRIMARY_TARGET,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.feature_engineering import audit_feature_leakage, engineer_features


def prepare_group_splits(
    data_path: Path = RAW_DATA_PATH,
    feature_cols: list = FULL_FEATURE_SET,
    target_col: str = PRIMARY_TARGET,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
):
    """
    Performs leakage-free group-based train/val/test splitting by UE_ID,
    fits StandardScaler on train features only, and returns split datasets.
    """
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    # Validate leakage
    audit = audit_feature_leakage(feature_cols, target_col)
    if not audit["is_clean"]:
        raise ValueError(f"Feature leakage detected: {audit['violations']}")

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio  # 0.30
    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df, groups=df["UE_ID"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Second split: Val vs Test (50/50 split of the 30% temp data => 15% each)
    val_rel_ratio = val_ratio / temp_ratio  # 0.50
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_rel_ratio, random_state=random_state)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["UE_ID"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Strict Leakage Verification
    train_ues = set(train_df["UE_ID"].unique())
    val_ues = set(val_df["UE_ID"].unique())
    test_ues = set(test_df["UE_ID"].unique())

    assert len(train_ues.intersection(val_ues)) == 0, "Train-Val UE leakage detected!"
    assert len(train_ues.intersection(test_ues)) == 0, "Train-Test UE leakage detected!"
    assert len(val_ues.intersection(test_ues)) == 0, "Val-Test UE leakage detected!"
    assert len(train_ues) + len(val_ues) + len(test_ues) == df["UE_ID"].nunique(), "UE count mismatch!"

    # Fit Scaler strictly on Train features
    scaler = StandardScaler()
    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    scaler.fit(X_train_raw)

    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    # Persist Preprocessor and Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / "preprocessor.joblib")

    split_metadata = {
        "random_seed": random_state,
        "feature_columns": feature_cols,
        "target_column": target_col,
        "train_ues_count": len(train_ues),
        "val_ues_count": len(val_ues),
        "test_ues_count": len(test_ues),
        "train_rows_count": len(train_df),
        "val_rows_count": len(val_df),
        "test_rows_count": len(test_df),
        "scaler_means": {col: float(m) for col, m in zip(feature_cols, scaler.mean_)},
        "scaler_scales": {col: float(s) for col, s in zip(feature_cols, scaler.scale_)},
        "leakage_audit": audit,
        "train_ue_list": [int(x) for x in sorted(list(train_ues))],
        "val_ue_list": [int(x) for x in sorted(list(val_ues))],
        "test_ue_list": [int(x) for x in sorted(list(test_ues))],
    }

    with open(METRICS_DIR / "split_info.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols,
        "target_name": target_col,
    }


if __name__ == "__main__":
    splits = prepare_group_splits()
    print("Group Split Completed Successfully:")
    print(f"Train: {splits['train_df']['UE_ID'].nunique()} UEs, {len(splits['train_df'])} rows")
    print(f"Val:   {splits['val_df']['UE_ID'].nunique()} UEs, {len(splits['val_df'])} rows")
    print(f"Test:  {splits['test_df']['UE_ID'].nunique()} UEs, {len(splits['test_df'])} rows")
    print(f"Preprocessor saved to {MODELS_DIR / 'preprocessor.joblib'}")
    print(f"Split info saved to {METRICS_DIR / 'split_info.json'}")
