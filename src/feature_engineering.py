"""
Feature engineering module for RIS dataset.
Adds domain-specific wireless propagation features:
1. Log2_RIS_Elements: Logarithmic scaling of reconfigurable elements (powers of 2: 16 -> 4, 64 -> 6, 256 -> 8, 1024 -> 10).
2. Cascaded_Distance_m: d_AP_RIS + d_RIS_UE (cascaded path length).
3. Product_Distance_m: d_AP_RIS * d_RIS_UE (cascaded multiplicative path loss component).
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers wireless propagation features without introducing target leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw or filtered DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.
    """
    df = df.copy()

    # 1. Logarithmic transform of RIS elements
    # Array gain in dB typically scales with log(N) or 20*log10(N)
    df["Log2_RIS_Elements"] = np.log2(df["RIS_Elements"]).astype(float)

    # 2. Cascaded distance (AP -> RIS -> UE total length)
    df["Cascaded_Distance_m"] = df["Distance_AP_RIS_m"] + df["Distance_RIS_UE_m"]

    # 3. Product distance (AP -> RIS * RIS -> UE path loss geometric parameter)
    df["Product_Distance_m"] = df["Distance_AP_RIS_m"] * df["Distance_RIS_UE_m"]

    return df


def audit_feature_leakage(selected_features: list, candidate_target: str) -> dict:
    """
    Validates that no energy or target-derived columns are included in feature set.
    """
    forbidden_tokens = ["energy", "target", "loss", "gain", "provisional"]
    violations = []
    for feat in selected_features:
        feat_lower = feat.lower()
        if any(token in feat_lower for token in forbidden_tokens):
            violations.append(feat)

    is_clean = len(violations) == 0
    return {
        "candidate_target": candidate_target,
        "selected_features": selected_features,
        "violations": violations,
        "is_clean": is_clean,
        "audit_status": "PASSED" if is_clean else "FAILED"
    }


if __name__ == "__main__":
    from config import FULL_FEATURE_SET, PRIMARY_TARGET, RAW_DATA_PATH
    raw_df = pd.read_csv(RAW_DATA_PATH)
    eng_df = engineer_features(raw_df)
    audit = audit_feature_leakage(FULL_FEATURE_SET, PRIMARY_TARGET)
    print("Feature Engineering Smoke Test:")
    print(f"Shape before: {raw_df.shape}, Shape after: {eng_df.shape}")
    print(f"Engineered columns: {[col for col in eng_df.columns if col not in raw_df.columns]}")
    print("Leakage Audit:", audit)
