"""
Module: data_validation.py
Performs structural, mathematical, and physical validation on the RIS dataset.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Add parent to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def validate_dataset(csv_path=None):
    if csv_path is None:
        csv_path = config.RAW_DATA_PATH
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    report = {}

    # 1. Structural Checks
    report["total_rows"] = int(len(df))
    report["total_columns"] = int(len(df.columns))
    report["columns"] = df.columns.tolist()
    report["missing_values"] = df.isnull().sum().to_dict()
    report["has_missing_values"] = bool(df.isnull().sum().sum() > 0)
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["unique_ue_count"] = int(df["UE_ID"].nunique())
    
    ris_counts = df["RIS_Elements"].value_counts().to_dict()
    report["ris_element_counts"] = {int(k): int(v) for k, v in ris_counts.items()}
    report["rows_per_ue"] = int(len(df) // df["UE_ID"].nunique())
    report["frequencies"] = [float(x) for x in df["Frequency_GHz"].unique()]

    # 2. Geometry Consistency Checks
    ap_pos = df[["AP_X_m", "AP_Y_m", "AP_Z_m"]].drop_duplicates().values.tolist()
    ris_pos = df[["RIS_X_m", "RIS_Y_m", "RIS_Z_m"]].drop_duplicates().values.tolist()
    report["ap_positions"] = ap_pos
    report["ris_positions"] = ris_pos
    report["receiver_z_range"] = [float(df["Receiver_Z_m"].min()), float(df["Receiver_Z_m"].max())]

    # Computed distances
    d_ap_ue_calc = np.sqrt(
        (df["Receiver_X_m"] - df["AP_X_m"])**2 +
        (df["Receiver_Y_m"] - df["AP_Y_m"])**2 +
        (df["Receiver_Z_m"] - df["AP_Z_m"])**2
    )
    max_d_ap_ue_err = float(np.abs(df["Distance_AP_UE_m"] - d_ap_ue_calc).max())
    report["max_ap_ue_distance_mismatch_m"] = max_d_ap_ue_err

    d_ap_ris_calc = np.sqrt(
        (df["RIS_X_m"] - df["AP_X_m"])**2 +
        (df["RIS_Y_m"] - df["AP_Y_m"])**2 +
        (df["RIS_Z_m"] - df["AP_Z_m"])**2
    )
    max_d_ap_ris_err = float(np.abs(df["Distance_AP_RIS_m"] - d_ap_ris_calc).max())
    report["max_ap_ris_distance_mismatch_m"] = max_d_ap_ris_err

    d_ris_ue_calc = np.sqrt(
        (df["Receiver_X_m"] - df["RIS_X_m"])**2 +
        (df["Receiver_Y_m"] - df["RIS_Y_m"])**2 +
        (df["Receiver_Z_m"] - df["RIS_Z_m"])**2
    )
    max_d_ris_ue_err = float(np.abs(df["Distance_RIS_UE_m"] - d_ris_ue_calc).max())
    report["max_ris_ue_distance_mismatch_m"] = max_d_ris_ue_err

    # 3. Channel Tap Energy Checks
    report["direct_energy_positive_finite"] = bool(
        (df["Direct_Channel_Tap_Energy"] > 0).all() and np.isfinite(df["Direct_Channel_Tap_Energy"]).all()
    )
    report["ris_energy_positive_finite"] = bool(
        (df["RIS_Channel_Tap_Energy"] > 0).all() and np.isfinite(df["RIS_Channel_Tap_Energy"]).all()
    )
    report["composite_energy_positive_finite"] = bool(
        (df["Composite_Channel_Tap_Energy"] > 0).all() and np.isfinite(df["Composite_Channel_Tap_Energy"]).all()
    )

    report["direct_energy_range"] = [float(df["Direct_Channel_Tap_Energy"].min()), float(df["Direct_Channel_Tap_Energy"].max())]
    report["ris_energy_range"] = [float(df["RIS_Channel_Tap_Energy"].min()), float(df["RIS_Channel_Tap_Energy"].max())]
    report["composite_energy_range"] = [float(df["Composite_Channel_Tap_Energy"].min()), float(df["Composite_Channel_Tap_Energy"].max())]

    # 4. Energy to Provisional Path Loss Identity Checks
    pl_no_ris_calc = -10.0 * np.log10(df["Direct_Channel_Tap_Energy"])
    max_pl_no_ris_diff = float(np.abs(df["Path_Loss_No_RIS_dB_PROVISIONAL"] - pl_no_ris_calc).max())
    report["max_diff_pl_no_ris_vs_energy_log_dB"] = max_pl_no_ris_diff

    ris_comp_calc = -10.0 * np.log10(df["RIS_Channel_Tap_Energy"])
    max_ris_comp_diff = float(np.abs(df["RIS_Component_Loss_dB_PROVISIONAL"] - ris_comp_calc).max())
    report["max_diff_ris_comp_loss_vs_energy_log_dB"] = max_ris_comp_diff

    pl_with_ris_calc = -10.0 * np.log10(df["Composite_Channel_Tap_Energy"])
    max_pl_with_ris_diff = float(np.abs(df["Path_Loss_With_RIS_dB_PROVISIONAL"] - pl_with_ris_calc).max())
    report["max_diff_pl_with_ris_vs_energy_log_dB"] = max_pl_with_ris_diff

    # 5. RIS Gain Identity Check: RIS_Gain = Path_Loss_No_RIS - Path_Loss_With_RIS
    gain_identity_calc = df["Path_Loss_No_RIS_dB_PROVISIONAL"] - df["Path_Loss_With_RIS_dB_PROVISIONAL"]
    max_gain_diff = float(np.abs(df["RIS_Gain_dB_PROVISIONAL"] - gain_identity_calc).max())
    report["max_diff_ris_gain_identity_dB"] = max_gain_diff
    report["ris_gain_identity_satisfied"] = bool(max_gain_diff < 1e-6)

    # Destructive vs Constructive interference count
    report["count_negative_ris_gain_destructive"] = int((df["RIS_Gain_dB_PROVISIONAL"] < 0).sum())
    report["count_positive_ris_gain_constructive"] = int((df["RIS_Gain_dB_PROVISIONAL"] > 0).sum())
    report["count_zero_ris_gain"] = int((df["RIS_Gain_dB_PROVISIONAL"] == 0).sum())

    # 6. Physical vs Provisional Classification
    report["target_status_values"] = df["Target_Status"].unique().tolist()
    report["is_provisional_unverified"] = bool("PROVISIONAL_UNVERIFIED" in df["Target_Status"].unique())
    report["scientific_classification"] = (
        "Dataset targets are mathematically consistent definitions derived from ray-tracing channel tap energies "
        "via -10*log10(E). However, targets are officially marked 'PROVISIONAL_UNVERIFIED' without absolute physical "
        "hardware antenna gain calibration or measurement normalization. All ML regression experiments therefore predict "
        "dataset-derived provisional targets with explicit academic transparency."
    )

    # Save to metrics
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = config.METRICS_DIR / "data_validation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Validation report saved to {out_file}")

    return report

if __name__ == "__main__":
    rep = validate_dataset()
    print("Dataset Validation Finished Successfully!")
    print(f"- Rows: {rep['total_rows']}, UEs: {rep['unique_ue_count']}, Missing: {rep['has_missing_values']}")
    print(f"- RIS Gain Identity Max Diff: {rep['max_diff_ris_gain_identity_dB']:.2e} dB (Satisfied: {rep['ris_gain_identity_satisfied']})")
    print(f"- Destructive Gain rows: {rep['count_negative_ris_gain_destructive']}, Constructive: {rep['count_positive_ris_gain_constructive']}")
