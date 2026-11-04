"""
Inference and Prediction Interface for RIS-Assisted Wireless Systems.
Accepts user geometry (UE coordinates) and RIS configuration, calculates propagation
distances from fixed AP and RIS sites, extracts engineered features, and outputs
predicted path loss and estimated RIS gain.
"""

from pathlib import Path
from typing import Dict, List, Union
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AP_COORDINATES,
    FULL_FEATURE_SET,
    MODELS_DIR,
    RECEIVER_Z_M,
    RIS_COORDINATES,
)


class RISPredictor:
    """
    Production-grade predictor for RIS wireless channel performance.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Parameters
        ----------
        model_type : str
            One of 'random_forest', 'mlp', 'xgboost'.
        """
        self.model_type = model_type.lower()
        self.preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

        if self.model_type in ["random_forest", "rf"]:
            self.model = joblib.load(MODELS_DIR / "random_forest" / "rf_model.joblib")
        elif self.model_type in ["mlp", "neural_network"]:
            self.model = joblib.load(MODELS_DIR / "mlp" / "mlp_model.joblib")
        elif self.model_type in ["xgboost", "xgb"]:
            self.model = joblib.load(MODELS_DIR / "xgboost" / "xgb_model.joblib")
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'random_forest', 'mlp', or 'xgboost'.")

    def predict_point(
        self,
        rx_x: float,
        rx_y: float,
        rx_z: float = RECEIVER_Z_M,
        ris_elements: int = 256,
    ) -> Dict[str, Union[float, int, str]]:
        """
        Predicts path loss for a single user location and RIS configuration.
        """
        df_single = pd.DataFrame([{
            "Receiver_X_m": float(rx_x),
            "Receiver_Y_m": float(rx_y),
            "Receiver_Z_m": float(rx_z),
            "RIS_Elements": int(ris_elements),
        }])

        preds_df = self.predict_batch(df_single)
        return preds_df.iloc[0].to_dict()

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Processes batch DataFrame containing at least Receiver_X_m, Receiver_Y_m, and RIS_Elements.
        Computes 3D distances from AP and RIS, features, and outputs predictions.
        """
        df = df_input.copy()

        if "Receiver_Z_m" not in df.columns:
            df["Receiver_Z_m"] = RECEIVER_Z_M

        # Geometry Distance Calculations
        ap_x, ap_y, ap_z = AP_COORDINATES
        ris_x, ris_y, ris_z = RIS_COORDINATES

        df["Distance_AP_UE_m"] = np.sqrt(
            (df["Receiver_X_m"] - ap_x) ** 2 +
            (df["Receiver_Y_m"] - ap_y) ** 2 +
            (df["Receiver_Z_m"] - ap_z) ** 2
        )

        df["Distance_RIS_UE_m"] = np.sqrt(
            (df["Receiver_X_m"] - ris_x) ** 2 +
            (df["Receiver_Y_m"] - ris_y) ** 2 +
            (df["Receiver_Z_m"] - ris_z) ** 2
        )

        # Distance AP to RIS is constant
        dist_ap_ris = np.sqrt(
            (ap_x - ris_x) ** 2 +
            (ap_y - ris_y) ** 2 +
            (ap_z - ris_z) ** 2
        )
        df["Distance_AP_RIS_m"] = dist_ap_ris

        # Engineered Features
        df["Log2_RIS_Elements"] = np.log2(df["RIS_Elements"]).astype(float)
        df["Cascaded_Distance_m"] = df["Distance_AP_RIS_m"] + df["Distance_RIS_UE_m"]
        df["Product_Distance_m"] = df["Distance_AP_RIS_m"] * df["Distance_RIS_UE_m"]

        X_raw = df[FULL_FEATURE_SET].values

        if self.model_type in ["mlp", "neural_network"]:
            X_input = self.preprocessor.transform(X_raw)
        else:
            X_input = X_raw

        predictions = self.model.predict(X_input)
        df["Predicted_Path_Loss_With_RIS_dB"] = np.round(predictions, 4)

        # Free-space / direct baseline path loss reference (approximate for display)
        # PL_fs = 20*log10(d) + 20*log10(f_GHz) + 92.45
        df["Direct_FSPL_Reference_dB"] = np.round(
            20 * np.log10(np.maximum(df["Distance_AP_UE_m"], 0.1)) + 20 * np.log10(60.0) + 32.44, 2
        )

        return df


def compare_all_configurations(rx_x: float, rx_y: float) -> pd.DataFrame:
    """
    Generates predictions across all 4 standard RIS element counts (16, 64, 256, 1024)
    using the best model (Random Forest).
    """
    predictor = RISPredictor(model_type="random_forest")
    rows = []
    for elem in [16, 64, 256, 1024]:
        res = predictor.predict_point(rx_x=rx_x, rx_y=rx_y, ris_elements=elem)
        rows.append({
            "RIS_Elements": elem,
            "Receiver_X_m": rx_x,
            "Receiver_Y_m": rx_y,
            "Distance_AP_UE_m": round(res["Distance_AP_UE_m"], 2),
            "Distance_RIS_UE_m": round(res["Distance_RIS_UE_m"], 2),
            "Predicted_PL_With_RIS_dB": res["Predicted_Path_Loss_With_RIS_dB"],
        })

    comp_df = pd.DataFrame(rows)
    # Estimate incremental gain relative to 16 elements
    base_pl = comp_df.loc[comp_df["RIS_Elements"] == 16, "Predicted_PL_With_RIS_dB"].values[0]
    comp_df["Relative_Improvement_vs_16_dB"] = np.round(base_pl - comp_df["Predicted_PL_With_RIS_dB"], 4)
    return comp_df


if __name__ == "__main__":
    predictor = RISPredictor("random_forest")
    sample_res = predictor.predict_point(rx_x=5.0, rx_y=15.0, ris_elements=256)
    print("Single Point Prediction Test:")
    for k, v in sample_res.items():
        print(f"  {k}: {v}")

    print("\nFour Configuration Comparison for UE at (5.0, 15.0):")
    print(compare_all_configurations(5.0, 15.0))
