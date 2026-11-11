"""
FastAPI Backend Service for RIS-Assisted mmWave Wireless Communication Research.
Serves High-Fidelity Machine Learning Surrogate Inferences, Geometry Computations,
and Academic Analytics Dashboard.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    AP_COORDINATES,
    FIGURES_DIR,
    METRICS_DIR,
    PRIMARY_TARGET,
    RAW_DATA_PATH,
    RECEIVER_Z_M,
    RIS_COORDINATES,
)
from src.predict import RISPredictor, compare_all_configurations

app = FastAPI(
    title="RIS Wireless Channel Surrogate Engine",
    description="Millimeter-Wave 60 GHz Empirical Ray-Tracing ML Prediction API",
    version="2.0.0",
)

# Enable CORS for local and cloud environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount figures directory for static image serving
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Load and cache predictors
predictors = {
    "random_forest": RISPredictor("random_forest"),
    "mlp": RISPredictor("mlp"),
    "xgboost": RISPredictor("xgboost"),
}

# Cache dataset
raw_df = pd.read_csv(RAW_DATA_PATH)

# Cache metrics
with open(METRICS_DIR / "model_comparison.json") as f:
    model_comparison_data = json.load(f)

with open(METRICS_DIR / "data_validation_report.json") as f:
    validation_report_data = json.load(f)


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "RIS ML Surrogate Engine",
        "models_loaded": list(predictors.keys()),
        "carrier_freq_ghz": 60.0,
    }


@app.get("/api/predict")
def predict_channel(
    rx_x: float = Query(8.5, description="Receiver X coordinate in meters"),
    rx_y: float = Query(22.0, description="Receiver Y coordinate in meters"),
    elements: int = Query(256, description="Reflector elements (16, 64, 256, 1024)"),
    model: str = Query("random_forest", description="Model: random_forest, mlp, xgboost"),
):
    model_key = model.lower()
    if model_key not in predictors:
        model_key = "random_forest"

    predictor = predictors[model_key]
    res = predictor.predict_point(rx_x=rx_x, rx_y=rx_y, ris_elements=elements)

    return {
        "rx_x": rx_x,
        "rx_y": rx_y,
        "rx_z": RECEIVER_Z_M,
        "elements": elements,
        "model": model_key,
        "predicted_path_loss_db": round(float(res["Predicted_Path_Loss_With_RIS_dB"]), 3),
        "distance_ap_ue": round(float(res["Distance_AP_UE_m"]), 2),
        "distance_ris_ue": round(float(res["Distance_RIS_UE_m"]), 2),
        "distance_ap_ris": round(float(res["Distance_AP_RIS_m"]), 2),
        "cascaded_distance": round(float(res["Cascaded_Distance_m"]), 2),
        "product_distance": round(float(res["Product_Distance_m"]), 2),
        "direct_fspl_reference_db": round(float(res["Direct_FSPL_Reference_dB"]), 2),
        "ap_coordinates": list(AP_COORDINATES),
        "ris_coordinates": list(RIS_COORDINATES),
    }


@app.get("/api/sweep")
def aperture_sweep(
    rx_x: float = Query(8.5, description="Receiver X coordinate in meters"),
    rx_y: float = Query(22.0, description="Receiver Y coordinate in meters"),
    model: str = Query("random_forest", description="Model: random_forest, mlp, xgboost"),
):
    model_key = model.lower()
    if model_key not in predictors:
        model_key = "random_forest"

    predictor = predictors[model_key]
    configurations = [16, 64, 256, 1024]
    results = []

    for elem in configurations:
        p = predictor.predict_point(rx_x=rx_x, rx_y=rx_y, ris_elements=elem)
        results.append({
            "elements": elem,
            "predicted_path_loss_db": round(float(p["Predicted_Path_Loss_With_RIS_dB"]), 3),
            "distance_ap_ue": round(float(p["Distance_AP_UE_m"]), 2),
            "distance_ris_ue": round(float(p["Distance_RIS_UE_m"]), 2),
        })

    base_pl = results[0]["predicted_path_loss_db"]
    for r in results:
        r["relative_gain_vs_16_db"] = round(base_pl - r["predicted_path_loss_db"], 4)

    return {
        "rx_x": rx_x,
        "rx_y": rx_y,
        "model": model_key,
        "sweep": results,
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "model_comparison": model_comparison_data,
        "validation_report": {
            "total_rows": validation_report_data["total_rows"],
            "unique_ues": validation_report_data["unique_ue_count"],
            "target_status": validation_report_data["target_status_values"],
            "identity_max_abs_diff_dB": validation_report_data["max_diff_ris_gain_identity_dB"],
            "scientific_classification": validation_report_data["scientific_classification"],
        },
    }


@app.get("/api/dataset-sample")
def get_dataset_sample(
    elements: int = Query(256, description="Reflector elements filter"),
    limit: int = Query(30, description="Max rows to return"),
):
    filtered = raw_df[raw_df["RIS_Elements"] == elements]
    cols = [
        "UE_ID",
        "Receiver_X_m",
        "Receiver_Y_m",
        "Distance_AP_UE_m",
        "Distance_RIS_UE_m",
        "RIS_Elements",
        "Composite_Channel_Tap_Energy",
        "Path_Loss_With_RIS_dB_PROVISIONAL",
        "RIS_Gain_dB_PROVISIONAL",
    ]
    sample = filtered[cols].head(limit).to_dict(orient="records")
    stats = filtered[["Distance_AP_UE_m", "Distance_RIS_UE_m", "Path_Loss_With_RIS_dB_PROVISIONAL", "RIS_Gain_dB_PROVISIONAL"]].describe().round(3).to_dict()

    return {
        "total_matching": len(filtered),
        "limit": limit,
        "records": sample,
        "summary_statistics": stats,
    }


# Serve the polished Single Page Application frontend
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def serve_index():
    index_path = APP_DIR / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend template not found</h1>", status_code=500)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
