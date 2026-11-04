"""
Global configuration for RIS Machine Learning Project.
All paths, random seeds, features, and model parameters are defined here.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset
RAW_DATA_PATH = DATA_DIR / "ris_pathloss_dataset_.csv"

# Reproducibility
RANDOM_SEED = 42

# Train / Val / Test Group Split Proportions (Grouped by UE_ID)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# System Geometry Constants (Indoor Factory 60 GHz)
AP_COORDINATES = (10.0, 20.0, 9.5)      # (X, Y, Z) in meters
RIS_COORDINATES = (0.0, 30.0, 5.5)      # (X, Y, Z) in meters
RECEIVER_Z_M = 1.5                      # Standard UE height
CARRIER_FREQ_GHZ = 60.0                 # mmWave carrier frequency
SUPPORTED_RIS_ELEMENTS = [16, 64, 256, 1024]

# Metadata & Identifier Columns (Excluded from ML Training)
METADATA_COLUMNS = [
    "UE_ID",
    "Target_Status",
    "Source_Scenario",
]

# Leakage / Energy Columns (Derived/algebraic equivalents of targets)
LEAKAGE_COLUMNS = [
    "Direct_Channel_Tap_Energy",
    "RIS_Channel_Tap_Energy",
    "Composite_Channel_Tap_Energy",
    "RIS_Component_Loss_dB_PROVISIONAL",
]

# Primary Prediction Target
PRIMARY_TARGET = "Path_Loss_With_RIS_dB_PROVISIONAL"

# Secondary Targets for Multi-Target Study
CANDIDATE_TARGETS = [
    "Path_Loss_With_RIS_dB_PROVISIONAL",
    "Path_Loss_No_RIS_dB_PROVISIONAL",
    "RIS_Gain_dB_PROVISIONAL",
]

# Constant columns across dataset (Zero variance, dropped for robust conditioning)
CONSTANT_COLUMNS = [
    "Frequency_GHz",
    "AP_X_m",
    "AP_Y_m",
    "AP_Z_m",
    "RIS_X_m",
    "RIS_Y_m",
    "RIS_Z_m",
    "Receiver_Z_m",
    "Distance_AP_RIS_m",
]

# Final Selected Core Features for Model Input
CORE_FEATURES = [
    "Receiver_X_m",
    "Receiver_Y_m",
    "Distance_AP_UE_m",
    "Distance_RIS_UE_m",
    "RIS_Elements",
]

# Engineered Features
ENGINEERED_FEATURES = [
    "Log2_RIS_Elements",
    "Cascaded_Distance_m",
    "Product_Distance_m",
]

# Combined Feature Set
FULL_FEATURE_SET = CORE_FEATURES + ENGINEERED_FEATURES
