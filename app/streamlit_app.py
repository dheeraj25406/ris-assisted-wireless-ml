"""
Interactive Streamlit Web Application:
ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

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


# Streamlit Page Config
st.set_page_config(
    page_title="RIS Wireless Channel ML Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_DIR / "model_comparison.json") as f:
        comp = json.load(f)
    with open(METRICS_DIR / "data_validation_report.json") as f:
        val = json.load(f)
    return comp, val


# Load cached data
raw_df = load_dataset()
comp_metrics, val_report = load_metrics()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to Page:",
    [
        "1. Project Overview & System",
        "2. Dataset & Spatial Explorer",
        "3. Model Evaluation & Benchmark",
        "4. Live RIS Prediction Interface",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Specifications:**
    - **Carrier:** 60 GHz (mmWave)
    - **Scenario:** Indoor Factory Ray-Tracing
    - **AP Site:** (10.0, 20.0, 9.5) m
    - **RIS Site:** (0.0, 30.0, 5.5) m
    - **UE Height:** 1.5 m
    - **RIS Configurations:** 16, 64, 256, 1024
    """
)

# =============================================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================================
if menu == "1. Project Overview & System":
    st.markdown('<div class="main-title">📡 ML-Based Performance Prediction of RIS-Assisted Wireless Systems</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">B.Tech Major Project — Wireless Communication & Applied Machine Learning</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 1. Motivation & Wireless Channel Modeling
        In next-generation mmWave and sub-THz wireless systems (5G-Advanced and 6G), high propagation attenuation and severe blockage pose major hurdles to coverage. **Reconfigurable Intelligent Surfaces (RIS)** address this challenge by deploying planar arrays of sub-wavelength metallic/dielectric elements capable of dynamically steering, focusing, or reflecting incident electromagnetic waves to establish synthetic line-of-sight (NLoS-to-LoS) links.

        Simulating full ray-tracing propagation for dynamic wireless environments is computationally intensive. This project develops and evaluates **machine learning regression models** (Random Forest, Multi-Layer Perceptron, and XGBoost) to accurately predict path loss from system geometry and RIS reflector configurations in sub-millisecond inference time.
        """
    )

    st.markdown(
        """
        <div class="alert-box">
        <strong>⚠️ Scientific Validation & Target Status:</strong><br>
        The target column in this ray-tracing dataset is labeled <code>Path_Loss_With_RIS_dB_PROVISIONAL</code> with target status <code>PROVISIONAL_UNVERIFIED</code>. 
        Rigorous verification proves that these values are strictly derived from ray-tracing channel tap energies via:
        <br><code>Path_Loss_dB = -10 * log10(Composite_Channel_Tap_Energy)</code><br>
        While mathematically consistent and physically meaningful, these represent dataset-derived energy metrics without absolute laboratory antenna calibration.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**A. Direct Channel (AP → UE)**\n\nTraditional direct path loss. Invariant to RIS element tuning for a fixed UE location.")
    with col2:
        st.info("**B. RIS Channel (AP → RIS → UE)**\n\nCascaded reflection channel. Steered by 16, 64, 256, or 1024 phase-shift elements.")
    with col3:
        st.info("**C. Composite Channel**\n\nSuperposition of direct and RIS channels: $h_{\\text{comp}} = h_{\\text{dir}} + h_{\\text{RIS}}$.")

    st.markdown("### Wireless System Topography")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(raw_df["Receiver_X_m"], raw_df["Receiver_Y_m"], c="#94A3B8", s=15, alpha=0.5, label="User Terminals (UE)")
    ax.scatter([10.0], [20.0], c="#DC2626", s=180, marker="^", edgecolors="black", lw=1.5, label="Access Point (AP) [Z=9.5m]")
    ax.scatter([0.0], [30.0], c="#2563EB", s=180, marker="s", edgecolors="black", lw=1.5, label="RIS Reflector [Z=5.5m]")
    ax.set_title("Indoor Factory Scenario Geometry (280 Unique UE Locations)", fontweight="bold")
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close()

# =============================================================================
# PAGE 2: DATASET & SPATIAL EXPLORER
# =============================================================================
elif menu == "2. Dataset & Spatial Explorer":
    st.markdown('<div class="main-title">📊 Dataset Inspection & Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2, col4, col5 = st.columns(4)
    col1.metric("Total Dataset Rows", len(raw_df))
    col2.metric("Unique UE Locations", raw_df["UE_ID"].nunique())
    col4.metric("Operating Carrier", "60.0 GHz")
    col5.metric("Configs per UE", "4 (16, 64, 256, 1024)")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Dataset Table & Filtering", "EDA Distributions", "Correlation Analysis"])

    with tab1:
        st.markdown("#### Filter Dataset by RIS Elements")
        selected_n = st.selectbox("Select RIS Element Count:", [16, 64, 256, 1024], index=2)
        filtered_df = raw_df[raw_df["RIS_Elements"] == selected_n]
        st.dataframe(filtered_df.head(20), use_container_width=True)

        st.markdown("#### Summary Statistics:")
        st.dataframe(
            filtered_df[
                [
                    "Distance_AP_UE_m",
                    "Distance_RIS_UE_m",
                    "Composite_Channel_Tap_Energy",
                    "Path_Loss_With_RIS_dB_PROVISIONAL",
                    "RIS_Gain_dB_PROVISIONAL",
                ]
            ].describe(),
            use_container_width=True,
        )

    with tab2:
        st.markdown("#### Core Feature & Target Distributions")
        eda_choice = st.selectbox(
            "Select Distribution Figure to View:",
            [
                "02_ue_spatial_distribution.png",
                "03_ap_to_ue_distance_distribution.png",
                "05_ris_to_ue_distance_distribution.png",
                "08_composite_channel_tap_energy_distribution.png",
                "10_path_loss_with_ris_distribution.png",
                "11_ris_gain_distribution.png",
                "13_path_loss_vs_distance.png",
                "14_performance_vs_ris_elements.png",
                "15_direct_vs_ris_assisted_comparison.png",
            ],
        )
        st.image(str(FIGURES_DIR / eda_choice), use_container_width=True)

    with tab3:
        st.markdown("#### Feature Correlation Matrix")
        st.image(str(FIGURES_DIR / "12_correlation_heatmap.png"), use_container_width=True)

# =============================================================================
# PAGE 3: MODEL EVALUATION & BENCHMARK
# =============================================================================
elif menu == "3. Model Evaluation & Benchmark":
    st.markdown('<div class="main-title">🏆 Machine Learning Model Comparison & Benchmark</div>', unsafe_allow_html=True)
    st.markdown("Evaluated strictly on **42 unseen UEs (168 rows)** with zero data leakage via GroupShuffleSplit.")

    st.markdown("### Overall Test Performance Summary")
    perf_data = []
    for model_name, metrics in comp_metrics["overall_test_metrics"].items():
        perf_data.append({
            "Model Architecture": model_name,
            "MAE (dB)": metrics["MAE"],
            "RMSE (dB)": metrics["RMSE"],
            "R² Score": metrics["R2"],
        })
    perf_df = pd.DataFrame(perf_data).sort_values("RMSE (dB)")
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.success(f"**Recommended Model:** **Random Forest Regressor** achieves the lowest MAE (0.1980 dB) and exceptional R² (0.9740), closely rivaled by MLP (RMSE 0.3253 dB, R² 0.9790). Both outperform the Baseline by >80% error reduction.")

    st.markdown("---")
    st.markdown("### Diagnostic Plots on Unseen Test Set")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIGURES_DIR / "eval_actual_vs_predicted.png"), caption="Actual vs. Predicted Scatter Plots (Test Set)")
        st.image(str(FIGURES_DIR / "eval_error_by_ris_elements.png"), caption="RMSE Breakdown Across RIS Elements")
    with col2:
        st.image(str(FIGURES_DIR / "eval_residual_distributions.png"), caption="Residual Error Distributions (y_true - y_pred)")
        st.image(str(FIGURES_DIR / "eval_feature_importance_comparison.png"), caption="Feature Importance (RF vs XGBoost)")

# =============================================================================
# PAGE 4: LIVE PREDICTION INTERFACE
# =============================================================================
elif menu == "4. Live RIS Prediction Interface":
    st.markdown('<div class="main-title">🔮 Interactive RIS Wireless Channel Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter UE coordinates to simulate path loss and evaluate RIS performance scaling.")

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        rx_x = st.number_input("Receiver X (m):", min_value=-25.0, max_value=30.0, value=8.5, step=0.5)
    with col_in2:
        rx_y = st.number_input("Receiver Y (m):", min_value=0.0, max_value=40.0, value=22.0, step=0.5)
    with col_in3:
        model_choice = st.selectbox("Select ML Model:", ["random_forest", "mlp", "xgboost"], format_func=lambda x: {"random_forest": "Random Forest (Best MAE)", "mlp": "MLP Neural Network", "xgboost": "XGBoost Regressor"}[x])

    predictor = RISPredictor(model_choice)
    single_res = predictor.predict_point(rx_x=rx_x, rx_y=rx_y, ris_elements=256)

    st.markdown("### Prediction Results for N = 256 Elements")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Path Loss", f"{single_res['Predicted_Path_Loss_With_RIS_dB']:.2f} dB")
    c2.metric("Distance AP → UE", f"{single_res['Distance_AP_UE_m']:.2f} m")
    c3.metric("Distance RIS → UE", f"{single_res['Distance_RIS_UE_m']:.2f} m")
    c4.metric("Free-Space Reference", f"{single_res['Direct_FSPL_Reference_dB']:.2f} dB")

    st.markdown("---")
    st.markdown("### Performance Scaling Across All 4 RIS Configurations (16, 64, 256, 1024)")
    sweep_df = compare_all_configurations(rx_x, rx_y)
    
    col_t1, col_t2 = st.columns([1.2, 1.8])
    with col_t1:
        st.dataframe(sweep_df[["RIS_Elements", "Predicted_PL_With_RIS_dB", "Relative_Improvement_vs_16_dB"]], hide_index=True, use_container_width=True)
    with col_t2:
        fig_sw, ax_sw = plt.subplots(figsize=(6, 3))
        ax_sw.plot([16, 64, 256, 1024], sweep_df["Predicted_PL_With_RIS_dB"], marker="o", color="#2563EB", lw=2)
        ax_sw.set_xscale("log", base=2)
        ax_sw.set_xticks([16, 64, 256, 1024])
        ax_sw.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_sw.set_xlabel("RIS Reflecting Elements (N)")
        ax_sw.set_ylabel("Predicted Path Loss (dB)")
        ax_sw.set_title(f"Path Loss Scaling for UE at ({rx_x}, {rx_y})")
        ax_sw.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_sw)
        plt.close()
