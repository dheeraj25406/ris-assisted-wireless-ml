# ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![Pytest Passing](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)

An end-to-end Machine Learning project investigating the impact of **Reconfigurable Intelligent Surfaces (RIS)** on millimeter-wave (60 GHz) wireless communication channels in an indoor factory scenario.

---

## 📌 Project Overview

This repository provides the complete implementation of a B.Tech Major Project designed to replace computationally expensive 3D ray-tracing simulations with real-time machine learning predictions.

- **Primary Goal:** Predict `Path_Loss_With_RIS_dB_PROVISIONAL` from wireless system geometry and RIS reflecting element counts ($N \in \{16, 64, 256, 1024\}$).
- **Core Models:** Random Forest Regressor, Multi-Layer Perceptron (MLP), and XGBoost Regressor.
- **Leakage Prevention:** Group-aware splitting (`GroupShuffleSplit` on `UE_ID`) guarantees zero spatial data leakage across train (70%), validation (15%), and test (15%) partitions.
- **Top Performance on Unseen Test UEs:**
  - **Random Forest:** MAE = **0.1980 dB** | RMSE = 0.3622 dB | $R^2$ = 0.9740
  - **MLP Neural Network:** MAE = 0.2234 dB | RMSE = **0.3253 dB** | $R^2$ = **0.9790**
  - **XGBoost:** MAE = 0.2877 dB | RMSE = 0.4100 dB | $R^2$ = 0.9667
  - **Baseline (Mean):** MAE = 2.0570 dB | RMSE = 2.2461 dB | $R^2$ = -0.0004

---

## 🗂️ Project Structure

```
RIS_ML_Project/
├── config.py                 # System geometry, feature sets, split ratios, seeds
├── main.py                   # Master pipeline runner CLI
├── requirements.txt          # Pinned Python package dependencies
├── README.md                 # Project documentation and user guide
├── data/
│   └── ris_pathloss_dataset_.csv  # 1,120 rows, 280 UEs, 60 GHz indoor factory
├── src/
│   ├── data_validation.py    # Structural & scientific mathematical consistency
│   ├── exploratory_analysis.py # Generates 15 publication-grade EDA figures
│   ├── feature_engineering.py # Domain features & leakage audit
│   ├── preprocessing.py      # Group-aware splitting & StandardScaler
│   ├── train_random_forest.py # RF training & tuning with GroupKFold
│   ├── train_mlp.py          # TransformedTarget MLP neural network training
│   ├── train_xgboost.py      # XGBoost training & tuning with GroupKFold
│   ├── evaluate_models.py    # Test evaluation, diagnostic plots & comparison
│   ├── candidate_targets_study.py # No-RIS & RIS Gain regression study
│   └── predict.py            # Standalone inference API & configuration sweep
├── app/
│   └── streamlit_app.py      # 4-page interactive Streamlit dashboard
├── results/
│   ├── figures/              # 15 EDA figures + 5 evaluation plots
│   ├── metrics/              # Validation reports & comparison metrics in JSON
│   └── predictions/          # test_predictions.csv
├── reports/
│   ├── project_report.md     # Full academic project report (28+ sections)
│   ├── results_summary.md    # Executive summary & viva cheat-sheet
│   └── presentation_slides.md # Complete viva slide deck script
└── tests/
    └── test_pipeline.py      # 5 automated Pytest integration tests
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
To execute the complete end-to-end pipeline (Data Validation $\to$ EDA $\to$ Group Splitting $\to$ Training $\to$ Test Evaluation $\to$ Candidate Targets Study):
```bash
python3 main.py --stage all
```

Or execute individual pipeline stages:
```bash
python3 main.py --stage validate       # Run structural & scientific validation
python3 main.py --stage eda            # Generate all 15 EDA plots
python3 main.py --stage train          # Train RF, MLP, and XGBoost models
python3 main.py --stage evaluate       # Evaluate models on unseen test UEs
python3 main.py --stage study-targets  # Study secondary targets (No-RIS & Gain)
```

---

## 🔮 Interactive Streamlit Application

Launch the interactive web interface:
```bash
streamlit run app/streamlit_app.py
```
**Features:**
1. **Project Overview:** Wireless architecture, coordinate layout, and scientific provisional notice.
2. **Dataset Explorer:** Filter by RIS element configurations and view all 15 EDA plots.
3. **Model Benchmark:** Performance comparison table and diagnostic plots on unseen test UEs.
4. **Live Channel Predictor:** Enter custom UE coordinates $(X, Y)$ and observe predicted path loss and a 4-configuration element sweep ($16 \to 1024$).

---

## 🧪 Automated Testing

Execute the test suite with Pytest:
```bash
pytest tests/test_pipeline.py -v
```
All 5 unit and integration tests verify:
- Raw dataset dimensions and 0 missing values
- Mathematical derivation of path loss from tap energies
- Zero leakage in GroupShuffleSplit across UEs
- Model artifact loading and physical inference bounds

---

## 📄 Academic Deliverables

- [Comprehensive Project Report](reports/project_report.md)
- [Results Summary & Viva Cheat-Sheet](reports/results_summary.md)
- [Viva Presentation Slides](reports/presentation_slides.md)
