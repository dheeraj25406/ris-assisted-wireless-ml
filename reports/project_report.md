# ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems

**Major Project Final Report — Bachelor of Technology (B.Tech)**  
**Department of Electronics & Communication Engineering / Computer Science**  

---

## Abstract

Next-generation millimeter-wave (60 GHz) wireless communications suffer from severe free-space attenuation, atmospheric absorption, and susceptibility to blockage. Reconfigurable Intelligent Surfaces (RIS) offer a transformative, energy-efficient solution by steering and focusing incident electromagnetic waves via planar arrays of programmable passive reflecting elements. However, simulating high-fidelity 3D ray-tracing channel characteristics for indoor factory environments is computationally demanding. 

This project investigates an indoor factory 60 GHz ray-tracing dataset containing 1,120 observations across 280 unique User Equipment (UE) locations under four distinct RIS element configurations ($N \in \{16, 64, 256, 1024\}$). First, a rigorous scientific validation establishes that the dataset targets, labeled `PROVISIONAL_UNVERIFIED`, are mathematically consistent definitions derived directly from channel tap energies via $\text{Path\_Loss\_dB} = -10 \log_{10}(E_{\text{composite}})$, satisfying the fundamental identity $\text{RIS\_Gain} = \text{Path\_Loss\_No\_RIS} - \text{Path\_Loss\_With\_RIS}$ within $5.00 \times 10^{-8}$ dB precision. To prevent catastrophic data leakage arising from repeated UE geometry across the four element configurations, a strict group-aware splitting protocol (`GroupShuffleSplit` on `UE_ID`) is enforced: 196 training UEs (784 rows, 70%), 42 validation UEs (168 rows, 15%), and 42 testing UEs (168 rows, 15%). 

Three machine learning regression models—**Random Forest**, **Multi-Layer Perceptron (MLP)**, and **XGBoost**—are trained using 5-fold group cross-validation. When evaluated on the strictly untouched test set of 42 unseen UEs, **Random Forest Regressor** achieves the lowest Mean Absolute Error (**0.1980 dB**, RMSE: 0.3622 dB, $R^2$: 0.9740), while **MLP** achieves the lowest Root Mean Squared Error (**0.3253 dB**, MAE: 0.2234 dB, $R^2$: 0.9790), outperforming the baseline mean predictor by over 90% error reduction. A candidate targets study demonstrates that direct prediction of `RIS_Gain` attains an MAE of 0.0361 dB, avoiding the error propagation inherent in algebraic differencing. Finally, an interactive Streamlit application and a fully automated Pytest test suite are developed, providing an end-to-end framework for real-time channel estimation in smart wireless environments.

---

## 1. Introduction & Motivation

Fifth-generation advanced (5G-Advanced) and upcoming sixth-generation (6G) cellular networks are aggressively expanding into higher carrier frequencies, specifically millimeter-wave (mmWave, 30–300 GHz) and sub-terahertz (sub-THz) bands. Operating at mmWave frequencies, such as 60 GHz, provides multi-gigahertz contiguous bandwidth essential for industrial automation, high-speed automated guided vehicles (AGVs), and ultra-reliable low-latency communications (URLLC).

However, electromagnetic propagation at 60 GHz incurs severe physical limitations:
1. **High Free-Space Path Loss (FSPL):** Governing by Friis' transmission equation, attenuation scales quadratically with frequency:
   $$\text{FSPL} = \left(\frac{4\pi d}{\lambda}\right)^2 = \left(\frac{4\pi d f}{c}\right)^2$$
2. **Severe Blockage:** Wavelengths at 60 GHz ($\lambda = 5\text{ mm}$) exhibit near-optical propagation, where physical obstacles (machinery, walls, humans) cast sharp shadow regions.
3. **High Power Consumption:** Active relays require power-hungry radio frequency (RF) chains, digital signal processors, and power amplifiers that amplify noise and consume significant energy.

To circumvent these obstacles, **Reconfigurable Intelligent Surfaces (RIS)** have emerged as an energy-efficient paradigm. An RIS is an engineered two-dimensional metasurface comprising $N$ passive, sub-wavelength metallic or dielectric elements. Each element incorporates tunable varactor diodes or PIN switches to introduce a controlled phase shift $\phi_n \in [-\pi, \pi]$ and amplitude attenuation $\Gamma_n \in [0, 1]$ to impinging waves. By adaptively configuring these phase shifts, the RIS reconfigures the propagation environment itself, synthesizing virtual line-of-sight (LoS) links between blocked Access Points (APs) and mobile UEs.

Despite these physical advantages, real-time wireless system optimization requires instantaneous knowledge of channel state information (CSI) and path loss. Numerical 3D ray-tracing simulations require hours of computational runtime per scenario, rendering real-time deployment intractable. Machine learning (ML) models trained on ray-tracing datasets offer an alternative: learning non-linear mappings between system geometry, reflector configurations, and composite channel metrics to produce sub-millisecond predictions.

---

## 2. Wireless Channel Propagation Architecture

The wireless communication system investigated in this study consists of an indoor factory environment operating at a carrier frequency of $f_c = 60\text{ GHz}$. The environment incorporates three distinct channel paths:

```
                  +-------------------------+
                  |   Fixed Access Point    |
                  |    (10.0, 20.0, 9.5)m   |
                  +-------------------------+
                         /           \
      Direct Channel    /             \  Incident Link
        (h_direct)     /               \  (AP -> RIS)
                      v                 v
          +---------------+     +-------------------------+
          |  Mobile UE    |     |  RIS Planar Reflector   |
          | (X, Y, 1.5)m  | <-- |    (0.0, 30.0, 5.5)m    |
          +---------------+     +-------------------------+
                     Reflected Link (RIS -> UE)
                         (h_RIS_assisted)
```

### 2.1 The Direct Channel ($h_{\text{direct}}$)
The direct link represents unobstructed propagation from the AP to the UE:
$$h_{\text{direct}} = \sum_{l=1}^{L_d} \alpha_l e^{-j \frac{2\pi}{\lambda} d_{l,\text{dir}}}$$
In this dataset, for any fixed UE position $(X_{\text{UE}}, Y_{\text{UE}}, 1.5)$, the direct channel geometry and tap energy remain invariant to the RIS configuration.

### 2.2 The RIS-Assisted Cascaded Channel ($h_{\text{RIS}}$)
The RIS-assisted path constitutes a cascaded dyadic link (AP $\to$ RIS $\to$ UE):
$$h_{\text{RIS}} = \sum_{n=1}^N h_{\text{AP-RIS}, n} \cdot \Gamma_n e^{j\phi_n} \cdot h_{\text{RIS-UE}, n}$$
where $N$ denotes the number of reflecting elements ($N \in \{16, 64, 256, 1024\}$). The received power of the RIS-assisted path in the far field scales with the cascaded product of distances:
$$P_r^{\text{RIS}} \propto \frac{G_t G_r G_{\text{RIS}}}{(d_{\text{AP-RIS}} \cdot d_{\text{RIS-UE}})^2}$$

### 2.3 The Composite Channel ($h_{\text{composite}}$)
The overall received signal is the complex superposition of the direct and RIS-assisted paths:
$$h_{\text{composite}} = h_{\text{direct}} + h_{\text{RIS}}$$
Because the two paths combine coherently in phase, constructive interference occurs when their phases align, reducing effective path loss. Conversely, when the phase difference is near $\pi$ radians, destructive interference occurs, causing the composite tap energy to drop below the direct tap energy.

---

## 3. Dataset Description & Invariants

The primary dataset used in this project is `ris_pathloss_dataset_.csv`, located in `data/ris_pathloss_dataset_.csv`.

### 3.1 Structural Characteristics
- **Row Count:** Exactly 1,120 rows.
- **Column Count:** Exactly 24 columns.
- **Missing Values:** Zero across all rows and columns.
- **Duplicates:** Zero duplicate rows.
- **Unique Receivers (UEs):** Exactly 280 unique UE coordinates.
- **Configurations per UE:** Exactly 4 rows per UE, corresponding to $N \in \{16, 64, 256, 1024\}$ elements (280 rows per configuration).

### 3.2 Fixed Geometric Invariants
Detailed numerical inspection reveals several invariant system parameters:
- **Carrier Frequency:** Fixed at $60.0\text{ GHz}$ for all 1,120 records.
- **AP Location:** Fixed at $(X, Y, Z) = (10.0, 20.0, 9.5)\text{ m}$.
- **RIS Location:** Fixed at $(X, Y, Z) = (0.0, 30.0, 5.5)\text{ m}$.
- **UE Elevation ($Z$):** Constant at $1.5\text{ m}$ (standard handheld terminal height).
- **AP-to-RIS Distance:** Fixed at $\sqrt{(10-0)^2 + (20-30)^2 + (9.5-5.5)^2} = \sqrt{100 + 100 + 16} = \sqrt{216} \approx 14.6969\text{ m}$.

*Implication:* Naively feeding constant columns into regression models increases matrix singularity risk without adding predictive information. Consequently, constant columns are identified and removed during preprocessing.

---

## 4. Scientific Validation & Target Analysis

A central requirement of rigorous engineering is validating dataset claims and physical realism rather than treating values blindly.

### 4.1 Target Status: `PROVISIONAL_UNVERIFIED`
Every single row in the dataset contains `Target_Status = PROVISIONAL_UNVERIFIED`. The targets must never be falsely presented as hardware-calibrated physical antenna measurements.

### 4.2 Mathematical Derivation from Channel Tap Energies
Investigation of the numeric columns confirms that the provisional path loss targets are mathematically derived from channel tap energies via logarithmic scaling:
$$\text{Path\_Loss\_No\_RIS\_dB} = -10 \log_{10}(\text{Direct\_Channel\_Tap\_Energy})$$
$$\text{RIS\_Component\_Loss\_dB} = -10 \log_{10}(\text{RIS\_Channel\_Tap\_Energy})$$
$$\text{Path\_Loss\_With\_RIS\_dB} = -10 \log_{10}(\text{Composite\_Channel\_Tap\_Energy})$$

The residual differences between the stored provisional dB values and $-10\log_{10}(E)$ are bounded below $0.02\text{ dB}$, fully attributable to 3-significant-figure rounding in the CSV energy representation.

### 4.3 Fundamental Mathematical Identity
The dataset provides `RIS_Gain_dB_PROVISIONAL`, defined as the reduction in path loss achieved by adding the RIS:
$$\text{RIS\_Gain} = \text{Path\_Loss\_No\_RIS} - \text{Path\_Loss\_With\_RIS}$$
Numerical audit confirms this identity holds across all 1,120 rows with a maximum absolute discrepancy of $5.00 \times 10^{-8}\text{ dB}$.

### 4.4 Constructive vs. Destructive Interference
Analysis of `RIS_Gain_dB_PROVISIONAL` reveals:
- **Positive Gain (Constructive Interference):** 650 observations exhibit positive gain (up to $+0.7107\text{ dB}$ at $N=1024$).
- **Negative Gain (Destructive Interference):** 470 observations exhibit negative gain (as low as $-0.3444\text{ dB}$).
- **Physical Interpretation:** Negative gain demonstrates that without active per-element phase alignment optimized for a specific UE position, the reflected multipath ray can arrive out-of-phase with the direct line-of-sight ray, causing partial destructive cancellation.

---

## 5. Exploratory Data Analysis (EDA)

All 15 required diagnostic figures have been generated and saved to `results/figures/`.

1. **RIS Elements Distribution (`01_ris_elements_distribution.png`):** Confirms an exact uniform balance of 280 rows for each configuration (16, 64, 256, 1024).
2. **UE Spatial Distribution (`02_ue_spatial_distribution.png`):** Displays the 2D layout of 280 UEs distributed across the factory floor ($X \in [-18, 28]\text{ m}$, $Y \in [2, 38]\text{ m}$) relative to the AP and RIS.
3. **Distance Distributions (`03_ap_to_ue_distance_distribution.png`, `04_ap_to_ris_distance_distribution.png`, `05_ris_to_ue_distance_distribution.png`):** Verifies that AP-to-RIS distance is single-valued ($14.697\text{ m}$), while AP-to-UE distance ranges from $8.5\text{ m}$ to $34.2\text{ m}$.
4. **Channel Tap Energy Distributions (`06`, `07`, `08`):** All energies are positive and finite ($E_{\text{direct}} \sim 10^{-10}\text{ to } 10^{-8}$, $E_{\text{RIS}} \sim 10^{-15}\text{ to } 10^{-10}$).
5. **Path Loss Distributions (`09`, `10`):** Path loss without RIS ranges from $81.79\text{ dB}$ to $90.46\text{ dB}$ (mean: $84.23\text{ dB}$). Path loss with RIS ranges from $81.59\text{ dB}$ to $90.43\text{ dB}$ (mean: $84.22\text{ dB}$).
6. **RIS Gain Distribution (`11_ris_gain_distribution.png`):** Centered near $0.0\text{ dB}$ with spread from $-0.34\text{ dB}$ to $+0.71\text{ dB}$.
7. **Correlation Heatmap (`12_correlation_heatmap.png`):** Shows strong correlation between $d_{\text{AP-UE}}$ and Path Loss ($r \approx 0.94$).
8. **Path Loss vs. Distance (`13_path_loss_vs_distance.png`):** Shows characteristic logarithmic path loss growth following log-distance path loss laws.
9. **Performance vs. RIS Elements (`14_performance_vs_ris_elements.png`):** Demonstrates that increasing $N$ shifts composite energy upward, expanding maximum achievable gain from $+0.04\text{ dB}$ at $N=16$ to $+0.71\text{ dB}$ at $N=1024$.
10. **Direct vs. RIS Comparison (`15_direct_vs_ris_assisted_comparison.png`):** Scatter analysis contrasting unassisted and assisted channel energy.

---

## 6. Feature Selection & Data Leakage Prevention

### 6.1 Feature Leakage Prevention Protocol
A frequent error in applied machine learning for wireless datasets is including variables that algebraically encode the target. We strictly exclude:
- `Direct_Channel_Tap_Energy`, `RIS_Channel_Tap_Energy`, `Composite_Channel_Tap_Energy` (direct target leak via $-10\log_{10}(E)$).
- `RIS_Component_Loss_dB_PROVISIONAL` (target-derived component).
- `UE_ID`, `Target_Status`, `Source_Scenario` (identifiers/metadata).

### 6.2 Group-Aware Data Splitting (`GroupShuffleSplit`)
Because each UE appears under four rows (one for each RIS configuration), ordinary row-wise random splitting would place three configurations of a UE in the training set and one in the test set. Because the spatial coordinates $(X_{\text{UE}}, Y_{\text{UE}})$ would be memorized by the models, row-wise splitting causes extreme optimistic leakage.

To ensure zero leakage, we enforce **group-based partitioning by `UE_ID`**:
- **Training Set:** 196 UEs (784 rows, 70.0%)
- **Validation Set:** 42 UEs (168 rows, 15.0%)
- **Test Set:** 42 UEs (168 rows, 15.0%)
- **Leakage Verification:** $\text{Train UEs} \cap \text{Val UEs} = \emptyset$, $\text{Train UEs} \cap \text{Test UEs} = \emptyset$, $\text{Val UEs} \cap \text{Test UEs} = \emptyset$.
- **Scaler Fitting:** `StandardScaler` is fitted strictly on the training partition and applied to validation and test partitions.

### 6.3 Final Input Features
The selected feature set includes physical geometry, configuration, and domain-engineered features:
1. `Receiver_X_m`: UE X-coordinate
2. `Receiver_Y_m`: UE Y-coordinate
3. `Distance_AP_UE_m`: Euclidean 3D distance AP-to-UE
4. `Distance_RIS_UE_m`: Euclidean 3D distance RIS-to-UE
5. `RIS_Elements`: Raw element count ($N \in \{16, 64, 256, 1024\}$)
6. `Log2_RIS_Elements`: $\log_2(N)$ (linearizes decibel array gain)
7. `Cascaded_Distance_m`: $d_{\text{AP-RIS}} + d_{\text{RIS-UE}}$
8. `Product_Distance_m`: $d_{\text{AP-RIS}} \cdot d_{\text{RIS-UE}}$

---

## 7. Machine Learning Models & Hyperparameter Tuning

We implement and evaluate three distinct regression paradigms:

### 7.1 Random Forest Regressor
An ensemble of decorrelated decision trees that constructs $B$ bootstrap aggregation trees. 
- **Tuned Hyperparameters:** `n_estimators = 200`, `max_depth = 10`, `min_samples_leaf = 2`, `min_samples_split = 5`.
- **Validation Group-CV RMSE:** $0.6399\text{ dB}$.

### 7.2 Multi-Layer Perceptron (MLP)
A fully-connected feedforward artificial neural network with standardized input and target transformations (`TransformedTargetRegressor`).
- **Architecture:** Two hidden layers of $(64, 32)$ neurons with ReLU activation.
- **Optimizer:** Adam with $L_2$ penalty $\alpha = 0.01$ and initial learning rate $0.001$.
- **Validation Group-CV RMSE:** $0.6258\text{ dB}$.

### 7.3 XGBoost Regressor
An extreme gradient boosted decision tree ensemble optimizing a second-order Taylor expansion of the mean squared error loss function.
- **Tuned Hyperparameters:** `n_estimators = 100`, `max_depth = 3`, `learning_rate = 0.03`, `subsample = 1.0`.
- **Validation Group-CV RMSE:** $0.6388\text{ dB}$.

---

## 8. Experimental Results & Performance Comparison

All models were evaluated on the **identical, untouched test set of 42 unseen UEs (168 rows)**.

### 8.1 Primary Target Evaluation: `Path_Loss_With_RIS_dB_PROVISIONAL`

| Model Architecture | Test MAE (dB) | Test RMSE (dB) | Test $R^2$ Score | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Mean Predictor)** | 2.0570 | 2.2461 | -0.0004 | — |
| **Random Forest Regressor** | **0.1980** | 0.3622 | 0.9740 | **90.4%** |
| **Multi-Layer Perceptron (MLP)** | 0.2234 | **0.3253** | **0.9790** | **89.1%** |
| **XGBoost Regressor** | 0.2877 | 0.4100 | 0.9667 | **86.0%** |

### 8.2 RIS Element-Wise Error Breakdown (Test Set)

| Model | 16 Elements RMSE (dB) | 64 Elements RMSE (dB) | 256 Elements RMSE (dB) | 1024 Elements RMSE (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.3651 | 0.3651 | 0.3621 | 0.3564 |
| **MLP** | 0.3458 | 0.3458 | 0.3150 | **0.2917** |
| **XGBoost** | 0.4149 | 0.4149 | 0.4079 | 0.4022 |

*Analysis:* MLP exhibits superior generalization on large arrays ($N=1024$), achieving an RMSE of $0.2917\text{ dB}$, while Random Forest delivers the most consistent mean absolute precision ($0.1980\text{ dB}$) across all distances.

---

## 9. Secondary Candidate Targets Study

In accordance with project guidelines, we separately analyzed the two secondary targets:

1. **`Path_Loss_No_RIS_dB_PROVISIONAL`:**
   - Numerical analysis proves that standard deviation within any given UE across the four configurations is exactly $0.000000\text{ dB}$. The unassisted channel is completely independent of the RIS.
   - Predicting No-RIS path loss with Random Forest yields MAE: $0.1908\text{ dB}$, RMSE: $0.3594\text{ dB}$, and $R^2: 0.9746$.

2. **`RIS_Gain_dB_PROVISIONAL`:**
   - The total dynamic range of RIS gain across the dataset is narrow ($-0.34\text{ dB}$ to $+0.71\text{ dB}$, total span $\approx 1.05\text{ dB}$).
   - **Direct Regression:** Training Random Forest directly to predict RIS Gain yields an MAE of **$0.0361\text{ dB}$** (sub-0.04 dB accuracy).
   - **Indirect Derivation ($\hat{PL}_{\text{No-RIS}} - \hat{PL}_{\text{With-RIS}}$):** Yields an MAE of $0.0625\text{ dB}$ ($R^2 < 0$), proving that direct modeling is substantially superior due to avoiding the accumulation of variance from differencing two independent estimators.

---

## 10. Software Architecture & Implementation

The project is structured according to modular engineering standards:

```
RIS_ML_Project/
├── config.py                 # Central configurations, coordinates, seeds, paths
├── main.py                   # Complete pipeline CLI runner
├── requirements.txt          # Pinned Python package dependencies
├── data/
│   └── ris_pathloss_dataset_.csv  # Primary 1,120-row dataset
├── src/
│   ├── data_validation.py    # Structural & mathematical consistency checks
│   ├── exploratory_analysis.py # 15 publication-grade EDA figures
│   ├── feature_engineering.py # Domain feature engineering & leakage audit
│   ├── preprocessing.py      # GroupShuffleSplit & train-fitted standardizer
│   ├── train_random_forest.py # RF training & tuning with GroupKFold
│   ├── train_mlp.py          # TransformedTarget MLP training
│   ├── train_xgboost.py      # XGBoost training & tuning with GroupKFold
│   ├── evaluate_models.py    # Unseen test set benchmark & diagnostic plots
│   ├── candidate_targets_study.py # No-RIS & RIS Gain regression study
│   └── predict.py            # Standalone inference API
├── app/
│   └── streamlit_app.py      # Interactive 4-page dashboard
├── results/
│   ├── figures/              # 15 EDA figures + 5 evaluation plots
│   ├── metrics/              # JSON metrics, audits, and split metadata
│   └── predictions/          # test_predictions.csv
└── tests/
    └── test_pipeline.py      # 5 automated Pytest integration tests
```

---

## 11. Viva Defense Preparation (Anticipated Q&A)

**Q1: Why is an ordinary random train/test split invalid for this dataset?**  
*Answer:* Each user location (UE) is measured under four separate RIS element configurations ($16, 64, 256, 1024$). If an ordinary random split is used, three configurations of a UE would appear in the training set and one in the test set. Because the input features include UE coordinates ($X, Y$), decision trees and neural networks easily memorize the coordinate location, resulting in artificial, leaked performance. Using `GroupShuffleSplit` on `UE_ID` guarantees that all four rows for any UE remain together, forcing the model to generalize to entirely unseen physical locations.

**Q2: Why are channel tap energies excluded from the feature inputs?**  
*Answer:* The path loss targets in this dataset were generated by ray-tracing simulations using the exact algebraic relationship $\text{Path\_Loss\_dB} = -10 \log_{10}(E)$. If tap energies were provided as input features, the machine learning models would simply learn the trivial algebraic log transformation rather than learning wireless propagation geometry.

**Q3: Why do some rows exhibit negative RIS gain?**  
*Answer:* In a composite wireless channel, the received signal is the vector sum of the direct ray and the reflected RIS rays: $h_{\text{composite}} = h_{\text{direct}} + h_{\text{RIS}}$. If the reflecting elements are not optimally phase-matched for a particular UE coordinate, the reflected wave arrives with opposite phase ($\approx 180^\circ$ phase shift) relative to the direct path, leading to destructive interference and reducing overall received power.

**Q4: What does the `PROVISIONAL_UNVERIFIED` status signify?**  
*Answer:* It indicates that the path loss metrics are dataset-derived values calculated from raw ray-tracing tap energies without hardware calibration against physical radio frequency testbeds. While mathematically verified and internally consistent, they represent provisional simulation targets.

---

## 12. Conclusion & Future Research Directions

This B.Tech major project successfully established a comprehensive, scientifically validated machine learning pipeline for predicting RIS-assisted wireless propagation at 60 GHz. By combining domain-informed feature engineering with strict leakage prevention, the developed models predict path loss with sub-0.20 dB precision, replacing expensive ray-tracing calculations with sub-millisecond inference.

**Future Directions:**
1. **Dynamic Frequency Generalization:** Extending the pipeline to multi-carrier frequency ranges (28 GHz, 60 GHz, 140 GHz sub-THz).
2. **Phase Optimization Synthesis:** Coupling the regressor with deep reinforcement learning to jointly predict path loss and optimize individual phase-shift profiles ($\phi_n$).
3. **Hardware-in-the-Loop Validation:** Measuring real-world RIS prototypes in an anechoic chamber to calibrate the provisional ray-tracing targets against empirical RF hardware.

---

## References

1. Wu, Q., & Zhang, R. (2019). Towards smart and reconfigurable environment: Intelligent reflecting surface aided wireless network. *IEEE Communications Magazine*, 58(1), 106-112.
2. Di Renzo, M., et al. (2020). Smart radio environments empowered by reconfigurable intelligent surfaces: How it works, state of research, and the road ahead. *IEEE Journal on Selected Areas in Communications*, 38(11), 2450-2525.
3. Rappaport, T. S., et al. (2013). Millimeter wave mobile communications for 5G cellular: It will work!. *IEEE Access*, 1, 335-349.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
