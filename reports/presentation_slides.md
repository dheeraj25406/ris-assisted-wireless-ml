# Viva Presentation Slide Deck Outline
**Project Title:** ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems  
**Degree:** Bachelor of Technology (B.Tech) Major Project  
**Author:** Candidate  

---

## Slide 1: Title & Overview
- **Title:** ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems
- **Domain:** 6G Wireless Technologies & Applied Machine Learning
- **Core Problem:** 60 GHz mmWave propagation suffers extreme path loss, blockage, and absorption. Reconfigurable Intelligent Surfaces (RIS) restore line-of-sight coverage, but full 3D electromagnetic ray-tracing simulations require prohibitive computational time.
- **Solution:** A physics-informed, leakage-free machine learning framework capable of predicting RIS-assisted path loss within 0.20 dB precision in sub-millisecond execution.

---

## Slide 2: Reconfigurable Intelligent Surface (RIS) Mechanics
- **Direct Link ($h_d$):** Fixed AP $(10, 20, 9.5)\text{m} \to \text{UE}$.
- **Assisted Link ($h_{\text{ris}}$):** Cascaded AP $\to$ RIS $(0, 30, 5.5)\text{m} \to \text{UE}$.
- **Composite Superposition:**
  $$h_{\text{composite}} = h_{\text{direct}} + \sum_{n=1}^N \Gamma_n e^{j \phi_n} h_{n,\text{cascaded}}$$
- **Element Configurations Evaluated:** 16, 64, 256, and 1024 reflecting elements.

---

## Slide 3: Dataset Architecture & Scientific Validation
- **Dataset Source:** Indoor factory 60 GHz ray-tracing scenario (`ris_pathloss_dataset_.csv`).
- **Dimensions:** 1,120 rows, 24 columns, 280 unique UE coordinates.
- **Target Verification:**
  - Target labeled `PROVISIONAL_UNVERIFIED`.
  - Rigorous audit confirmed:
    $$\text{Path\_Loss\_dB} = -10 \log_{10}(\text{Channel\_Tap\_Energy})$$
  - Identity satisfied within $5.0 \times 10^{-8}$ dB:
    $$\text{RIS\_Gain} = \text{Path\_Loss\_No\_RIS} - \text{Path\_Loss\_With\_RIS}$$
  - Transparent academic distinction: structural dataset validation vs. physical calibrated measurement.

---

## Slide 4: Data Leakage Prevention Methodology
- **The Pitfall of Random Splitting:** The same UE appears in 4 rows (16, 64, 256, 1024 elements). Random row splits leak UE spatial coordinates $(X, Y)$ into the test set, creating artificially inflated, invalid accuracy.
- **Our Solution:** Group-Aware Splitting by `UE_ID`:
  - **Training:** 196 UEs (784 rows, 70%)
  - **Validation:** 42 UEs (168 rows, 15%)
  - **Testing:** 42 UEs (168 rows, 15%)
  - **Zero Leakage:** Strictly disjoint UE sets. Preprocessor fitted only on training data.
  - **Feature Sanitization:** Channel tap energies excluded from inputs (direct target leak).

---

## Slide 5: Model Architectures & Hyperparameter Tuning
- **Random Forest Regressor:** 200 estimators, max depth 10, min samples leaf 2.
- **Multi-Layer Perceptron (MLP):** (64, 32) hidden layers, ReLU, TransformedTargetRegressor with standardized scaling.
- **XGBoost Regressor:** Gradient boosted trees, depth 3, learning rate 0.03, subsample 1.0.
- All tuned using **5-Fold GroupKFold Cross-Validation** on training UEs.

---

## Slide 6: Benchmark on Unseen Test Set (42 UEs / 168 Rows)
- **Baseline (Mean):** MAE: 2.0570 dB | RMSE: 2.2461 dB | $R^2$: -0.0004
- **Random Forest:** MAE: **0.1980 dB** | RMSE: 0.3622 dB | $R^2$: 0.9740
- **MLP Neural Net:** MAE: 0.2234 dB | RMSE: **0.3253 dB** | $R^2$: **0.9790**
- **XGBoost Regressor:** MAE: 0.2877 dB | RMSE: 0.4100 dB | $R^2$: 0.9667
- **Conclusion:** Random Forest provides the lowest absolute error; MLP delivers top generalized variance explanation.

---

## Slide 7: Feature Importance & Engineering Insights
- **Engineered Features:** $\log_2(N)$, Cascaded Distance ($d_{AP-RIS} + d_{RIS-UE}$), Product Distance ($d_{AP-RIS} \times d_{RIS-UE}$).
- **Feature Importance Hierarchy:**
  1. $d_{AP-UE}$ (AP-to-UE distance dominates free-space propagation) ~65-95%.
  2. $X_{\text{UE}}$ and $Y_{\text{UE}}$ spatial coordinate offsets ~4-30%.
  3. Cascaded/Product distance & RIS Elements ~1-2%.

---

## Slide 8: Interactive Deployment & Streamlit Web App
- Real-time 2D/3D visualization of AP, RIS, and configurable UE terminal.
- Live path loss prediction in < 5 ms.
- 4-configuration element sweep (16 $\to$ 1024) displaying incremental array gains.
- Complete transparency with dataset browser and error diagnostics.

---

## Slide 9: Anticipated Viva Q&A
- **Q1: Why are some RIS gains negative?**
  *A:* Destructive multipath interference between the direct line-of-sight ray and the reflected RIS ray when phase alignment is imperfect.
- **Q2: Why not train on channel tap energy?**
  *A:* Because path loss in dB was derived algebraically as $-10 \log_{10}(E)$. Using $E$ as an input would constitute severe target leakage.
- **Q3: What does PROVISIONAL_UNVERIFIED imply?**
  *A:* The numbers reflect uncalibrated ray-tracing tap energies; our models predict dataset-derived provisional targets with full scientific honesty.
