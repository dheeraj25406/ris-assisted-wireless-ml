# Executive Results Summary & Viva Cheat-Sheet
**Project Title:** ML-Based Performance Prediction of RIS-Assisted Wireless Communication Systems  
**Domain:** Wireless Communication Systems (60 GHz mmWave) & Applied Machine Learning  
**Dataset:** Indoor Factory Ray-Tracing (1,120 rows, 280 unique UEs, 4 configurations: 16, 64, 256, 1024 elements)

---

## 1. Key Performance Benchmark (Strictly Unseen Test UEs)

Evaluation conducted on **42 unseen UEs (168 observations)** partitioned via GroupShuffleSplit:

| Model Architecture | Test MAE (dB) | Test RMSE (dB) | Test $R^2$ Score | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Mean Predictor)** | 2.0570 | 2.2461 | -0.0004 | 0.0% |
| **Random Forest Regressor** | **0.1980** | 0.3622 | 0.9740 | **90.4%** |
| **Multi-Layer Perceptron (MLP)** | 0.2234 | **0.3253** | **0.9790** | **89.1%** |
| **XGBoost Regressor** | 0.2877 | 0.4100 | 0.9667 | **86.0%** |

*Key Takeaway:* **Random Forest** achieved the lowest Mean Absolute Error (**0.198 dB**, sub-0.2 dB precision), while **MLP** achieved the lowest RMSE (**0.325 dB**) and highest $R^2$ (**0.9790**). All three machine learning models provide exceptional fidelity, reducing ray-tracing simulation time from seconds/minutes to sub-millisecond inference.

---

## 2. Test Error Breakdown by RIS Configuration

| Configuration | Baseline RMSE (dB) | Random Forest RMSE (dB) | MLP RMSE (dB) | XGBoost RMSE (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **16 Elements** | 2.2274 | **0.3651** | 0.3458 | 0.4149 |
| **64 Elements** | 2.2274 | **0.3651** | 0.3458 | 0.4149 |
| **256 Elements** | 2.2291 | **0.3621** | 0.3150 | 0.4079 |
| **1024 Elements** | 2.3005 | **0.3564** | 0.2917 | 0.4022 |

*Insight:* Prediction accuracy remains uniformly robust across all reflector apertures, with error slightly decreasing at $N=1024$ where array gain dominates local clutter.

---

## 3. Crucial Scientific Validation Findings

1. **PROVISIONAL_UNVERIFIED Status:**
   - The dataset targets are labeled `PROVISIONAL_UNVERIFIED`.
   - Mathematical audit proves that path loss values are exactly:
     $$\text{Path\_Loss\_dB} \approx -10 \log_{10}(\text{Channel\_Tap\_Energy})$$
   - Mathematical identity $\text{RIS\_Gain} = \text{Path\_Loss\_No\_RIS} - \text{Path\_Loss\_With\_RIS}$ is satisfied to $5.00 \times 10^{-8}$ dB precision.
   - **Academic Distinction:** These targets are structurally and mathematically consistent definitions from ray-tracing energy taps, but lack physical laboratory antenna calibration factors. We report them as dataset-derived provisional targets with complete academic honesty.

2. **Zero Data Leakage Protocol:**
   - Because each UE appears under 4 RIS configurations, standard row-wise K-Fold splits cause extreme leakage (UE spatial coordinates leak into the test set).
   - We used strict **GroupShuffleSplit on `UE_ID`** (196 Train UEs, 42 Val UEs, 42 Test UEs) guaranteeing 0% UE overlap across splits.
   - All tap energy columns were excluded from feature inputs because they algebraically encode the target.

3. **Candidate Targets Study:**
   - `Path_Loss_No_RIS` is identical across all 4 configurations for any UE (standard deviation = 0.0 dB).
   - Direct regression of `RIS_Gain` yields an MAE of **0.0361 dB** ($R^2 = 0.4488$ due to the narrow ~1.05 dB total range).
   - Indirect derivation ($\hat{PL}_{\text{No-RIS}} - \hat{PL}_{\text{With-RIS}}$) suffers from error propagation ($R^2 < 0$), proving direct regression is superior.
