# Predictive Maintenance Failure Classification

Binary classification pipeline predicting machine failure on the AI4I 2020 dataset (10,000 records, 3.39% failures).

## Overview
- **Data Preprocessing:** Dropped administrative IDs and failure mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target leakage.
- **Feature Engineering:** Derived temperature gap ($\Delta T$), mechanical power ($P_{\text{mech}}$), and torque-tool wear interaction stress index.
- **Model Evaluation:** Benchmarked classifiers using Stratified 5-Fold Cross-Validation.
- **Threshold Optimization:** Calibrated the decision threshold ($t = 0.354$) using out-of-fold training probabilities.
- **Holdout Test Results (2,000 records):**
  - **Precision:** 94.9% (3 false alarms)
  - **Recall:** 82.4% (56/68 failures caught)
  - **F1-Score:** 0.882

## Project Structure
- `Predictive_Maintenance.ipynb`: Complete analysis, modeling, and evaluation notebook.
- `ai4i2020.csv`: AI4I 2020 predictive maintenance dataset.
- `output_8_0.png`: Confusion matrix, ROC, and PR curves.
- `output_9_0.png`: Feature importance chart.