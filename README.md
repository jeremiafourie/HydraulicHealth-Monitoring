# HydraulicHealth-Monitoring

Predictive‑maintenance for hydraulic systems: estimate component health from sensor data.

**Institution:** Belgium Campus  
**Course:** MLG382 – Machine Learning for Predictive Analytics  
**Group S:**

- Jeremia Fourie
- Juan Oosthuizen
- Busisiwe Radebe
- Phumlani Ntuli  
  **Submission Date:** 24 April 2025, 11:59 PM

## 📚 Project Overview

Use sensor readings from a hydraulic test rig to predict health indices (0–100) for four components (pump, valve, cooler, accumulator). Compare simple regressors (Random Forest) and build a mini dashboard for early‑warning alerts.

## Problem Statement

Hydraulic component failures cause costly downtime.  
**Goal:** Given current cycle’s sensor data, predict each component’s health so maintenance can be scheduled before failure.

## Hypotheses

1. Tree-based classifiers (Random Forest, XGBoost) using cycle-level summary statistics will provide a reliable baseline for classifying pump leakage and accumulator pressure.

2. Addressing class imbalance (e.g., via SMOTE or class weights) will improve macro-F1 scores, especially on under-represented leakage/severe classes.

3. Advanced models (LightGBM, MLP) trained on engineered features (statistics + FFT) will outperform baseline classifiers in both accuracy and macro-F1 across all targets.

## Key EDA Findings

- **Imbalanced Targets**

  - Pump leakage: 55% no leak, 22% weak, 22% severe
  - Accumulator pressure skewed toward 90 bar cycles

- **Summary Statistics Signal**

  - PS1 mean vs. cooler_pct: r ≈ 0.45
  - FS2 std vs. acc_pressure: r ≈ 0.50

- **Distinct Time-Series Patterns**

  - Severe-leak cycles show an early PS1 spike (~5 s) not seen in no-leak cycles

- **Frequency-Domain Shifts**

  - EPS1 FFT peak shifts from ~1.2 Hz (healthy) to ~1.5 Hz (worn valves)

- **Division of work:**
  - _Jeremia Fourie:_ Baseline modeling and hyperparameter tuning and input form for dash app
  - _Juan Oosthuizen:_ Exploratory Data Analysis, visualizations and feature engineering
  - _Busisiwe Radebe:_ Data loading, cleaning, and report writing
  - _Phumlani Ntuli:_ dash app prediction modal callbacks and deployment

Feel free to open issues or submit pull requests for improvements!
