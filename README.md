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
Use sensor readings from a hydraulic test rig to predict health indices (0–100) for four components (pump, valve, cooler, accumulator). Compare simple regressors (Random Forest) with sequence models (LSTM) and build a mini dashboard for early‑warning alerts.

## Problem Statement
Hydraulic component failures cause costly downtime.  
**Goal:** Given current cycle’s sensor data, predict each component’s health so maintenance can be scheduled before failure.

## Hypotheses
1. Combining pressure, flow, temperature, and power sensors improves accuracy.  
2. Sliding‑window statistics (mean, std, slope over recent cycles) outperform single‑cycle inputs.  
3. An LSTM sequence model reduces error versus tabular methods.
