# PA-LSTM
DIGITAL TWINS VIRTUAL HUMAN (COMP5532)Group Assignment:
PA-LSTM: Physics-Informed Acceleration-Aware LSTM  for Surgical Trajectory Latency Compensation
# Surgical Trajectory Latency Compensation

This repository contains the course project implementation for surgical trajectory prediction under communication delay.

Core idea: use deep learning models (LSTM variants / physics-informed designs) to compensate latency and improve tracking accuracy in teleoperation settings.

## Project Structure

This project is organized into two main folders for GitHub usage:

- `ModelResult/`  
  Streamlit demo client + exported artifacts for direct visualization and inference.
  - `app.py`: Streamlit app entry
  - `artifacts.pkl`: processed artifacts and evaluation data
  - `*.pth`: trained weight files
  - local `README.md`: detailed demo run instructions

- `Model/`  
  Model development and experiment notebooks (`.ipynb`), including training/evaluation workflows.

In short:
- `Model` = notebook code (training/experiments)
- `ModelResult` = runnable demo + trained weights

## What This Repo Demonstrates

- RMSE comparison under different communication delays
- 3D trajectory visualization (Ground Truth / Delayed / Kalman / LSTM-based methods)
- Training and validation loss curves
- Ablation study for physics-informed loss



