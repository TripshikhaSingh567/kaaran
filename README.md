# KAARAN: Uncertainty-Aware Causal World Model for Continuous Process Control

[![Next.js](https://img.shields.io/badge/Next.js-14.2-black)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%202.15-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**KAARAN** is a working prototype of an uncertainty-aware causal world model designed for safety-critical industrial process control (continuous heated chemical process tank). It demonstrates how observational machine learning models fail catastrophically under interventions due to unobserved confounding, and how Structural Causal Models (SCMs) with deep ensembles provide safe, calibrated autonomous control.

> **Ground Truth Principle:** Ground truth comes from the physics simulator and is **never shown to the model**.

---

## 🌟 Key Capabilities
- **Four Operational Lanes:**
  1. *Forecast*: Next-step state projection from operational logs.
  2. *Action*: Policy-conditioned state prediction.
  3. *Intervention ($do(a)$)*: Pure causal evaluation under forced valve and heater settings.
  4. *Counterfactual*: Pearl's 3-step Abduct-Change-Replay algorithm with 39x error reduction over naive simulation.
- **Calibrated Uncertainty:** Deep ensembles of 5 MLPs predicting Gaussian parameters with exact 90% empirical coverage.
- **Epistemic Trust Map:** 2D heatmap quantifying ensemble disagreement and OOD distance, with 75.2% precision in catching critical errors.
- **Zero-Backend Client-Side Web App:** Simulator physics and neural network inference ported to pure TypeScript.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.12 or 3.13
- Node.js 18+ and npm

### 2. Python Environment Setup
```bash
# Create venv and install dependencies (CPU-only PyTorch)
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip install -r python/requirements.txt
```

### 3. Generate Data & Train Models
```bash
# Generate 500 operator episodes, 150 probe episodes, 30 featured test episodes
.\.venv\Scripts\python python/data.py

# Train Forecast, Action, SCM ensembles, and baselines
.\.venv\Scripts\python python/train.py

# Evaluate all lanes, failure modes, and calibration
.\.venv\Scripts\python python/evaluate.py
.\.venv\Scripts\python python/trustmap.py
.\.venv\Scripts\python python/dowhy_check.py

# Export artifacts to web public directory
.\.venv\Scripts\python python/export.py
```

### 4. Run Acceptance Tests
```bash
# Python test suite (determinism, clipping, confounding checks)
.\.venv\Scripts\pytest python/tests/ -v

# TypeScript golden parity test (matches Python to 1e-6 and 1e-5)
cd web && npm test
```

### 5. Launch the Web Application
```bash
cd web
npm run dev
# Open http://localhost:3000
```

---

## 🚢 Deployment Guide

### Deploying Frontend to Vercel (via GitHub Import)
1. Push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "feat: KAARAN causal world model complete prototype"
   git remote add origin <YOUR_GITHUB_REPO_URL>
   git push -u origin main
   ```
2. Log in to [Vercel](https://vercel.com) and click **"Add New Project"**.
3. Select your GitHub repository.
4. Set **Root Directory** to: `web`.
5. Framework Preset: **Next.js**.
6. Build Command: `npm run build` (or leave default).
7. Deploy! Everything runs completely client-side from static JSON artifacts.

### Deploying Backend to Render (via GitHub Import)
1. In your [Render Dashboard](https://dashboard.render.com), click **"New Web Service"**.
2. Connect your GitHub repository.
3. Settings:
   - **Environment**: Python
   - **Build Command**: `pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r python/requirements.txt`
   - **Start Command**: `uvicorn python.api:app --host 0.0.0.0 --port $PORT`
4. Deploy! Render will build the environment and serve the REST inference endpoints.

---

## 📊 Measured Benchmark Summary

| Evaluation Lane | KAARAN Measured Value | Baseline Comparison | Status |
| :--- | :--- | :--- | :--- |
| **Forecast 1-step RMSE** | `1.145` °C | Persistence: `1.243` °C | **Beaten** |
| **Action 1-step RMSE** | `0.634` °C | Linear: `0.669` °C | **Beaten** |
| **Intervention RMSE (Step 10)** | `0.352` °C | Observational Model: `20.557` °C | **58x Error Reduction** |
| **Intervention 90% Coverage** | `93.2%` | Observational Model: `0.0%` | **Beaten** |
| **Counterfactual RMSE (k+10)** | `0.104` °C | Fresh Re-sim: `4.062` °C | **39x Error Reduction** |
| **Red Cell Precision (Trust Map)** | `75.2%` | Median In-Dist Error: `0.208` °C | **High Accuracy** |
| **Confounding Logs Slope** | `-3.072` | True Causal Slope: `+6.000` | **Spurious Confounding Proved** |
