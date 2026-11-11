"""
Training pipeline for KAARAN ensembles:
1. Forecast Ensemble: input s = [T, L], trained on operator logs
2. Action Ensemble: input [s, a] = [T, L, v, h], trained on operator logs
3. SCM Ensemble: input [s, a] = [T, L, v, h], trained on probe data only
4. Linear Baseline: scikit-learn LinearRegression on operator logs
5. Low-data SCM models: 5%, 20%, 100% probe data
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.models import EnsembleModel, LinearBaselineModel


def load_transitions_from_episodes(episodes_path: str, in_mode: str = "sa"):
    """
    Extracts transitions from episodes JSON.
    in_mode: "s" for [T, L], "sa" for [T, L, v, h]
    """
    with open(episodes_path, "r") as f:
        episodes = json.load(f)

    X_list = []
    y_list = []

    for ep in episodes:
        T = ep["T"]
        L = ep["L"]
        v = ep["v"]
        h = ep["h"]
        H = len(v)

        for t in range(H):
            dT = T[t + 1] - T[t]
            dL = L[t + 1] - L[t]

            if in_mode == "s":
                X_list.append([T[t], L[t]])
            elif in_mode == "sa":
                X_list.append([T[t], L[t], v[t], h[t]])
            else:
                raise ValueError(f"Unknown in_mode: {in_mode}")

            y_list.append([dT, dL])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_all_models(data_dir: str = "data", models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)

    print("[KAARAN Train] Loading datasets...")
    op_train_path = os.path.join(data_dir, "train_operator.json")
    probe_train_path = os.path.join(data_dir, "train_probe.json")

    # Transitions for Forecast (s -> delta) from operator logs
    X_fc_op, y_fc_op = load_transitions_from_episodes(op_train_path, in_mode="s")
    # Transitions for Action (sa -> delta) from operator logs
    X_act_op, y_act_op = load_transitions_from_episodes(op_train_path, in_mode="sa")
    # Transitions for SCM (sa -> delta) from probe data only!
    X_scm_probe, y_scm_probe = load_transitions_from_episodes(probe_train_path, in_mode="sa")

    print(f"  Operator transitions: {len(X_act_op)}")
    print(f"  Probe transitions: {len(X_scm_probe)}")

    # 1. Forecast Ensemble (input dim = 2)
    print("[KAARAN Train] Training Forecast Ensemble (5 members, input s)...")
    forecast_ens = EnsembleModel(in_dim=2, num_members=5, hidden_dim=64)
    forecast_ens.fit(X_fc_op, y_fc_op, epochs=30, batch_size=128, lr=1e-3, seed=101)

    # 2. Action Ensemble (input dim = 4, operator logs)
    print("[KAARAN Train] Training Action Ensemble (5 members, input s, a from logs)...")
    action_ens = EnsembleModel(in_dim=4, num_members=5, hidden_dim=64)
    action_ens.fit(X_act_op, y_act_op, epochs=30, batch_size=128, lr=1e-3, seed=202)

    # 3. SCM Ensemble (input dim = 4, probe data only)
    print("[KAARAN Train] Training SCM Ensemble (5 members, input s, a from PROBE data)...")
    scm_ens = EnsembleModel(in_dim=4, num_members=5, hidden_dim=64)
    scm_ens.fit(X_scm_probe, y_scm_probe, epochs=35, batch_size=64, lr=1e-3, seed=303)

    # 4. Linear Baseline Model (operator logs)
    print("[KAARAN Train] Fitting Linear Baseline Model...")
    linear_base = LinearBaselineModel()
    linear_base.fit(X_act_op, y_act_op)

    # Save models to JSON
    with open(os.path.join(models_dir, "weights_forecast.json"), "w") as f:
        json.dump(forecast_ens.to_dict(), f)

    with open(os.path.join(models_dir, "weights_action.json"), "w") as f:
        json.dump(action_ens.to_dict(), f)

    with open(os.path.join(models_dir, "weights_scm.json"), "w") as f:
        json.dump(scm_ens.to_dict(), f)

    with open(os.path.join(models_dir, "weights_linear.json"), "w") as f:
        json.dump(linear_base.to_dict(), f)

    # 5. Low-data SCM models: 5% and 20%
    print("[KAARAN Train] Training Low-data SCM models (5% and 20% probe data)...")
    with open(probe_train_path, "r") as f:
        probe_episodes = json.load(f)

    for frac in [0.05, 0.20]:
        n_sub = max(1, int(len(probe_episodes) * frac))
        sub_episodes = probe_episodes[:n_sub]
        sub_path = os.path.join(models_dir, f"sub_probe_{int(frac*100)}.json")
        with open(sub_path, "w") as f:
            json.dump(sub_episodes, f)

        X_sub, y_sub = load_transitions_from_episodes(sub_path, in_mode="sa")
        os.remove(sub_path)

        scm_sub = EnsembleModel(in_dim=4, num_members=5, hidden_dim=64)
        scm_sub.fit(X_sub, y_sub, epochs=35, batch_size=32, lr=1e-3, seed=int(404 + frac * 100))

        with open(os.path.join(models_dir, f"weights_scm_{int(frac*100)}pct.json"), "w") as f:
            json.dump(scm_sub.to_dict(), f)

    print("[KAARAN Train] All models trained and saved successfully.")


if __name__ == "__main__":
    train_all_models()
