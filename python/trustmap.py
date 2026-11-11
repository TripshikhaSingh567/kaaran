"""
Trust Map generation for KAARAN.
Computes:
1. SCM Ensemble Disagreement: std_m(mu_m(s, a))
2. Standardized Nearest-Neighbor Distance to probe dataset
3. True simulator error on a (T, L) grid for 4 action presets
4. Rank correlation and Precision of Red Cells against true error
"""

import os
import sys
import json
import yaml
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.sim import HeatedTankSimulator
from python.models import EnsembleModel
from python.train import load_transitions_from_episodes


def generate_trust_map(
    data_dir: str = "data",
    models_dir: str = "models",
    output_dir: str = "eval_results",
    spec_path: str = "spec/kaaran_spec.yaml",
):
    os.makedirs(output_dir, exist_ok=True)

    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    sim = HeatedTankSimulator(
        level_valve_coeff=spec["dynamics_constants"]["level_valve_coeff"],
        level_decay_coeff=spec["dynamics_constants"]["level_decay_coeff"],
        level_noise_scale=spec["dynamics_constants"]["level_noise_scale"],
        heat_base_rate=spec["dynamics_constants"]["heat_base_rate"],
        heat_ref_level=spec["dynamics_constants"]["heat_ref_level"],
        heat_min_level=spec["dynamics_constants"]["heat_min_level"],
        cool_valve_factor=spec["dynamics_constants"]["cool_valve_factor"],
        cool_ambient_temp=spec["dynamics_constants"]["cool_ambient_temp"],
        loss_ambient_factor=spec["dynamics_constants"]["loss_ambient_factor"],
        loss_ambient_temp=spec["dynamics_constants"]["loss_ambient_temp"],
        evap_rate=spec["dynamics_constants"]["evap_rate"],
        evap_threshold_temp=spec["dynamics_constants"]["evap_threshold_temp"],
        temp_confounder_effect=spec["dynamics_constants"]["temp_confounder_effect"],
        temp_noise_base=spec["dynamics_constants"]["temp_noise_base"],
        temp_noise_valve_coeff=spec["dynamics_constants"]["temp_noise_valve_coeff"],
    )

    print("[KAARAN TrustMap] Loading SCM Ensemble...")
    with open(os.path.join(models_dir, "weights_scm.json"), "r") as f:
        scm_dict = json.load(f)
    scm_ens = EnsembleModel(scm_dict["in_dim"], scm_dict["num_members"], scm_dict["hidden_dim"])
    scm_ens.load_from_dict(scm_dict)

    # Load probe training transitions for distance calculation
    probe_path = os.path.join(data_dir, "train_probe.json")
    X_probe, _ = load_transitions_from_episodes(probe_path, in_mode="sa")

    # Fit NearestNeighbors on standardized probe data
    probe_mean = np.mean(X_probe, axis=0)
    probe_std = np.std(X_probe, axis=0)
    probe_std = np.where(probe_std < 1e-6, 1.0, probe_std)
    X_probe_norm = (X_probe - probe_mean) / probe_std

    nn_finder = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn_finder.fit(X_probe_norm)

    # 4 Presets
    presets_cfg = {
        "operator_typical": {"name": "Operator Typical", "v": 50.0, "h": 0.55},
        "do_v30": {"name": "do(v = 30)", "v": 30.0, "h": 0.55},
        "do_h0": {"name": "do(h = 0)", "v": 50.0, "h": 0.0},
        "do_h1": {"name": "do(h = 1) [Stress]", "v": 50.0, "h": 1.0},
    }

    # Grid definition: T in [20, 100] (step 5), L in [15, 95] (step 5)
    T_grid = np.arange(20.0, 105.0, 5.0)  # 17 values
    L_grid = np.arange(15.0, 100.0, 5.0)  # 17 values

    rng = np.random.default_rng(888)

    # First pass: compute disagreement, distance, and true error across all presets
    all_cells_info = []

    for preset_key, p_val in presets_cfg.items():
        v_val = p_val["v"]
        h_val = p_val["h"]

        for T in T_grid:
            for L in L_grid:
                x = np.array([T, L, v_val, h_val], dtype=np.float32)
                x_norm = (x - probe_mean) / probe_std

                # SCM prediction
                pred = scm_ens.predict_one_step(x)
                ens_mu_dT = float(pred["mean"][0])
                # Disagreement across 5 members
                member_dTs = [float(m[0]) for m in pred["member_means"]]
                disagreement = float(np.std(member_dTs))

                # Distance to probe data
                dists, _ = nn_finder.kneighbors(x_norm.reshape(1, -1))
                dist = float(np.mean(dists[0]))

                # True simulator error: average of 40 stochastic steps
                true_dTs = []
                for _ in range(40):
                    U = float(rng.normal(0, 1))
                    eps_T = float(rng.normal(0, 1))
                    eps_L = float(rng.normal(0, 1))
                    (T_next, _), comp = sim.step((T, L), (v_val, h_val), U, eps_T, eps_L)
                    true_dTs.append(T_next - T)
                true_mean_dT = float(np.mean(true_dTs))
                true_err = abs(ens_mu_dT - true_mean_dT)

                # Flag in-distribution if within typical operating setpoints
                is_in_dist = (35.0 <= T <= 65.0) and (35.0 <= L <= 75.0) and (dist < 1.5)

                all_cells_info.append({
                    "preset": preset_key,
                    "T": float(T),
                    "L": float(L),
                    "v": v_val,
                    "h": h_val,
                    "disagreement": disagreement,
                    "distance": dist,
                    "true_error": true_err,
                    "is_in_dist": is_in_dist,
                })

    # Calibrate thresholds on in-distribution cells
    in_dist_cells = [c for c in all_cells_info if c["is_in_dist"]]
    in_dist_disagreements = [c["disagreement"] for c in in_dist_cells]
    in_dist_distances = [c["distance"] for c in in_dist_cells]
    in_dist_true_errors = [c["true_error"] for c in in_dist_cells]

    disagreement_amber = float(np.percentile(in_dist_disagreements, 80.0))
    disagreement_red = float(np.percentile(in_dist_disagreements, 95.0))
    distance_amber = float(np.percentile(in_dist_distances, 80.0))
    distance_red = float(np.percentile(in_dist_distances, 95.0))
    in_dist_median_error = float(np.median(in_dist_true_errors))

    # Second pass: assign green / amber / red status
    presets_output = {}
    for p_key, p_val in presets_cfg.items():
        presets_output[p_key] = {
            "name": p_val["name"],
            "action": {"v": p_val["v"], "h": p_val["h"]},
            "grid": [],
        }

    all_disagreements = []
    all_true_errors = []
    red_cells = []

    for cell in all_cells_info:
        disag = cell["disagreement"]
        dist = cell["distance"]
        err = cell["true_error"]

        if (disag >= disagreement_red) or (dist >= distance_red):
            status = "red"
        elif (disag >= disagreement_amber) or (dist >= distance_amber):
            status = "amber"
        else:
            status = "green"

        cell_entry = {
            "T": round(cell["T"], 1),
            "L": round(cell["L"], 1),
            "status": status,
            "disagreement": round(disag, 3),
            "distance": round(dist, 3),
            "true_error": round(err, 3),
        }
        presets_output[cell["preset"]]["grid"].append(cell_entry)

        all_disagreements.append(disag + 0.5 * dist)
        all_true_errors.append(err)
        if status == "red":
            red_cells.append(err)

    # Rank correlation
    spearman_corr, _ = spearmanr(all_disagreements, all_true_errors)

    # Precision of red cells: share of red cells whose true error is > 2 * in-dist median
    err_threshold = 2.0 * in_dist_median_error
    if len(red_cells) > 0:
        red_precision = float(np.mean([1.0 if e > err_threshold else 0.0 for e in red_cells]))
    else:
        red_precision = 1.0

    trustmap_data = {
        "mock": False,
        "thresholds": {
            "disagreement_amber": round(disagreement_amber, 4),
            "disagreement_red": round(disagreement_red, 4),
            "distance_amber": round(distance_amber, 4),
            "distance_red": round(distance_red, 4),
            "in_dist_median_error": round(in_dist_median_error, 4),
            "critical_error_threshold": round(err_threshold, 4),
        },
        "scores": {
            "rank_correlation_spearman": round(float(spearman_corr), 4),
            "precision_red_cells": round(red_precision, 4),
            "in_distribution_median_error": round(in_dist_median_error, 4),
            "total_cells": len(all_cells_info),
            "red_cells_count": len(red_cells),
        },
        "presets": presets_output,
    }

    with open(os.path.join(output_dir, "trustmap.json"), "w") as f:
        json.dump(trustmap_data, f, indent=2)

    print(f"[KAARAN TrustMap] Trust Map Generated:")
    print(f"  Spearman Rank Correlation: {spearman_corr:.4f}")
    print(f"  Precision of Red Cells: {red_precision:.2%}")
    print(f"  In-Distribution Median Error: {in_dist_median_error:.4f} °C")
    return trustmap_data


if __name__ == "__main__":
    generate_trust_map()
