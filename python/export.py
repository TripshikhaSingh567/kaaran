"""
Export Pipeline for KAARAN.
Gathers all model weights, evaluation artifacts, datasets, and golden parity vectors,
and publishes them to web/public/artifacts/ without any mock flags.
"""

import os
import sys
import json
import shutil
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.sim import HeatedTankSimulator
from python.models import EnsembleModel


def export_all(
    models_dir: str = "models",
    eval_dir: str = "eval_results",
    data_dir: str = "data",
    spec_path: str = "spec/kaaran_spec.yaml",
    target_dir: str = "web/public/artifacts",
):
    os.makedirs(target_dir, exist_ok=True)

    print(f"[KAARAN Export] Exporting all artifacts to '{target_dir}'...")

    # 1. Export spec.json
    with open(spec_path, "r") as f:
        spec_yaml = yaml.safe_load(f)
    spec_json = dict(spec_yaml)
    spec_json["mock"] = False
    with open(os.path.join(target_dir, "spec.json"), "w") as f:
        json.dump(spec_json, f, indent=2)

    # 2. Copy evaluation JSONs
    eval_files = [
        "metrics.json",
        "calibration.json",
        "interventions_eval.json",
        "counterfactual_eval.json",
        "lowdata.json",
        "missing_obs.json",
        "confounding.json",
        "trustmap.json",
        "dowhy.json",
    ]
    for ef in eval_files:
        src = os.path.join(eval_dir, ef)
        dst = os.path.join(target_dir, ef)
        if os.path.exists(src):
            with open(src, "r") as f:
                content = json.load(f)
            content["mock"] = False
            with open(dst, "w") as f:
                json.dump(content, f, indent=2)
            print(f"  Exported {ef}")

    # 3. Copy weights JSONs
    weight_files = [
        "weights_forecast.json",
        "weights_action.json",
        "weights_scm.json",
        "weights_linear.json",
    ]
    for wf in weight_files:
        src = os.path.join(models_dir, wf)
        dst = os.path.join(target_dir, wf)
        if os.path.exists(src):
            with open(src, "r") as f:
                content = json.load(f)
            content["mock"] = False
            with open(dst, "w") as f:
                json.dump(content, f)
            print(f"  Exported {wf}")

    # 4. Export episodes_test.json (5 featured + 5 held-out operator episodes for UI responsiveness)
    featured_path = os.path.join(data_dir, "featured_episodes.json")
    operator_path = os.path.join(data_dir, "test_operator.json")

    ui_episodes = []
    if os.path.exists(featured_path):
        with open(featured_path, "r") as f:
            feat_list = json.load(f)
        for i, ep in enumerate(feat_list[:10]):
            ui_episodes.append({
                "id": i,
                "type": "featured",
                "steps": len(ep["v"]),
                "T": [round(float(x), 2) for x in ep["T"]],
                "L": [round(float(x), 2) for x in ep["L"]],
                "v": [round(float(x), 2) for x in ep["v"]],
                "h": [round(float(x), 3) for x in ep["h"]],
                "U": [round(float(x), 3) for x in ep["U"]],
                "eps_T": [round(float(x), 3) for x in ep["eps_T"]],
                "eps_L": [round(float(x), 3) for x in ep["eps_L"]],
            })

    if os.path.exists(operator_path):
        with open(operator_path, "r") as f:
            op_list = json.load(f)
        for i, ep in enumerate(op_list[:10]):
            ui_episodes.append({
                "id": 100 + i,
                "type": "held_out_operator",
                "steps": len(ep["v"]),
                "T": [round(float(x), 2) for x in ep["T"]],
                "L": [round(float(x), 2) for x in ep["L"]],
                "v": [round(float(x), 2) for x in ep["v"]],
                "h": [round(float(x), 3) for x in ep["h"]],
            })

    with open(os.path.join(target_dir, "episodes_test.json"), "w") as f:
        json.dump({"mock": False, "episodes": ui_episodes}, f, indent=2)
    print("  Exported episodes_test.json")

    # 5. Export golden_parity.json
    # Simulator steps with their U/eps inputs and outputs, and model predictions for 5 inputs
    sim = HeatedTankSimulator(
        level_valve_coeff=spec_yaml["dynamics_constants"]["level_valve_coeff"],
        level_decay_coeff=spec_yaml["dynamics_constants"]["level_decay_coeff"],
        level_noise_scale=spec_yaml["dynamics_constants"]["level_noise_scale"],
        heat_base_rate=spec_yaml["dynamics_constants"]["heat_base_rate"],
        heat_ref_level=spec_yaml["dynamics_constants"]["heat_ref_level"],
        heat_min_level=spec_yaml["dynamics_constants"]["heat_min_level"],
        cool_valve_factor=spec_yaml["dynamics_constants"]["cool_valve_factor"],
        cool_ambient_temp=spec_yaml["dynamics_constants"]["cool_ambient_temp"],
        loss_ambient_factor=spec_yaml["dynamics_constants"]["loss_ambient_factor"],
        loss_ambient_temp=spec_yaml["dynamics_constants"]["loss_ambient_temp"],
        evap_rate=spec_yaml["dynamics_constants"]["evap_rate"],
        evap_threshold_temp=spec_yaml["dynamics_constants"]["evap_threshold_temp"],
        temp_confounder_effect=spec_yaml["dynamics_constants"]["temp_confounder_effect"],
        temp_noise_base=spec_yaml["dynamics_constants"]["temp_noise_base"],
        temp_noise_valve_coeff=spec_yaml["dynamics_constants"]["temp_noise_valve_coeff"],
    )

    with open(os.path.join(models_dir, "weights_scm.json"), "r") as f:
        scm_dict = json.load(f)
    scm_ens = EnsembleModel(scm_dict["in_dim"], scm_dict["num_members"], scm_dict["hidden_dim"])
    scm_ens.load_from_dict(scm_dict)

    test_vectors = [
        {"T": 45.0, "L": 55.0, "v": 50.0, "h": 0.5, "U": 0.2, "eps_T": -0.1, "eps_L": 0.05},
        {"T": 75.0, "L": 30.0, "v": 80.0, "h": 0.0, "U": -0.5, "eps_T": 0.3, "eps_L": -0.2},
        {"T": 32.0, "L": 70.0, "v": 10.0, "h": 1.0, "U": 1.2, "eps_T": -0.4, "eps_L": 0.1},
        {"T": 55.0, "L": 60.0, "v": 45.0, "h": 0.45, "U": 0.0, "eps_T": 0.0, "eps_L": 0.0},
        {"T": 85.0, "L": 85.0, "v": 60.0, "h": 0.8, "U": -0.1, "eps_T": 0.15, "eps_L": -0.05},
    ]

    sim_steps = []
    model_preds = []

    for vec in test_vectors:
        s = (vec["T"], vec["L"])
        a = (vec["v"], vec["h"])
        (T_next, L_next), comp = sim.step(s, a, vec["U"], vec["eps_T"], vec["eps_L"])
        sim_steps.append({
            "input": vec,
            "output": {
                "T_next": float(T_next),
                "L_next": float(L_next),
                "delta_T": float(comp["delta_T"]),
                "delta_L": float(comp["delta_L"]),
            }
        })

        x_arr = np.array([vec["T"], vec["L"], vec["v"], vec["h"]], dtype=np.float32)
        pred = scm_ens.predict_one_step(x_arr)
        model_preds.append({
            "model": "scm_ensemble",
            "input": {"T": vec["T"], "L": vec["L"], "v": vec["v"], "h": vec["h"]},
            "output": {
                "mean_delta_T": float(pred["mean"][0]),
                "mean_delta_L": float(pred["mean"][1]),
                "std_delta_T": float(pred["std"][0]),
                "std_delta_L": float(pred["std"][1]),
                "member_means": [[float(m[0]), float(m[1])] for m in pred["member_means"]],
                "member_sigmas": [[float(s[0]), float(s[1])] for s in pred["member_sigmas"]],
            }
        })

    golden_parity = {
        "mock": False,
        "simulator_steps": sim_steps,
        "model_predictions": model_preds,
    }

    with open(os.path.join(target_dir, "golden_parity.json"), "w") as f:
        json.dump(golden_parity, f, indent=2)
    print("  Exported golden_parity.json")

    print("[KAARAN Export] All real artifacts exported successfully.")


if __name__ == "__main__":
    export_all()
