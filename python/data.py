"""
Dataset generation and confounding calibration for KAARAN.
Generates:
- 500 operator episodes (train)
- 150 probe episodes (train)
- 100 operator episodes (held-out test)
- 30 featured test episodes (60 steps) stored with exact (U, eps_T, eps_L) arrays
"""

import os
import sys
import json
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.sim import HeatedTankSimulator, OperatorPolicy, ProbePolicy


def generate_episode(
    sim: HeatedTankSimulator,
    policy_type: str,
    steps: int,
    rng: np.random.Generator,
    operator_policy: OperatorPolicy,
    probe_policy: ProbePolicy,
    store_latents: bool = False,
    initial_state: Tuple[float, float] = None,
) -> Dict[str, Any]:
    if initial_state is None:
        T0 = float(rng.uniform(30.0, 60.0))
        L0 = float(rng.uniform(30.0, 80.0))
    else:
        T0, L0 = float(initial_state[0]), float(initial_state[1])

    s = (T0, L0)
    if policy_type == "probe":
        probe_policy.reset(rng)

    T_hist = [T0]
    L_hist = [L0]
    v_hist = []
    h_hist = []
    U_hist = []
    eps_T_hist = []
    eps_L_hist = []

    for t in range(steps):
        U = float(rng.normal(0.0, 1.0))
        eps_T = float(rng.normal(0.0, 1.0))
        eps_L = float(rng.normal(0.0, 1.0))

        if policy_type == "operator":
            v, h = operator_policy.act(s, U, rng)
        elif policy_type == "probe":
            v, h = probe_policy.act(s, rng)
        else:
            raise ValueError(f"Unknown policy type {policy_type}")

        (T_next, L_next), _ = sim.step(s, (v, h), U=U, eps_T=eps_T, eps_L=eps_L)

        v_hist.append(v)
        h_hist.append(h)
        if store_latents:
            U_hist.append(U)
            eps_T_hist.append(eps_T)
            eps_L_hist.append(eps_L)

        s = (T_next, L_next)
        T_hist.append(T_next)
        L_hist.append(L_next)

    episode = {
        "steps": steps,
        "policy": policy_type,
        "T": T_hist,
        "L": L_hist,
        "v": v_hist,
        "h": h_hist,
    }
    if store_latents:
        episode["U"] = U_hist
        episode["eps_T"] = eps_T_hist
        episode["eps_L"] = eps_L_hist

    return episode


def calibrate_confounding(sim: HeatedTankSimulator, spec: Dict[str, Any], rng: np.random.Generator):
    """
    Calibrate operator policy constants (up to 50% tuning) until:
    (a) Plain linear regression of delta_T on h from operator logs has slope < 30% of true effect (or negative).
    (b) Logs-only model and true simulator disagree by at least 3 °C after 10 steps under do(h = 0).
    """
    op_cfg = spec["operator_policy"]
    
    # Try current constants first, or tune h_U if needed
    best_params = dict(op_cfg)
    
    for multiplier in [1.0, 1.2, 1.35, 1.5, 0.8]:
        tuned_h_U = op_cfg["h_U"] * multiplier
        op_policy = OperatorPolicy(
            h_base=op_cfg["h_base"],
            h_U=tuned_h_U,
            h_T_coeff=op_cfg["h_T_coeff"],
            h_target_T=op_cfg["h_target_T"],
            h_noise_scale=op_cfg["h_noise_scale"],
            v_base=op_cfg["v_base"],
            v_U=op_cfg["v_U"],
            v_L_coeff=op_cfg["v_L_coeff"],
            v_target_L=op_cfg["v_target_L"],
            v_noise_scale=op_cfg["v_noise_scale"],
        )
        probe_policy = ProbePolicy(
            hold_steps=spec["probe_policy"]["hold_steps"],
            safety_interlock_T=spec["probe_policy"]["safety_interlock_T"],
        )

        # Generate sample operator logs
        test_rng = np.random.default_rng(42)
        h_vals = []
        delta_T_vals = []
        for _ in range(100):
            ep = generate_episode(sim, "operator", 120, test_rng, op_policy, probe_policy)
            for t in range(len(ep["h"])):
                h_vals.append(ep["h"][t])
                delta_T_vals.append(ep["T"][t+1] - ep["T"][t])

        h_arr = np.array(h_vals)
        dT_arr = np.array(delta_T_vals)

        # Plain linear regression slope: cov(h, dT) / var(h)
        slope_h = float(np.cov(h_arr, dT_arr)[0, 1] / (np.var(h_arr) + 1e-8))

        # True causal effect (finite difference in noise-free simulator at nominal state)
        T_nom, L_nom, v_nom = 50.0, 50.0, 50.0
        (T_h1, _), _ = sim.step((T_nom, L_nom), (v_nom, 1.0), U=0.0, eps_T=0.0, eps_L=0.0)
        (T_h0, _), _ = sim.step((T_nom, L_nom), (v_nom, 0.0), U=0.0, eps_T=0.0, eps_L=0.0)
        true_causal_effect = float(T_h1 - T_h0)

        check_a = (slope_h < 0.0) or (slope_h < 0.30 * true_causal_effect)

        # Check (b): 10 steps under do(h=0) from nominal state
        # True simulator 10-step mean rollout under do(h=0, v=50)
        true_T10_list = []
        for _ in range(200):
            s = (50.0, 50.0)
            for _ in range(10):
                s, _ = sim.step(s, (50.0, 0.0), U=float(test_rng.normal(0, 1)), eps_T=float(test_rng.normal(0, 1)), eps_L=float(test_rng.normal(0, 1)))
            true_T10_list.append(s[0])
        true_T10_mean = float(np.mean(true_T10_list))

        # Observational prediction after 10 steps:
        # In observational logs, when h=0, what was delta_T?
        # Using linear fit: delta_T = intercept + slope_h * h + ...
        # Over 10 steps under h=0, predicted drop is ~ 10 * intercept.
        intercept = float(np.mean(dT_arr) - slope_h * np.mean(h_arr))
        pred_T10 = 50.0 + 10 * (intercept + slope_h * 0.0)

        disagreement = abs(pred_T10 - true_T10_mean)
        check_b = disagreement >= 3.0

        if check_a and check_b:
            best_params["h_U"] = tuned_h_U
            return best_params, {
                "slope_h": slope_h,
                "true_causal_effect": true_causal_effect,
                "disagreement_10step": disagreement,
                "pred_T10": pred_T10,
                "true_T10_mean": true_T10_mean,
                "passed_a": check_a,
                "passed_b": check_b,
                "multiplier": multiplier,
            }

    # Return default if loop finishes
    return best_params, {
        "slope_h": slope_h,
        "true_causal_effect": true_causal_effect,
        "disagreement_10step": disagreement,
        "passed_a": check_a,
        "passed_b": check_b,
    }


def generate_all_data(spec_path: str = "spec/kaaran_spec.yaml", output_dir: str = "data"):
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

    rng = np.random.default_rng(spec["data_collection"]["random_seed"])

    # Calibrate confounding
    calibrated_op_cfg, calib_results = calibrate_confounding(sim, spec, rng)
    print(f"[KAARAN Data] Confounding Calibration Results:")
    print(f"  Tuned h_U: {calibrated_op_cfg['h_U']:.4f}")
    print(f"  Linear regression slope on logs: {calib_results['slope_h']:.4f}")
    print(f"  True causal effect: {calib_results['true_causal_effect']:.4f}")
    print(f"  10-step do(h=0) disagreement: {calib_results['disagreement_10step']:.2f} °C")
    print(f"  Check (a) passed: {calib_results['passed_a']}")
    print(f"  Check (b) passed: {calib_results['passed_b']}")

    op_policy = OperatorPolicy(
        h_base=calibrated_op_cfg["h_base"],
        h_U=calibrated_op_cfg["h_U"],
        h_T_coeff=calibrated_op_cfg["h_T_coeff"],
        h_target_T=calibrated_op_cfg["h_target_T"],
        h_noise_scale=calibrated_op_cfg["h_noise_scale"],
        v_base=calibrated_op_cfg["v_base"],
        v_U=calibrated_op_cfg["v_U"],
        v_L_coeff=calibrated_op_cfg["v_L_coeff"],
        v_target_L=calibrated_op_cfg["v_target_L"],
        v_noise_scale=calibrated_op_cfg["v_noise_scale"],
    )

    probe_policy = ProbePolicy(
        hold_steps=spec["probe_policy"]["hold_steps"],
        safety_interlock_T=spec["probe_policy"]["safety_interlock_T"],
    )

    # 1. Train operator episodes: 500
    n_op_train = spec["data_collection"]["train_operator_episodes"]
    print(f"[KAARAN Data] Generating {n_op_train} training operator episodes...")
    train_operator = [
        generate_episode(sim, "operator", 120, rng, op_policy, probe_policy)
        for _ in range(n_op_train)
    ]

    # 2. Train probe episodes: 150
    n_probe_train = spec["data_collection"]["train_probe_episodes"]
    print(f"[KAARAN Data] Generating {n_probe_train} training probe episodes...")
    train_probe = [
        generate_episode(sim, "probe", 120, rng, op_policy, probe_policy)
        for _ in range(n_probe_train)
    ]

    # 3. Held-out test operator episodes: 100
    n_op_test = spec["data_collection"]["test_operator_episodes"]
    print(f"[KAARAN Data] Generating {n_op_test} held-out test operator episodes...")
    test_operator = [
        generate_episode(sim, "operator", 120, rng, op_policy, probe_policy)
        for _ in range(n_op_test)
    ]

    # 4. Featured test episodes: 30 (60 steps, with latents stored)
    n_featured = spec["data_collection"]["featured_test_episodes"]
    print(f"[KAARAN Data] Generating {n_featured} featured test episodes (with latents)...")
    featured_episodes = [
        generate_episode(sim, "operator", 60, rng, op_policy, probe_policy, store_latents=True)
        for _ in range(n_featured)
    ]

    # Save datasets
    def save_json(path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    save_json(os.path.join(output_dir, "train_operator.json"), train_operator)
    save_json(os.path.join(output_dir, "train_probe.json"), train_probe)
    save_json(os.path.join(output_dir, "test_operator.json"), test_operator)
    save_json(os.path.join(output_dir, "featured_episodes.json"), featured_episodes)
    save_json(os.path.join(output_dir, "confounding_calibration.json"), {
        "calibrated_operator_policy": calibrated_op_cfg,
        "calibration_results": calib_results,
    })

    print(f"[KAARAN Data] Generation complete. Files saved to '{output_dir}/'")
    return calib_results


if __name__ == "__main__":
    generate_all_data()
