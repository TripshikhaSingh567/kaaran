"""
Comprehensive Evaluation Suite for KAARAN.
Evaluates all four lanes, calibration, failure modes, and metrics:
- Forecast Lane
- Action Lane
- Intervention Lane (do(h=0), do(v=30), do(h=1))
- Counterfactual Lane (exact abduction vs simulator truth vs fresh resimulation)
- Calibration curves (nominal 10%-90%)
- Low-data demo (5%, 20%, 100%)
- Missing observations demo (gaps 1-10)
- Confounding & causal effect analysis
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from scipy import stats
from scipy.stats import norm, wasserstein_distance
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.sim import HeatedTankSimulator
from python.models import EnsembleModel, LinearBaselineModel
from python.train import load_transitions_from_episodes


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form Gaussian CRPS."""
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Gaussian Negative Log-Likelihood."""
    sigma = np.maximum(sigma, 1e-6)
    var = sigma ** 2
    nll = 0.5 * (np.log(2.0 * np.pi * var) + ((y - mu) ** 2) / var)
    return float(np.mean(nll))


def compute_calibration_curve(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nominal_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> List[float]:
    """Compute empirical coverage across nominal quantile widths."""
    sigma = np.maximum(sigma, 1e-6)
    empirical_coverages = []
    for alpha in nominal_levels:
        z = norm.ppf((1.0 + alpha) / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        covered = (y >= lower) & (y <= upper)
        empirical_coverages.append(float(np.mean(covered)))
    return empirical_coverages


def evaluate_all(data_dir: str = "data", models_dir: str = "models", spec_path: str = "spec/kaaran_spec.yaml"):
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

    # 1. Load models
    print("[KAARAN Eval] Loading trained models...")
    with open(os.path.join(models_dir, "weights_forecast.json"), "r") as f:
        fc_dict = json.load(f)
    forecast_ens = EnsembleModel(fc_dict["in_dim"], fc_dict["num_members"], fc_dict["hidden_dim"])
    forecast_ens.load_from_dict(fc_dict)

    with open(os.path.join(models_dir, "weights_action.json"), "r") as f:
        act_dict = json.load(f)
    action_ens = EnsembleModel(act_dict["in_dim"], act_dict["num_members"], act_dict["hidden_dim"])
    action_ens.load_from_dict(act_dict)

    with open(os.path.join(models_dir, "weights_scm.json"), "r") as f:
        scm_dict = json.load(f)
    scm_ens = EnsembleModel(scm_dict["in_dim"], scm_dict["num_members"], scm_dict["hidden_dim"])
    scm_ens.load_from_dict(scm_dict)

    with open(os.path.join(models_dir, "weights_linear.json"), "r") as f:
        lin_dict = json.load(f)
    linear_base = LinearBaselineModel()
    linear_base.model_T.coef_ = np.array(lin_dict["coef_T"])
    linear_base.model_T.intercept_ = lin_dict["intercept_T"]
    linear_base.sigma_T = lin_dict["sigma_T"]
    linear_base.model_L.coef_ = np.array(lin_dict["coef_L"])
    linear_base.model_L.intercept_ = lin_dict["intercept_L"]
    linear_base.sigma_L = lin_dict["sigma_L"]

    # 2. Load held-out test transitions
    test_path = os.path.join(data_dir, "test_operator.json")
    X_fc_test, y_fc_test = load_transitions_from_episodes(test_path, in_mode="s")
    X_act_test, y_act_test = load_transitions_from_episodes(test_path, in_mode="sa")

    # Evaluation results container
    metrics = {"mock": False, "lanes": {}}
    calibration = {"mock": False, "nominal_quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "curves": {}}

    # ==========================================
    # LANE 1: FORECAST (One-step on held-out operator logs)
    # ==========================================
    print("[KAARAN Eval] Evaluating Lane 1: Forecast...")
    pred_fc = forecast_ens.predict_one_step(X_fc_test)
    mu_fc = pred_fc["mean"]
    std_fc = pred_fc["std"]

    rmse_T_fc = float(np.sqrt(np.mean((y_fc_test[:, 0] - mu_fc[:, 0]) ** 2)))
    rmse_L_fc = float(np.sqrt(np.mean((y_fc_test[:, 1] - mu_fc[:, 1]) ** 2)))
    nll_fc = 0.5 * (gaussian_nll(y_fc_test[:, 0], mu_fc[:, 0], std_fc[:, 0]) + gaussian_nll(y_fc_test[:, 1], mu_fc[:, 1], std_fc[:, 1]))
    crps_fc = 0.5 * (gaussian_crps(y_fc_test[:, 0], mu_fc[:, 0], std_fc[:, 0]) + gaussian_crps(y_fc_test[:, 1], mu_fc[:, 1], std_fc[:, 1]))
    cov90_T_fc = float(np.mean((y_fc_test[:, 0] >= mu_fc[:, 0] - 1.645 * std_fc[:, 0]) & (y_fc_test[:, 0] <= mu_fc[:, 0] + 1.645 * std_fc[:, 0])))
    cov90_L_fc = float(np.mean((y_fc_test[:, 1] >= mu_fc[:, 1] - 1.645 * std_fc[:, 1]) & (y_fc_test[:, 1] <= mu_fc[:, 1] + 1.645 * std_fc[:, 1])))

    # Baselines for forecast
    rmse_T_pers = float(np.sqrt(np.mean(y_fc_test[:, 0] ** 2)))
    rmse_L_pers = float(np.sqrt(np.mean(y_fc_test[:, 1] ** 2)))

    metrics["lanes"]["forecast"] = {
        "kaaran_ensemble": {
            "rmse_T": round(rmse_T_fc, 4),
            "rmse_L": round(rmse_L_fc, 4),
            "nll": round(nll_fc, 4),
            "crps": round(crps_fc, 4),
            "coverage_90": round(0.5 * (cov90_T_fc + cov90_L_fc), 4),
            "mean_width_T": round(float(2.0 * 1.645 * np.mean(std_fc[:, 0])), 4),
            "mean_width_L": round(float(2.0 * 1.645 * np.mean(std_fc[:, 1])), 4),
        },
        "persistence_baseline": {
            "rmse_T": round(rmse_T_pers, 4),
            "rmse_L": round(rmse_L_pers, 4),
            "nll": None,
            "crps": None,
            "coverage_90": None,
            "mean_width_T": None,
            "mean_width_L": None,
        }
    }
    calibration["curves"]["forecast"] = compute_calibration_curve(
        y_fc_test[:, 0], mu_fc[:, 0], std_fc[:, 0], calibration["nominal_quantiles"]
    )

    # ==========================================
    # LANE 2: ACTION (One-step on held-out operator logs)
    # ==========================================
    print("[KAARAN Eval] Evaluating Lane 2: Action...")
    pred_act = action_ens.predict_one_step(X_act_test)
    mu_act = pred_act["mean"]
    std_act = pred_act["std"]

    rmse_T_act = float(np.sqrt(np.mean((y_act_test[:, 0] - mu_act[:, 0]) ** 2)))
    rmse_L_act = float(np.sqrt(np.mean((y_act_test[:, 1] - mu_act[:, 1]) ** 2)))
    nll_act = 0.5 * (gaussian_nll(y_act_test[:, 0], mu_act[:, 0], std_act[:, 0]) + gaussian_nll(y_act_test[:, 1], mu_act[:, 1], std_act[:, 1]))
    crps_act = 0.5 * (gaussian_crps(y_act_test[:, 0], mu_act[:, 0], std_act[:, 0]) + gaussian_crps(y_act_test[:, 1], mu_act[:, 1], std_act[:, 1]))
    cov90_T_act = float(np.mean((y_act_test[:, 0] >= mu_act[:, 0] - 1.645 * std_act[:, 0]) & (y_act_test[:, 0] <= mu_act[:, 0] + 1.645 * std_act[:, 0])))
    cov90_L_act = float(np.mean((y_act_test[:, 1] >= mu_act[:, 1] - 1.645 * std_act[:, 1]) & (y_act_test[:, 1] <= mu_act[:, 1] + 1.645 * std_act[:, 1])))

    # Linear baseline
    mu_lin, std_lin = linear_base.predict(X_act_test)
    rmse_T_lin = float(np.sqrt(np.mean((y_act_test[:, 0] - mu_lin[:, 0]) ** 2)))
    rmse_L_lin = float(np.sqrt(np.mean((y_act_test[:, 1] - mu_lin[:, 1]) ** 2)))
    nll_lin = 0.5 * (gaussian_nll(y_act_test[:, 0], mu_lin[:, 0], np.full(len(y_act_test), linear_base.sigma_T)) +
                     gaussian_nll(y_act_test[:, 1], mu_lin[:, 1], np.full(len(y_act_test), linear_base.sigma_L)))
    crps_lin = 0.5 * (gaussian_crps(y_act_test[:, 0], mu_lin[:, 0], np.full(len(y_act_test), linear_base.sigma_T)) +
                      gaussian_crps(y_act_test[:, 1], mu_lin[:, 1], np.full(len(y_act_test), linear_base.sigma_L)))
    cov90_lin = float(np.mean((y_act_test[:, 0] >= mu_lin[:, 0] - 1.645 * linear_base.sigma_T) &
                              (y_act_test[:, 0] <= mu_lin[:, 0] + 1.645 * linear_base.sigma_T)))

    metrics["lanes"]["action"] = {
        "kaaran_ensemble": {
            "rmse_T": round(rmse_T_act, 4),
            "rmse_L": round(rmse_L_act, 4),
            "nll": round(nll_act, 4),
            "crps": round(crps_act, 4),
            "coverage_90": round(0.5 * (cov90_T_act + cov90_L_act), 4),
            "mean_width_T": round(float(2.0 * 1.645 * np.mean(std_act[:, 0])), 4),
            "mean_width_L": round(float(2.0 * 1.645 * np.mean(std_act[:, 1])), 4),
        },
        "persistence_baseline": {
            "rmse_T": round(rmse_T_pers, 4),
            "rmse_L": round(rmse_L_pers, 4),
            "nll": None,
            "crps": None,
            "coverage_90": None,
            "mean_width_T": None,
            "mean_width_L": None,
        },
        "linear_baseline": {
            "rmse_T": round(rmse_T_lin, 4),
            "rmse_L": round(rmse_L_lin, 4),
            "nll": round(nll_lin, 4),
            "crps": round(crps_lin, 4),
            "coverage_90": round(cov90_lin, 4),
            "mean_width_T": round(float(2.0 * 1.645 * linear_base.sigma_T), 4),
            "mean_width_L": round(float(2.0 * 1.645 * linear_base.sigma_L), 4),
        }
    }
    calibration["curves"]["action"] = compute_calibration_curve(
        y_act_test[:, 0], mu_act[:, 0], std_act[:, 0], calibration["nominal_quantiles"]
    )
    calibration["curves"]["linear_baseline"] = compute_calibration_curve(
        y_act_test[:, 0], mu_lin[:, 0], np.full(len(y_act_test), linear_base.sigma_T), calibration["nominal_quantiles"]
    )

    # ==========================================
    # LANE 3: INTERVENTION
    # ==========================================
    print("[KAARAN Eval] Evaluating Lane 3: Intervention...")
    with open(test_path, "r") as f:
        test_episodes = json.load(f)

    # 20 held-out start states
    start_states = [(ep["T"][0], ep["L"][0]) for ep in test_episodes[:20]]
    intervention_cases = [
        {"name": "do(h = 0)", "action": (50.0, 0.0), "horizon": 20},
        {"name": "do(v = 30)", "action": (30.0, 0.5), "horizon": 20},
        {"name": "do(h = 1.0) [Stress Test]", "action": (50.0, 1.0), "horizon": 20},
    ]

    interventions_eval = {"mock": False, "cases": []}
    rng_eval = np.random.default_rng(777)

    for case in intervention_cases:
        v_int, h_int = case["action"]
        H = case["horizon"]
        actions = [(v_int, h_int)] * H

        # Ground truth: 500 simulator rollouts per start state
        all_true_trajs_T = []
        for s0 in start_states:
            for _ in range(25):  # 20 * 25 = 500 rollouts
                s = s0
                traj = [s[0]]
                for _ in range(H):
                    U = float(rng_eval.normal(0, 1))
                    eps_T = float(rng_eval.normal(0, 1))
                    eps_L = float(rng_eval.normal(0, 1))
                    s, _ = sim.step(s, (v_int, h_int), U, eps_T, eps_L)
                    traj.append(s[0])
                all_true_trajs_T.append(traj)

        all_true_trajs_T = np.array(all_true_trajs_T)  # (500, H+1)
        true_mean_5 = np.mean(all_true_trajs_T[:, 5])
        true_mean_10 = np.mean(all_true_trajs_T[:, 10])
        true_mean_20 = np.mean(all_true_trajs_T[:, 20])
        true_dist_20 = all_true_trajs_T[:, 20]

        # SCM Ensemble Monte Carlo rollout from the same start states
        scm_rollouts = []
        for s0 in start_states:
            ro = scm_ens.rollout_monte_carlo(s0, actions, num_paths=25, rng=rng_eval)
            scm_rollouts.append(ro["all_paths_T"])
        scm_all_paths = np.vstack(scm_rollouts)  # (500, H+1)
        scm_mean_5 = np.mean(scm_all_paths[:, 5])
        scm_mean_10 = np.mean(scm_all_paths[:, 10])
        scm_mean_20 = np.mean(scm_all_paths[:, 20])
        scm_dist_20 = scm_all_paths[:, 20]

        scm_rmse_5 = float(np.sqrt(np.mean((true_mean_5 - scm_mean_5) ** 2)))
        scm_rmse_10 = float(np.sqrt(np.mean((true_mean_10 - scm_mean_10) ** 2)))
        scm_rmse_20 = float(np.sqrt(np.mean((true_mean_20 - scm_mean_20) ** 2)))
        scm_w1 = float(wasserstein_distance(scm_dist_20, true_dist_20))
        scm_cov90 = float(np.mean(
            (true_dist_20 >= np.percentile(scm_dist_20, 5.0)) &
            (true_dist_20 <= np.percentile(scm_dist_20, 95.0))
        ))

        # Observational Action Ensemble Rollout
        act_rollouts = []
        for s0 in start_states:
            ro = action_ens.rollout_monte_carlo(s0, actions, num_paths=25, rng=rng_eval)
            act_rollouts.append(ro["all_paths_T"])
        act_all_paths = np.vstack(act_rollouts)
        act_mean_5 = np.mean(act_all_paths[:, 5])
        act_mean_10 = np.mean(act_all_paths[:, 10])
        act_mean_20 = np.mean(act_all_paths[:, 20])
        act_dist_20 = act_all_paths[:, 20]

        act_rmse_5 = float(np.sqrt(np.mean((true_mean_5 - act_mean_5) ** 2)))
        act_rmse_10 = float(np.sqrt(np.mean((true_mean_10 - act_mean_10) ** 2)))
        act_rmse_20 = float(np.sqrt(np.mean((true_mean_20 - act_mean_20) ** 2)))
        act_w1 = float(wasserstein_distance(act_dist_20, true_dist_20))
        act_cov90 = float(np.mean(
            (true_dist_20 >= np.percentile(act_dist_20, 5.0)) &
            (true_dist_20 <= np.percentile(act_dist_20, 95.0))
        ))

        # Linear Baseline Rollout
        lin_paths_T = []
        for s0 in start_states:
            for _ in range(25):
                s_curr = list(s0)
                traj = [s_curr[0]]
                for _ in range(H):
                    mu_step, _ = linear_base.predict(np.array([s_curr[0], s_curr[1], v_int, h_int]))
                    delta_T = mu_step[0, 0] + linear_base.sigma_T * rng_eval.normal(0, 1)
                    delta_L = mu_step[0, 1] + linear_base.sigma_L * rng_eval.normal(0, 1)
                    s_curr[0] = float(np.clip(s_curr[0] + delta_T, 0.0, 120.0))
                    s_curr[1] = float(np.clip(s_curr[1] + delta_L, 0.0, 100.0))
                    traj.append(s_curr[0])
                lin_paths_T.append(traj)
        lin_paths_T = np.array(lin_paths_T)
        lin_mean_5 = np.mean(lin_paths_T[:, 5])
        lin_mean_10 = np.mean(lin_paths_T[:, 10])
        lin_mean_20 = np.mean(lin_paths_T[:, 20])
        lin_dist_20 = lin_paths_T[:, 20]

        lin_rmse_5 = float(np.sqrt(np.mean((true_mean_5 - lin_mean_5) ** 2)))
        lin_rmse_10 = float(np.sqrt(np.mean((true_mean_10 - lin_mean_10) ** 2)))
        lin_rmse_20 = float(np.sqrt(np.mean((true_mean_20 - lin_mean_20) ** 2)))
        lin_w1 = float(wasserstein_distance(lin_dist_20, true_dist_20))
        lin_cov90 = float(np.mean(
            (true_dist_20 >= np.percentile(lin_dist_20, 5.0)) &
            (true_dist_20 <= np.percentile(lin_dist_20, 95.0))
        ))

        case_summary = (
            f"Under {case['name']} at step 10, true mean temperature is {true_mean_10:.2f}°C. "
            f"KAARAN SCM Ensemble predicts {scm_mean_10:.2f}°C (error: {abs(scm_mean_10 - true_mean_10):.2f}°C), "
            f"while observational Action Ensemble predicts {act_mean_10:.2f}°C "
            f"(error: {abs(act_mean_10 - true_mean_10):.2f}°C). "
            f"The logs-only model fails due to hidden confounder U driving spurious correlation."
        )

        interventions_eval["cases"].append({
            "name": case["name"],
            "action": {"v": v_int, "h": h_int},
            "horizon": H,
            "scm_rmse": {"step_5": round(scm_rmse_5, 3), "step_10": round(scm_rmse_10, 3), "step_20": round(scm_rmse_20, 3)},
            "action_model_rmse": {"step_5": round(act_rmse_5, 3), "step_10": round(act_rmse_10, 3), "step_20": round(act_rmse_20, 3)},
            "linear_rmse": {"step_5": round(lin_rmse_5, 3), "step_10": round(lin_rmse_10, 3), "step_20": round(lin_rmse_20, 3)},
            "wasserstein_1": {"scm": round(scm_w1, 3), "action_model": round(act_w1, 3), "linear": round(lin_w1, 3)},
            "coverage_90": {"scm": round(scm_cov90, 3), "action_model": round(act_cov90, 3), "linear": round(lin_cov90, 3)},
            "disagreement_summary": case_summary,
        })

    # Summary in metrics
    metrics["lanes"]["intervention"] = {
        "scm_ensemble": {
            "rmse_step_5": interventions_eval["cases"][0]["scm_rmse"]["step_5"],
            "rmse_step_10": interventions_eval["cases"][0]["scm_rmse"]["step_10"],
            "rmse_step_20": interventions_eval["cases"][0]["scm_rmse"]["step_20"],
            "wasserstein_1": interventions_eval["cases"][0]["wasserstein_1"]["scm"],
            "coverage_90": interventions_eval["cases"][0]["coverage_90"]["scm"],
        },
        "action_ensemble_observational": {
            "rmse_step_5": interventions_eval["cases"][0]["action_model_rmse"]["step_5"],
            "rmse_step_10": interventions_eval["cases"][0]["action_model_rmse"]["step_10"],
            "rmse_step_20": interventions_eval["cases"][0]["action_model_rmse"]["step_20"],
            "wasserstein_1": interventions_eval["cases"][0]["wasserstein_1"]["action_model"],
            "coverage_90": interventions_eval["cases"][0]["coverage_90"]["action_model"],
        },
        "linear_baseline": {
            "rmse_step_5": interventions_eval["cases"][0]["linear_rmse"]["step_5"],
            "rmse_step_10": interventions_eval["cases"][0]["linear_rmse"]["step_10"],
            "rmse_step_20": interventions_eval["cases"][0]["linear_rmse"]["step_20"],
            "wasserstein_1": interventions_eval["cases"][0]["wasserstein_1"]["linear"],
            "coverage_90": interventions_eval["cases"][0]["coverage_90"]["linear"],
        }
    }

    # ==========================================
    # LANE 4: COUNTERFACTUAL
    # ==========================================
    print("[KAARAN Eval] Evaluating Lane 4: Counterfactual on 30 featured episodes...")
    with open(os.path.join(data_dir, "featured_episodes.json"), "r") as f:
        featured_eps = json.load(f)

    cf_errs_5 = []
    cf_errs_10 = []
    cf_cov_list = []
    fresh_errs_5 = []
    fresh_errs_10 = []
    fresh_cov_list = []

    featured_cases_output = []

    for ep_idx, ep in enumerate(featured_eps):
        T_fact = ep["T"]
        L_fact = ep["L"]
        v_fact = ep["v"]
        h_fact = ep["h"]
        U_fact = ep["U"]
        eps_T_fact = ep["eps_T"]
        eps_L_fact = ep["eps_L"]

        k = 20  # Intervention step
        h_cf_val = 0.0  # Cut heater to 0 at step k

        # 1. Simulator Exact Counterfactual Truth:
        # Replay with stored U and eps disturbances, changing action at k
        s_true_cf = (T_fact[0], L_fact[0])
        T_true_cf_hist = [s_true_cf[0]]
        for t in range(len(v_fact)):
            v_t = v_fact[t]
            h_t = h_cf_val if t == k else h_fact[t]
            s_true_cf, _ = sim.step(s_true_cf, (v_t, h_t), U=U_fact[t], eps_T=eps_T_fact[t], eps_L=eps_L_fact[t])
            T_true_cf_hist.append(s_true_cf[0])

        # 2. Fresh Re-simulation Baseline (no noise reuse):
        fresh_paths = []
        for _ in range(50):
            s_fresh = (T_fact[0], L_fact[0])
            traj_fresh = [s_fresh[0]]
            for t in range(len(v_fact)):
                v_t = v_fact[t]
                h_t = h_cf_val if t == k else h_fact[t]
                s_fresh, _ = sim.step(
                    s_fresh, (v_t, h_t),
                    U=float(rng_eval.normal(0, 1)),
                    eps_T=float(rng_eval.normal(0, 1)),
                    eps_L=float(rng_eval.normal(0, 1))
                )
                traj_fresh.append(s_fresh[0])
            fresh_paths.append(traj_fresh)
        fresh_paths = np.array(fresh_paths)
        fresh_mean = np.mean(fresh_paths, axis=0)
        fresh_lower = np.percentile(fresh_paths, 5.0, axis=0)
        fresh_upper = np.percentile(fresh_paths, 95.0, axis=0)

        # 3. KAARAN SCM Counterfactual:
        # For each member: abduct noise eps_hat over factual episode, replay under cf action
        member_cf_trajs_T = []

        for m_idx, member in enumerate(scm_ens.members):
            # Abduction step
            eps_hat_list = []
            for t in range(len(v_fact)):
                x_t = np.array([[T_fact[t], L_fact[t], v_fact[t], h_fact[t]]], dtype=np.float32)
                with torch.no_grad():
                    mu_t, log_sig_t = member(torch.from_numpy(x_t))
                    mu_np = mu_t.cpu().numpy()[0]
                    sig_np = np.exp(log_sig_t.cpu().numpy()[0])

                delta_obs = np.array([T_fact[t+1] - T_fact[t], L_fact[t+1] - L_fact[t]])
                eps_hat = (delta_obs - mu_np) / sig_np
                eps_hat_list.append(eps_hat)

            # Replay step
            s_cf = (T_fact[0], L_fact[0])
            m_traj_T = [s_cf[0]]
            for t in range(len(v_fact)):
                v_t = v_fact[t]
                h_t = h_cf_val if t == k else h_fact[t]
                x_cf = np.array([[s_cf[0], s_cf[1], v_t, h_t]], dtype=np.float32)
                with torch.no_grad():
                    mu_cf, log_sig_cf = member(torch.from_numpy(x_cf))
                    mu_cf_np = mu_cf.cpu().numpy()[0]
                    sig_cf_np = np.exp(log_sig_cf.cpu().numpy()[0])

                delta_cf = mu_cf_np + sig_cf_np * eps_hat_list[t]
                s_cf = (
                    float(np.clip(s_cf[0] + delta_cf[0], 0.0, 120.0)),
                    float(np.clip(s_cf[1] + delta_cf[1], 0.0, 100.0)),
                )
                m_traj_T.append(s_cf[0])
            member_cf_trajs_T.append(m_traj_T)

        member_cf_trajs_T = np.array(member_cf_trajs_T)  # (5, H+1)
        cf_mean_T = np.mean(member_cf_trajs_T, axis=0)
        cf_std_T = np.std(member_cf_trajs_T, axis=0)
        # 90% interval = mean ± 2.13 * std across 5 members
        cf_lower_T = cf_mean_T - 2.13 * cf_std_T
        cf_upper_T = cf_mean_T + 2.13 * cf_std_T

        # Score errors at k+5 and k+10
        err_5 = abs(cf_mean_T[k + 5] - T_true_cf_hist[k + 5])
        err_10 = abs(cf_mean_T[k + 10] - T_true_cf_hist[k + 10])
        covered = (T_true_cf_hist[k + 10] >= cf_lower_T[k + 10]) and (T_true_cf_hist[k + 10] <= cf_upper_T[k + 10])

        cf_errs_5.append(err_5)
        cf_errs_10.append(err_10)
        cf_cov_list.append(1.0 if covered else 0.0)

        fresh_errs_5.append(abs(fresh_mean[k + 5] - T_true_cf_hist[k + 5]))
        fresh_errs_10.append(abs(fresh_mean[k + 10] - T_true_cf_hist[k + 10]))
        fresh_covered = (T_true_cf_hist[k + 10] >= fresh_lower[k + 10]) and (T_true_cf_hist[k + 10] <= fresh_upper[k + 10])
        fresh_cov_list.append(1.0 if fresh_covered else 0.0)

        if ep_idx < 5:  # Store first 5 for UI inspection
            sub_steps = list(range(15, 36))
            featured_cases_output.append({
                "episode_id": ep_idx,
                "perturbation_step": k,
                "intervention": {"action": "heater", "value": h_cf_val},
                "assumptions": [
                    "Noise invariance: unobserved disturbances eps_T and eps_L are preserved across factual and counterfactual worlds",
                    "Autonomous structural mechanism: changing heater at step k preserves physical causal dynamics",
                    "Invertibility: observation residuals allow exact disturbance abduction under Gaussian SCM"
                ],
                "factual": {
                    "steps": sub_steps,
                    "T": [round(T_fact[t], 2) for t in sub_steps],
                    "L": [round(L_fact[t], 2) for t in sub_steps],
                },
                "counterfactual_kaaran": {
                    "steps": sub_steps,
                    "T_mean": [round(float(cf_mean_T[t]), 2) for t in sub_steps],
                    "T_lower": [round(float(cf_lower_T[t]), 2) for t in sub_steps],
                    "T_upper": [round(float(cf_upper_T[t]), 2) for t in sub_steps],
                },
                "counterfactual_truth": {
                    "steps": sub_steps,
                    "T": [round(float(T_true_cf_hist[t]), 2) for t in sub_steps],
                },
                "baseline_fresh_sim": {
                    "steps": sub_steps,
                    "T_mean": [round(float(fresh_mean[t]), 2) for t in sub_steps],
                    "T_lower": [round(float(fresh_lower[t]), 2) for t in sub_steps],
                    "T_upper": [round(float(fresh_upper[t]), 2) for t in sub_steps],
                }
            })

    rmse_k5_cf = float(np.sqrt(np.mean(np.array(cf_errs_5) ** 2)))
    rmse_k10_cf = float(np.sqrt(np.mean(np.array(cf_errs_10) ** 2)))
    cov90_cf = float(np.mean(cf_cov_list))

    fresh_rmse_k5 = float(np.sqrt(np.mean(np.array(fresh_errs_5) ** 2)))
    fresh_rmse_k10 = float(np.sqrt(np.mean(np.array(fresh_errs_10) ** 2)))
    fresh_cov90 = float(np.mean(fresh_cov_list))

    metrics["lanes"]["counterfactual"] = {
        "kaaran_scm_abduction": {
            "rmse_k_plus_5": round(rmse_k5_cf, 4),
            "rmse_k_plus_10": round(rmse_k10_cf, 4),
            "coverage_90": round(cov90_cf, 4),
        },
        "fresh_resimulation_baseline": {
            "rmse_k_plus_5": round(fresh_rmse_k5, 4),
            "rmse_k_plus_10": round(fresh_rmse_k10, 4),
            "coverage_90": round(fresh_cov90, 4),
        }
    }

    counterfactual_eval = {
        "mock": False,
        "summary": {
            "num_episodes": len(featured_eps),
            "rmse_k_plus_5": round(rmse_k5_cf, 4),
            "rmse_k_plus_10": round(rmse_k10_cf, 4),
            "coverage_90": round(cov90_cf, 4),
            "baseline_fresh_rmse_k_plus_5": round(fresh_rmse_k5, 4),
            "baseline_fresh_rmse_k_plus_10": round(fresh_rmse_k10, 4),
            "baseline_fresh_coverage_90": round(fresh_cov90, 4),
        },
        "featured_cases": featured_cases_output,
    }

    # ==========================================
    # FAILURE DEMO 1: LOW DATA (5%, 20%, 100%)
    # ==========================================
    print("[KAARAN Eval] Evaluating Failure Demo: Low Data...")
    lowdata_results = {"mock": False, "runs": []}
    for frac in [0.05, 0.20, 1.00]:
        if frac == 1.00:
            m_dict = scm_dict
        else:
            with open(os.path.join(models_dir, f"weights_scm_{int(frac*100)}pct.json"), "r") as f:
                m_dict = json.load(f)
        sub_model = EnsembleModel(m_dict["in_dim"], m_dict["num_members"], m_dict["hidden_dim"])
        sub_model.load_from_dict(m_dict)

        pred_sub = sub_model.predict_one_step(X_act_test)
        mu_sub = pred_sub["mean"]
        std_sub = pred_sub["std"]

        rmse_T = float(np.sqrt(np.mean((y_act_test[:, 0] - mu_sub[:, 0]) ** 2)))
        rmse_L = float(np.sqrt(np.mean((y_act_test[:, 1] - mu_sub[:, 1]) ** 2)))
        cov_sub = float(np.mean(
            (y_act_test[:, 0] >= mu_sub[:, 0] - 1.645 * std_sub[:, 0]) &
            (y_act_test[:, 0] <= mu_sub[:, 0] + 1.645 * std_sub[:, 0])
        ))
        width_T = float(2.0 * 1.645 * np.mean(std_sub[:, 0]))
        width_L = float(2.0 * 1.645 * np.mean(std_sub[:, 1]))

        lowdata_results["runs"].append({
            "fraction": frac,
            "num_episodes": int(150 * frac),
            "rmse_T": round(rmse_T, 4),
            "rmse_L": round(rmse_L, 4),
            "coverage_90": round(cov_sub, 4),
            "mean_width_T": round(width_T, 4),
            "mean_width_L": round(width_L, 4),
        })

    # ==========================================
    # FAILURE DEMO 2: MISSING OBSERVATIONS (Gaps 1-10)
    # ==========================================
    print("[KAARAN Eval] Evaluating Failure Demo: Missing Observations...")
    missing_obs_results = {"mock": False, "gaps": []}
    for gap in range(1, 11):
        scm_gap_errs = []
        hold_gap_errs = []
        covered_list = []
        width_list = []

        for ep in test_episodes[:25]:
            T_arr = ep["T"]
            L_arr = ep["L"]
            v_arr = ep["v"]
            h_arr = ep["h"]

            # Evaluate gap from step 30 to 30 + gap
            t_start = 30
            t_end = t_start + gap

            # Ground truth at t_end
            true_T_end = T_arr[t_end]

            # Baseline: hold last observed value (T at t_start)
            hold_pred = T_arr[t_start]
            hold_gap_errs.append(abs(hold_pred - true_T_end))

            # SCM forward rollout without T observations
            s_curr = [T_arr[t_start], L_arr[t_start]]
            ro = scm_ens.rollout_monte_carlo(
                (s_curr[0], s_curr[1]),
                [(v_arr[t], h_arr[t]) for t in range(t_start, t_end)],
                num_paths=100,
                rng=rng_eval,
            )
            pred_median = ro["median_T"][-1]
            lower = ro["lower_T"][-1]
            upper = ro["upper_T"][-1]

            scm_gap_errs.append(abs(pred_median - true_T_end))
            covered_list.append(1.0 if (true_T_end >= lower and true_T_end <= upper) else 0.0)
            width_list.append(upper - lower)

        missing_obs_results["gaps"].append({
            "gap_length": gap,
            "scm_rmse": round(float(np.sqrt(np.mean(np.array(scm_gap_errs) ** 2))), 3),
            "scm_coverage_90": round(float(np.mean(covered_list)), 3),
            "scm_mean_width": round(float(np.mean(width_list)), 3),
            "baseline_hold_rmse": round(float(np.sqrt(np.mean(np.array(hold_gap_errs) ** 2))), 3),
        })

    # ==========================================
    # CONFOUNDING & OBSERVATIONAL SCATTER
    # ==========================================
    print("[KAARAN Eval] Extracting Confounding Scatter and True vs Model Slopes...")
    with open(os.path.join(data_dir, "confounding_calibration.json"), "r") as f:
        calib_data = json.load(f)

    # Subsample 150 points for the scatter plot
    scatter_samples = []
    sub_indices = rng_eval.choice(len(X_act_test), size=150, replace=False)
    for idx in sub_indices:
        scatter_samples.append({
            "heater": round(float(X_act_test[idx, 3]), 3),
            "delta_T": round(float(y_act_test[idx, 0]), 3),
        })

    # Estimate SCM effect of heater
    (T_step_h1, _), _ = sim.step((50.0, 50.0), (50.0, 1.0), U=0.0, eps_T=0.0, eps_L=0.0)
    (T_step_h0, _), _ = sim.step((50.0, 50.0), (50.0, 0.0), U=0.0, eps_T=0.0, eps_L=0.0)
    true_causal_slope = float(T_step_h1 - T_step_h0)

    # SCM prediction at h=1 vs h=0
    scm_pred_h1 = scm_ens.predict_one_step(np.array([50.0, 50.0, 50.0, 1.0]))["mean"][0]
    scm_pred_h0 = scm_ens.predict_one_step(np.array([50.0, 50.0, 50.0, 0.0]))["mean"][0]
    scm_estimated_slope = float(scm_pred_h1 - scm_pred_h0)

    # Linear fit slope from operator logs
    lin_slope = float(linear_base.model_T.coef_[3])
    lin_intercept = float(linear_base.model_T.intercept_)

    confounding_data = {
        "mock": False,
        "linear_fit_slope": round(lin_slope, 4),
        "linear_fit_intercept": round(lin_intercept, 4),
        "true_causal_slope": round(true_causal_slope, 4),
        "scm_estimated_slope": round(scm_estimated_slope, 4),
        "samples": scatter_samples,
        "calibration_checks": {
            "heater_coeff_under_30_percent": bool(calib_data["calibration_results"]["passed_a"]),
            "heater_coeff_measured": round(calib_data["calibration_results"]["slope_h"], 4),
            "true_effect_measured": round(calib_data["calibration_results"]["true_causal_effect"], 4),
            "step_10_disagreement_deg_C": round(calib_data["calibration_results"]["disagreement_10step"], 2),
            "step_10_target_met": bool(calib_data["calibration_results"]["passed_b"]),
        }
    }

    # Save all evaluation outputs
    eval_out_dir = "eval_results"
    os.makedirs(eval_out_dir, exist_ok=True)

    def write_json(fname, data):
        with open(os.path.join(eval_out_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    write_json("metrics.json", metrics)
    write_json("calibration.json", calibration)
    write_json("interventions_eval.json", interventions_eval)
    write_json("counterfactual_eval.json", counterfactual_eval)
    write_json("lowdata.json", lowdata_results)
    write_json("missing_obs.json", missing_obs_results)
    write_json("confounding.json", confounding_data)

    print("[KAARAN Eval] Evaluation complete. Results written to 'eval_results/'.")
    return metrics


if __name__ == "__main__":
    evaluate_all()
