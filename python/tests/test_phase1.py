"""
Acceptance tests for Phase 1:
1. Determinism of simulator step function given (s, a, U, eps_T, eps_L)
2. Domain boundary clipping for state and action variables
3. Confounding check (a): Plain linear regression slope < 30% true effect or negative
4. Confounding check (b): Disagreement between logs model and true simulator >= 3 °C after 10 steps under do(h=0)
"""

import os
import yaml
import pytest
import numpy as np

from python.sim import HeatedTankSimulator, OperatorPolicy, ProbePolicy
from python.data import calibrate_confounding, generate_episode


@pytest.fixture
def spec():
    spec_path = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "kaaran_spec.yaml")
    with open(spec_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sim(spec):
    return HeatedTankSimulator(
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


def test_determinism(sim):
    """Verify sim.step is purely deterministic given (s, a, U, eps_T, eps_L)."""
    s = (45.2, 63.8)
    a = (72.5, 0.65)
    U = 0.85
    eps_T = -0.42
    eps_L = 0.33

    out1, comp1 = sim.step(s, a, U, eps_T, eps_L)
    out2, comp2 = sim.step(s, a, U, eps_T, eps_L)

    assert out1[0] == out2[0], "Temperature output must be identical"
    assert out1[1] == out2[1], "Level output must be identical"
    assert comp1 == comp2, "Components must be identical"


def test_clipping_bounds(sim):
    """Verify state and action clipping guarantees."""
    # Extreme high heating and inflow
    s = (119.0, 99.0)
    a = (200.0, 5.0)  # Over max bounds
    U = -10.0  # extreme confounder driving temp up (-1.5 * -10 = +15)
    eps_T = 10.0
    eps_L = 10.0

    (T_next, L_next), _ = sim.step(s, a, U, eps_T, eps_L)
    assert 0.0 <= T_next <= 120.0, f"T_next {T_next} out of bounds [0, 120]"
    assert 0.0 <= L_next <= 100.0, f"L_next {L_next} out of bounds [0, 100]"

    # Extreme low cooling and draining
    s = (2.0, 1.0)
    a = (-50.0, -2.0)  # Under min bounds
    U = 10.0  # extreme confounder driving temp down
    eps_T = -10.0
    eps_L = -10.0

    (T_next_low, L_next_low), _ = sim.step(s, a, U, eps_T, eps_L)
    assert 0.0 <= T_next_low <= 120.0, f"T_next {T_next_low} out of bounds [0, 120]"
    assert 0.0 <= L_next_low <= 100.0, f"L_next {L_next_low} out of bounds [0, 100]"


def test_confounding_calibration(sim, spec):
    """
    Verify:
    (a) plain linear regression of one-step temperature change on heater duty, trained on operator logs,
        gives a heater coefficient that is negative or under 30% of the true effect;
    (b) the logs-only model and the true simulator disagree by at least 3 °C after 10 steps under do(h = 0).
    """
    rng = np.random.default_rng(spec["data_collection"]["random_seed"])
    calibrated_cfg, results = calibrate_confounding(sim, spec, rng)

    assert results["passed_a"], (
        f"Confounding check (a) failed: slope_h={results['slope_h']:.4f}, "
        f"true_effect={results['true_causal_effect']:.4f} "
        f"(ratio={results['slope_h']/results['true_causal_effect']:.2%})"
    )

    assert results["passed_b"], (
        f"Confounding check (b) failed: 10-step disagreement {results['disagreement_10step']:.2f} °C "
        f"is less than 3.0 °C requirement."
    )
