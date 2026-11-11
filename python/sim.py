"""
KAARAN Heated Process Tank Simulator.
Deterministic physical transition dynamics given (s, a, U, eps_T, eps_L).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class HeatedTankSimulator:
    """
    Physical simulation of a heated continuous process tank.
    State: s = (T: Temperature in °C [0, 120], L: Liquid level in % [0, 100])
    Action: a = (v: Valve opening [0, 100], h: Heater duty [0, 1])
    Latent Confounder: U ~ N(0, 1)
    Process Noise: eps_T ~ N(0, 1), eps_L ~ N(0, 1)
    """

    def __init__(
        self,
        level_valve_coeff: float = 0.06,
        level_decay_coeff: float = 0.05,
        level_noise_scale: float = 0.3,
        heat_base_rate: float = 6.0,
        heat_ref_level: float = 50.0,
        heat_min_level: float = 15.0,
        cool_valve_factor: float = 0.002,
        cool_ambient_temp: float = 15.0,
        loss_ambient_factor: float = 0.02,
        loss_ambient_temp: float = 25.0,
        evap_rate: float = 0.01,
        evap_threshold_temp: float = 70.0,
        temp_confounder_effect: float = -1.5,
        temp_noise_base: float = 0.15,
        temp_noise_valve_coeff: float = 0.008,
    ):
        self.level_valve_coeff = level_valve_coeff
        self.level_decay_coeff = level_decay_coeff
        self.level_noise_scale = level_noise_scale
        self.heat_base_rate = heat_base_rate
        self.heat_ref_level = heat_ref_level
        self.heat_min_level = heat_min_level
        self.cool_valve_factor = cool_valve_factor
        self.cool_ambient_temp = cool_ambient_temp
        self.loss_ambient_factor = loss_ambient_factor
        self.loss_ambient_temp = loss_ambient_temp
        self.evap_rate = evap_rate
        self.evap_threshold_temp = evap_threshold_temp
        self.temp_confounder_effect = temp_confounder_effect
        self.temp_noise_base = temp_noise_base
        self.temp_noise_valve_coeff = temp_noise_valve_coeff

    def step(
        self,
        s: Tuple[float, float],
        a: Tuple[float, float],
        U: float,
        eps_T: float,
        eps_L: float,
    ) -> Tuple[Tuple[float, float], Dict[str, float]]:
        """
        Pure deterministic step function given state, action, confounder, and noise disturbances.
        s = (T, L)
        a = (v, h)
        """
        T, L = float(s[0]), float(s[1])
        v, h = float(a[0]), float(a[1])

        # Clip action to domain bounds
        v = float(np.clip(v, 0.0, 100.0))
        h = float(np.clip(h, 0.0, 1.0))

        # Hydraulic dynamics: L' = clip(L + 0.06*v - 0.05*L + 0.3*eps_L, 0, 100)
        delta_L_raw = self.level_valve_coeff * v - self.level_decay_coeff * L + self.level_noise_scale * eps_L
        L_next = float(np.clip(L + delta_L_raw, 0.0, 100.0))

        # Thermal dynamics
        effective_L = max(L, self.heat_min_level)
        heat = self.heat_base_rate * h * (self.heat_ref_level / effective_L)
        cool = self.cool_valve_factor * v * (T - self.cool_ambient_temp)
        loss = self.loss_ambient_factor * (T - self.loss_ambient_temp)
        evap = self.evap_rate * (max(T - self.evap_threshold_temp, 0.0) ** 2)

        confounder_term = self.temp_confounder_effect * U
        thermal_noise_sigma = self.temp_noise_base + self.temp_noise_valve_coeff * v
        noise_term = thermal_noise_sigma * eps_T

        delta_T_raw = heat - cool - loss - evap + confounder_term + noise_term
        T_next = float(np.clip(T + delta_T_raw, 0.0, 120.0))

        components = {
            "heat": heat,
            "cool": cool,
            "loss": loss,
            "evap": evap,
            "confounder_term": confounder_term,
            "thermal_noise_sigma": thermal_noise_sigma,
            "delta_T": T_next - T,
            "delta_L": L_next - L,
        }

        return (T_next, L_next), components

    def noise_free_step(
        self,
        s: Tuple[float, float],
        a: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Noise-free and confounder-free step (U=0, eps_T=0, eps_L=0) for true causal effect calculation."""
        (T_next, L_next), _ = self.step(s, a, U=0.0, eps_T=0.0, eps_L=0.0)
        return T_next, L_next


class OperatorPolicy:
    """
    Operator Policy reacting to setpoint error and latent confounder U:
    h = clip(0.55 + 0.25*U + 0.02*(55-T) + 0.05*n_h, 0, 1)
    v = clip(50 + 8*U + 0.7*(60-L) + 3*n_v, 0, 100)
    """

    def __init__(
        self,
        h_base: float = 0.55,
        h_U: float = 0.25,
        h_T_coeff: float = 0.02,
        h_target_T: float = 55.0,
        h_noise_scale: float = 0.05,
        v_base: float = 50.0,
        v_U: float = 8.0,
        v_L_coeff: float = 0.7,
        v_target_L: float = 60.0,
        v_noise_scale: float = 3.0,
    ):
        self.h_base = h_base
        self.h_U = h_U
        self.h_T_coeff = h_T_coeff
        self.h_target_T = h_target_T
        self.h_noise_scale = h_noise_scale
        self.v_base = v_base
        self.v_U = v_U
        self.v_L_coeff = v_L_coeff
        self.v_target_L = v_target_L
        self.v_noise_scale = v_noise_scale

    def act(
        self,
        s: Tuple[float, float],
        U: float,
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        T, L = s[0], s[1]
        n_h = rng.normal(0.0, 1.0)
        n_v = rng.normal(0.0, 1.0)

        h_val = self.h_base + self.h_U * U + self.h_T_coeff * (self.h_target_T - T) + self.h_noise_scale * n_h
        v_val = self.v_base + self.v_U * U + self.v_L_coeff * (self.v_target_L - L) + self.v_noise_scale * n_v

        h = float(np.clip(h_val, 0.0, 1.0))
        v = float(np.clip(v_val, 0.0, 100.0))
        return v, h


class ProbePolicy:
    """
    Randomized probe policy:
    Every 8 steps, sample v ~ U(0, 100) and h ~ U(0, 1) and hold.
    Safety interlock: if T > 72, force h = 0.
    """

    def __init__(
        self,
        hold_steps: int = 8,
        safety_interlock_T: float = 72.0,
    ):
        self.hold_steps = hold_steps
        self.safety_interlock_T = safety_interlock_T
        self.current_v = 50.0
        self.current_h = 0.5
        self.step_count = 0

    def reset(self, rng: np.random.Generator) -> Tuple[float, float]:
        self.current_v = float(rng.uniform(0.0, 100.0))
        self.current_h = float(rng.uniform(0.0, 1.0))
        self.step_count = 0
        return self.current_v, self.current_h

    def act(
        self,
        s: Tuple[float, float],
        rng: np.random.Generator,
    ) -> Tuple[float, float]:
        if self.step_count % self.hold_steps == 0:
            self.current_v = float(rng.uniform(0.0, 100.0))
            self.current_h = float(rng.uniform(0.0, 1.0))

        v = self.current_v
        h = self.current_h

        # Safety interlock
        T = s[0]
        if T > self.safety_interlock_T:
            h = 0.0

        self.step_count += 1
        return v, h
