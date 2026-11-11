/**
 * Deterministic Heated Tank Simulator in TypeScript.
 * Matches Python implementation exactly for golden parity.
 */

export interface SimulatorConstants {
  level_valve_coeff: number;
  level_decay_coeff: number;
  level_noise_scale: number;
  heat_base_rate: number;
  heat_ref_level: number;
  heat_min_level: number;
  cool_valve_factor: number;
  cool_ambient_temp: number;
  loss_ambient_factor: number;
  loss_ambient_temp: number;
  evap_rate: number;
  evap_threshold_temp: number;
  temp_confounder_effect: number;
  temp_noise_base: number;
  temp_noise_valve_coeff: number;
}

export const DEFAULT_CONSTANTS: SimulatorConstants = {
  level_valve_coeff: 0.06,
  level_decay_coeff: 0.05,
  level_noise_scale: 0.3,
  heat_base_rate: 6.0,
  heat_ref_level: 50.0,
  heat_min_level: 15.0,
  cool_valve_factor: 0.002,
  cool_ambient_temp: 15.0,
  loss_ambient_factor: 0.02,
  loss_ambient_temp: 25.0,
  evap_rate: 0.01,
  evap_threshold_temp: 70.0,
  temp_confounder_effect: -1.5,
  temp_noise_base: 0.15,
  temp_noise_valve_coeff: 0.008,
};

export interface StepOutput {
  T_next: number;
  L_next: number;
  delta_T: number;
  delta_L: number;
}

export function clip(val: number, min: number, max: number): number {
  return Math.min(Math.max(val, min), max);
}

export function simulatorStep(
  s: [number, number],
  a: [number, number],
  U: number,
  eps_T: number,
  eps_L: number,
  cfg: SimulatorConstants = DEFAULT_CONSTANTS
): StepOutput {
  const T = s[0];
  const L = s[1];
  const v = clip(a[0], 0.0, 100.0);
  const h = clip(a[1], 0.0, 1.0);

  // Hydraulic: L' = clip(L + 0.06*v - 0.05*L + 0.3*eps_L, 0, 100)
  const delta_L_raw = cfg.level_valve_coeff * v - cfg.level_decay_coeff * L + cfg.level_noise_scale * eps_L;
  const L_next = clip(L + delta_L_raw, 0.0, 100.0);

  // Thermal:
  const effective_L = Math.max(L, cfg.heat_min_level);
  const heat = cfg.heat_base_rate * h * (cfg.heat_ref_level / effective_L);
  const cool = cfg.cool_valve_factor * v * (T - cfg.cool_ambient_temp);
  const loss = cfg.loss_ambient_factor * (T - cfg.loss_ambient_temp);
  const evap = cfg.evap_rate * Math.pow(Math.max(T - cfg.evap_threshold_temp, 0.0), 2);

  const confounder_term = cfg.temp_confounder_effect * U;
  const thermal_noise_sigma = cfg.temp_noise_base + cfg.temp_noise_valve_coeff * v;
  const noise_term = thermal_noise_sigma * eps_T;

  const delta_T_raw = heat - cool - loss - evap + confounder_term + noise_term;
  const T_next = clip(T + delta_T_raw, 0.0, 120.0);

  return {
    T_next,
    L_next,
    delta_T: T_next - T,
    delta_L: L_next - L,
  };
}
