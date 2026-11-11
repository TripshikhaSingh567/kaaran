/**
 * MLP Ensemble Forward Pass in TypeScript.
 * Matches PyTorch CPU weights and computation down to 1e-5.
 */

export interface MemberWeights {
  in_mean: number[];
  in_std: number[];
  fc1_weight: number[][]; // [64, in_dim]
  fc1_bias: number[];     // [64]
  fc2_weight: number[][]; // [64, 64]
  fc2_bias: number[];     // [64]
  out_weight: number[][]; // [4, 64]
  out_bias: number[];     // [4]
}

export interface EnsembleWeights {
  mock?: boolean;
  in_dim: number;
  num_members: number;
  hidden_dim: number;
  members: MemberWeights[];
}

export interface PredictionOutput {
  mean_delta_T: number;
  mean_delta_L: number;
  std_delta_T: number;
  std_delta_L: number;
  member_means: [number, number][];
  member_sigmas: [number, number][];
}

function matVecMul(W: number[][], x: number[], b: number[]): number[] {
  const rows = W.length;
  const cols = W[0].length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = b[i];
    const row = W[i];
    for (let j = 0; j < cols; j++) {
      sum += row[j] * x[j];
    }
    out[i] = sum;
  }
  return out;
}

function applyTanh(arr: number[]): number[] {
  return arr.map(v => Math.tanh(v));
}

export function forwardMember(
  x: number[],
  weights: MemberWeights
): { mu: [number, number]; sigma: [number, number] } {
  // 1. Standardize input
  const x_norm = x.map((val, idx) => {
    const std = weights.in_std[idx] < 1e-6 ? 1.0 : weights.in_std[idx];
    return (val - weights.in_mean[idx]) / std;
  });

  // 2. Layer 1: Linear -> Tanh
  const h1 = applyTanh(matVecMul(weights.fc1_weight, x_norm, weights.fc1_bias));

  // 3. Layer 2: Linear -> Tanh
  const h2 = applyTanh(matVecMul(weights.fc2_weight, h1, weights.fc2_bias));

  // 4. Layer 3 (out): Linear -> [mu_T, mu_L, log_sigma_T, log_sigma_L]
  const out = matVecMul(weights.out_weight, h2, weights.out_bias);

  const mu_T = out[0];
  const mu_L = out[1];

  // Clamp log_sigma between -4.0 and 2.0
  const log_sig_T = Math.min(Math.max(out[2], -4.0), 2.0);
  const log_sig_L = Math.min(Math.max(out[3], -4.0), 2.0);

  const sig_T = Math.exp(log_sig_T);
  const sig_L = Math.exp(log_sig_L);

  return {
    mu: [mu_T, mu_L],
    sigma: [sig_T, sig_L],
  };
}

export function predictEnsemble(
  x: number[],
  ensemble: EnsembleWeights
): PredictionOutput {
  const member_means: [number, number][] = [];
  const member_sigmas: [number, number][] = [];

  let sum_mu_T = 0.0;
  let sum_mu_L = 0.0;
  let sum_var_T = 0.0;
  let sum_var_L = 0.0;

  const M = ensemble.members.length;

  for (let m = 0; m < M; m++) {
    const res = forwardMember(x, ensemble.members[m]);
    member_means.push(res.mu);
    member_sigmas.push(res.sigma);

    sum_mu_T += res.mu[0];
    sum_mu_L += res.mu[1];
    sum_var_T += res.sigma[0] * res.sigma[0];
    sum_var_L += res.sigma[1] * res.sigma[1];
  }

  const ens_mu_T = sum_mu_T / M;
  const ens_mu_L = sum_mu_L / M;

  // Epistemic variance (variance of means)
  let epistemic_T = 0.0;
  let epistemic_L = 0.0;
  for (let m = 0; m < M; m++) {
    const diff_T = member_means[m][0] - ens_mu_T;
    const diff_L = member_means[m][1] - ens_mu_L;
    epistemic_T += diff_T * diff_T;
    epistemic_L += diff_L * diff_L;
  }
  epistemic_T /= M;
  epistemic_L /= M;

  // Aleatoric variance (mean of variances)
  const aleatoric_T = sum_var_T / M;
  const aleatoric_L = sum_var_L / M;

  const total_var_T = aleatoric_T + epistemic_T;
  const total_var_L = aleatoric_L + epistemic_L;

  return {
    mean_delta_T: ens_mu_T,
    mean_delta_L: ens_mu_L,
    std_delta_T: Math.sqrt(total_var_T),
    std_delta_L: Math.sqrt(total_var_L),
    member_means,
    member_sigmas,
  };
}
