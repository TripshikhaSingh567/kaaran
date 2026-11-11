import test from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import ported TypeScript modules (via transpile or direct TS execution or compiled JS)
// Let's implement pure parity check functions directly or import from src/lib
import { simulatorStep, DEFAULT_CONSTANTS } from '../src/lib/simulator.ts';
import { predictEnsemble } from '../src/lib/model.ts';

const parityDataPath = path.resolve(__dirname, '../public/artifacts/golden_parity.json');

test('Golden Parity Test: Simulator matches Python to 1e-6 and Model matches to 1e-5', async (t) => {
  assert.ok(fs.existsSync(parityDataPath), 'golden_parity.json must exist');
  const parityData = JSON.parse(fs.readFileSync(parityDataPath, 'utf-8'));

  if (parityData.mock === true) {
    console.log('Skipping parity test while artifacts are in mock state.');
    return;
  }

  // 1. Verify Simulator Parity (tolerance: 1e-6)
  for (let i = 0; i < parityData.simulator_steps.length; i++) {
    const step = parityData.simulator_steps[i];
    const inp = step.input;
    const expected = step.output;

    const actual = simulatorStep(
      [inp.T, inp.L],
      [inp.v, inp.h],
      inp.U,
      inp.eps_T,
      inp.eps_L,
      DEFAULT_CONSTANTS
    );

    const diff_T = Math.abs(actual.T_next - expected.T_next);
    const diff_L = Math.abs(actual.L_next - expected.L_next);

    assert.ok(
      diff_T < 1e-6,
      `Step ${i} T_next mismatch: actual=${actual.T_next}, expected=${expected.T_next}, diff=${diff_T}`
    );
    assert.ok(
      diff_L < 1e-6,
      `Step ${i} L_next mismatch: actual=${actual.L_next}, expected=${expected.L_next}, diff=${diff_L}`
    );
  }

  // 2. Verify Model Parity (tolerance: 1e-5)
  const scmWeightsPath = path.resolve(__dirname, '../public/artifacts/weights_scm.json');
  assert.ok(fs.existsSync(scmWeightsPath), 'weights_scm.json must exist');
  const scmWeights = JSON.parse(fs.readFileSync(scmWeightsPath, 'utf-8'));

  for (let i = 0; i < parityData.model_predictions.length; i++) {
    const predCase = parityData.model_predictions[i];
    const inp = predCase.input;
    const expected = predCase.output;

    const actual = predictEnsemble([inp.T, inp.L, inp.v, inp.h], scmWeights);

    const diff_mu_T = Math.abs(actual.mean_delta_T - expected.mean_delta_T);
    const diff_mu_L = Math.abs(actual.mean_delta_L - expected.mean_delta_L);
    const diff_std_T = Math.abs(actual.std_delta_T - expected.std_delta_T);
    const diff_std_L = Math.abs(actual.std_delta_L - expected.std_delta_L);

    assert.ok(
      diff_mu_T < 1e-5,
      `Model test ${i} mean_delta_T mismatch: actual=${actual.mean_delta_T}, expected=${expected.mean_delta_T}, diff=${diff_mu_T}`
    );
    assert.ok(
      diff_mu_L < 1e-5,
      `Model test ${i} mean_delta_L mismatch: actual=${actual.mean_delta_L}, expected=${expected.mean_delta_L}, diff=${diff_mu_L}`
    );
    assert.ok(
      diff_std_T < 1e-5,
      `Model test ${i} std_delta_T mismatch: actual=${actual.std_delta_T}, expected=${expected.std_delta_T}, diff=${diff_std_T}`
    );
    assert.ok(
      diff_std_L < 1e-5,
      `Model test ${i} std_delta_L mismatch: actual=${actual.std_delta_L}, expected=${expected.std_delta_L}, diff=${diff_std_L}`
    );
  }
});
