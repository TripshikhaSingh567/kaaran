# KAARAN: Operational Boundaries, Assumptions & Model Limitations

## 1. Unreliable Operating Regions (Quantified from Real Benchmarks)

The Epistemic Trust Map evaluates 1,156 state-action cells across $(T, L) \in [20, 100]^\circ\text{C} \times [15, 95]\%$:

1. **Extreme Thermal Region ($T > 72^\circ\text{C}$)**:
   - **Reason**: The probe policy has a strict safety interlock cutting heater duty to zero when $T > 72^\circ\text{C}$. The model therefore has zero training observations of active heating at elevated temperatures.
   - **Non-Linear Dynamics**: Beyond $70^\circ\text{C}$, latent evaporation kicks in non-linearly:
     $$\text{evap} = 0.01 \cdot \max(T - 70, 0)^2$$
   - **Measured Error**: In this region, true simulator error spikes to $> 3.5^\circ\text{C}$, flagged as **RED** by the Trust Map with 75.24% precision.

2. **Low Tank Depth ($L < 20\%$)**:
   - **Reason**: Thermal mass drops drastically. Small heater duty inputs produce sudden, violent temperature spikes ($\text{heat} \propto 1 / \max(L, 15)$).
   - **Measured Error**: Disagreement between ensemble members increases by $2.8\times$ compared to the nominal in-distribution baseline ($0.2076^\circ\text{C}$).

3. **High Valve / Cold Quench Regime ($v > 85\%, T > 80^\circ\text{C}$)**:
   - Cold inflow at $15^\circ\text{C}$ creates large convective gradients ($0.002 \cdot v \cdot (T - 15)$) that produce severe hydraulic-thermal coupling.

---

## 2. Structural & Architectural Model Limits

1. **Discrete 1-Minute Sampling Time**:
   - Sub-minute fluid transport delays, thermal stratification, and valve actuator slew rates are treated as instantaneous within the 1-minute discrete step.
   - High-frequency oscillations are absorbed into the heteroscedastic noise term $\epsilon_T$.

2. **Additive Gaussian Disturbance Assumption**:
   - The abduction engine assumes unobserved noise $\hat{\epsilon}$ is conditionally standard Gaussian:
     $$\hat{\epsilon}_t = \frac{s_{t+1} - \mu(s_t, a_t)}{\sigma(s_t, a_t)}$$
   - If the system is struck by heavy-tailed non-Gaussian shocks (e.g. sudden structural leak or power surge), Gaussian abduction under-estimates the tail risk.

3. **Two-Layer MLP Expressiveness**:
   - The ensemble consists of 5 members with 2 hidden layers of 64 units each and `tanh` activations. While highly effective on smooth thermal curves, sharp discontinuities in physical state spaces (e.g., phase transitions) would require piecewise or neural ODE formulations.

---

## 3. Invalidation Conditions (What Would Break the Causal Assumptions?)

KAARAN's causal guarantees rest upon specific identification requirements. The model will fail if:

1. **Probe Randomization is Violated (Confounded Experiments)**:
   - If randomized probe testing is executed during maintenance where ambient weather $U$ influences the technician's choice of test values, the backdoor edge $U \to a$ re-opens, destroying causal identifiability.

2. **Mechanism Non-Autonomy (Physical Cross-Talk)**:
   - The SCM assumes that intervening on the valve $do(v)$ or heater $do(h)$ does not alter the underlying transition law $f(\cdot)$. If an intervention physically damages a valve seal or cracks an element, the structural equations cease to hold.

3. **Sensor Calibration Drift**:
   - If the temperature sensor drifts over time (e.g. +2°C bias), historical disturbance abduction is corrupted, degrading counterfactual accuracy to the level of fresh re-simulation.

---

## 4. Recommended Safe Deployment Envelope

For autonomous or semi-autonomous supervisory control, KAARAN should operate strictly within:
- **Temperature ($T$)**: $35.0^\circ\text{C} \le T \le 68.0^\circ\text{C}$
- **Liquid Level ($L$)**: $30.0\% \le L \le 80.0\%$
- **Valve Position ($v$)**: $15\% \le v \le 85\%$
- **Heater Duty ($h$)**: $0.0 \le h \le 0.85$
- **Trust Map Gate**: Automated interlock to require operator confirmation whenever a proposed action transitions the Trust Map cell into **AMBER** or **RED**.
