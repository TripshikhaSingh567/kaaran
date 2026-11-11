# KAARAN Specification Document

## 1. Executive Overview
**KAARAN** is an uncertainty-aware causal world model designed for industrial process control, specifically applied to a continuous heated process tank system. Ground truth is governed by underlying physical conservation laws subject to hidden confounding, unobserved disturbances, and non-linearities.

## 2. State & Action Spaces
- **State $s = (T, L)$**:
  - $T$: Fluid Temperature in degrees Celsius ($[0, 120]^\circ\text{C}$). Safety limit is $72^\circ\text{C}$.
  - $L$: Liquid Level percentage ($[0, 100]\%$).
- **Action $a = (v, h)$**:
  - $v$: Inflow Control Valve position ($[0, 100]\%$).
  - $h$: Heating Element duty cycle ($[0, 1]$).
- **Latents and Disturbances**:
  - $U_t \sim \mathcal{N}(0, 1)$: Hidden ambient confounder affecting both operator actions and process thermal dynamics.
  - $\epsilon_{T,t} \sim \mathcal{N}(0, 1)$: Unobserved thermal process disturbance.
  - $\epsilon_{L,t} \sim \mathcal{N}(0, 1)$: Unobserved hydraulic process disturbance.

## 3. Physical Transition Dynamics
For discrete time-step $\Delta t = 1$ minute:
$$L_{t+1} = \text{clip}\left(L_t + 0.06 v_t - 0.05 L_t + 0.3 \epsilon_{L,t},\ 0,\ 100\right)$$

$$\text{heat}_t = 6.0 \cdot h_t \cdot \left(\frac{50}{\max(L_t, 15)}\right)$$
$$\text{cool}_t = 0.002 \cdot v_t \cdot (T_t - 15)$$
$$\text{loss}_t = 0.02 \cdot (T_t - 25)$$
$$\text{evap}_t = 0.01 \cdot \max(T_t - 70, 0)^2$$
$$T_{t+1} = \text{clip}\left(T_t + \text{heat}_t - \text{cool}_t - \text{loss}_t - \text{evap}_t - 1.5 U_t + (0.15 + 0.008 v_t)\epsilon_{T,t},\ 0,\ 120\right)$$

## 4. Behavioral Policies
1. **Operator Policy (Observational Logs)**:
   The operator actively reacts to ambient conditions ($U_t$) and regulation setpoints ($T^*=55^\circ\text{C}, L^*=60\%$):
   $$h_t = \text{clip}\left(0.55 + 0.25 U_t + 0.02(55 - T_t) + 0.05 n_{h,t},\ 0,\ 1\right)$$
   $$v_t = \text{clip}\left(50 + 8 U_t + 0.7(60 - L_t) + 3 n_{v,t},\ 0,\ 100\right)$$
   where $n_{h,t}, n_{v,t} \sim \mathcal{N}(0, 1)$.
   *Causal Implication*: Because $U_t$ drives both $h_t$ and $T_{t+1}$, naive regression from observational logs conflates correlation with causation.

2. **Probe Policy (Randomized Causal Identification)**:
   Randomized actions independent of $U_t$:
   - Every 8 steps: draw $v_t \sim \mathcal{U}(0, 100)$ and $h_t \sim \mathcal{U}(0, 1)$ and hold.
   - Safety Interlock: If $T_t > 72^\circ\text{C}$, force $h_t = 0$.

## 5. Model Architecture & Ensembling
- **Structure**: Deep Ensembles of 5 MLPs (2 hidden layers of 64 units, `tanh` activation, standardized inputs).
- **Outputs**: Gaussian predictive distributions $\mathcal{N}(\mu(s, a), \sigma^2(s, a))$ predicting one-step state transitions $(\Delta T, \Delta L)$.
- **Uncertainty Decomposition**:
  $$\mu_*(s, a) = \frac{1}{M}\sum_{m=1}^M \mu_m(s, a)$$
  $$\sigma_*^2(s, a) = \frac{1}{M}\sum_{m=1}^M \sigma_m^2(s, a) + \frac{1}{M}\sum_{m=1}^M \left(\mu_m(s, a) - \mu_*(s, a)\right)^2$$
- **Counterfactual Engine**:
  1. *Abduction*: $\hat{\epsilon}_t^{(m)} = \frac{s_{t+1} - \mu_m(s_t, a_t)}{\sigma_m(s_t, a_t)}$
  2. *Action Modification*: $a_k \leftarrow a'$
  3. *Replay*: $s_{t+1}^{(m)} = \mu_m(s_t, a_t) + \sigma_m(s_t, a_t) \cdot \hat{\epsilon}_t^{(m)}$

## 6. Trust Map Architecture
- 2D grid over $(T, L) \in [20, 100]^\circ\text{C} \times [10, 95]\%$ for action regimes.
- Metrics:
  1. Epistemic uncertainty: Ensemble disagreement $\text{std}_m(\mu_m)$.
  2. Out-of-Distribution (OOD) distance: Mahalanobis or standardized Euclidean distance to training probe dataset.
- Status classification: Green (nominal trust), Amber (elevated risk / extrapolation), Red (untrustworthy / critical risk).
