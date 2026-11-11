# KAARAN 3-Minute Hackathon Demo Script

> **Target Presentation Time:** 3 minutes (180 seconds)  
> **Live URL:** Vercel deployment link  
> **Key Message:** Ground truth comes from the physics simulator and is **never shown to the model**. Naive AI learns spurious correlation from operator logs; KAARAN uses causal world models and epistemic uncertainty guardrails to ensure safe, autonomous control.

---

## Minute 0:00 - 0:45 | The Problem & Spurious Correlations
1. **Show Header & Ground Rule**:
   - Point to the top banner: *"Ground truth comes from the simulator and is never shown to the model."*
   - *"Judges, in continuous industrial control, observational data is deeply confounded. Human operators react to unobserved ambient conditions $U$ (e.g. ambient cold). When it gets cold, they turn the heater UP. As a result, in historical logs, high heater duty correlates with lower temperatures."*
2. **Click "Scorecard & Demos" Tab**:
   - Scroll to the **Confounder Scatter Plot**:
   - *"Look at this real regression line from our 60,000 operator transition steps: the slope is **`-3.07`**! A conventional ML model concludes that turning the heater on cools the tank down, while true physics is **`+6.00`**. DoWhy proves formally that observational data alone CANNOT identify the true causal effect."*

---

## Minute 0:45 - 1:30 | The Intervention Catastrophe (do(a))
3. **Click "Intervene do(a)" Tab**:
   - Select the preset **`do(h = 0)`** (cut heater to 0 at nominal $T=50^\circ\text{C}$).
   - Point to the live fan chart:
   - *"Watch what happens when an autonomous agent tries to intervene. The observational model (red dashed line) predicts temperature stays warm at 45.6°C because it falsely believes ambient $U$ is hot. But true physics (black dots) and our KAARAN SCM (blue band) plummet to under 20°C!"*
   - Point to the callout badge:
   - *"The observational model has an error of **`45.55 °C`**! KAARAN's SCM Ensemble, trained on randomized probe data, matches physical reality within **`0.35 °C`** with calibrated 90% confidence bands."*

---

## Minute 1:30 - 2:15 | Counterfactual "What-If" Incident Analysis
4. **Click "What-If (Counterfactual)" Tab**:
   - Choose **Case #0 (Episode 0)**, where heater was cut at step $k=20$.
   - Walk through the 3 tabs: **1. Abduct -> 2. Change -> 3. Replay**:
   - *"KAARAN doesn't just do fresh simulation. It executes Pearl's 3-step counterfactual algorithm. It inverts historical transitions to abduct the exact disturbance shocks $\hat{\epsilon}_t$ that hit that specific batch, modifies the decision, and replays."*
   - Point to the chart comparison:
   - *"Notice the orange line: a standard simulation without noise reuse has an error of **`4.06 °C`**. KAARAN achieves **`0.10 °C`**—a **39x error reduction** because it preserves historical physical reality."*

---

## Minute 2:15 - 3:00 | Trust Map Guardrails & Safe Deployment
5. **Click "Trust Map" Tab**:
   - Switch presets between *Operator Typical* and *do(h = 1) [Stress]*:
   - *"In a real plant, how do we know when the model doesn't know? Our Trust Map combines ensemble epistemic disagreement and standardized distance to training data."*
   - Hover over a red cell at $T=85^\circ\text{C}, L=85\%$:
   - *"Look at the inspector: OOD distance $> 2.0\sigma$, disagreement spikes, and the model turns RED. Red cells achieve **75.24% precision** in detecting true simulator errors $> 2\times$ the safe baseline. The model refuses to hallucinate."*
6. **Wrap up**:
   - *"KAARAN bridges the gap between pure ML and physical process reality. Thank you!"*

---

## Top 5 Likely Judge Questions & Honest Answers

### Q1: "Why can't you just use standard Reinforcement Learning or System Identification?"
**Answer:** Standard RL and SysID typically either assume full state observability (which fails when latent confounders like ambient conditions $U$ exist) or require tens of thousands of dangerous online exploratory trials. KAARAN explicitly separates observational logs from a minimal randomized probe budget (just 150 short episodes) to achieve formal causal identification without endangering the physical plant.

### Q2: "How do you guarantee that your abducted disturbances $\hat{\epsilon}$ are correct?"
**Answer:** Abduction relies on the invertibility of the structural equation $s_{t+1} = \mu(s_t, a_t) + \sigma(s_t, a_t)\epsilon_t$. Under our conditional Gaussian assumption, $\hat{\epsilon}$ is uniquely determined by the standardized residual. If non-linear, non-additive shocks occur, abduction provides the best linear unbiased estimate under Gaussian projection, which we demonstrated still beats fresh re-simulation by 39x.

### Q3: "What happens if the operator changes their policy over time?"
**Answer:** In observational models, operator drift ruins prediction. But KAARAN's SCM is trained on randomized probe data, making it invariant to the operator's behavioral policy. The underlying physical transition mechanism $f(s, a)$ remains invariant even if operator behavior shifts completely.

### Q4: "Why does the observational model have a negative slope (-3.07) for heater duty?"
**Answer:** The human operator is compensating for the hidden ambient confounder $U$: when $U$ is low (cold weather), the operator turns the heater up ($h \uparrow$). Since $U$ directly causes thermal loss, the logs exhibit a strong negative correlation between $h$ and $\Delta T$. Linear regression conflates this correlation with causation.

### Q5: "How is the web application running client-side with zero server?"
**Answer:** We ported the simulator dynamics and the 5-member MLP ensemble forward pass directly into TypeScript. Pre-computed weights and evaluation artifacts are stored as static JSON assets. The browser executes Monte Carlo rollouts and neural network inferences in sub-millisecond client-side loops with zero API keys or backend dependencies.
