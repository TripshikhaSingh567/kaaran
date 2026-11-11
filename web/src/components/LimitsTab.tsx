"use client";

import React from "react";
import { AlertTriangle, ShieldX, HelpCircle, CheckCircle, Flame, Droplet } from "lucide-react";

interface LimitsTabProps {
  trustmapData: any;
  confoundingData: any;
}

export const LimitsTab: React.FC<LimitsTabProps> = ({ trustmapData, confoundingData }) => {
  const scores = trustmapData?.scores || {};
  const thresholds = trustmapData?.thresholds || {};

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>Operational Limitations & Boundary Analysis</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Documented failure boundaries, structural assumptions, and un-modelled physical regimes verified by real benchmark data.
          </p>
        </div>
        <div className="text-xs font-mono text-slate-500">
          Source: <span className="font-semibold text-navy-800">docs/LIMITATIONS.md</span>
        </div>
      </div>

      {/* 3 Core Limitations Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Unreliable Operating Regions */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center gap-2 font-bold text-sm text-rose-700 border-b pb-2">
            <ShieldX className="w-4 h-4" />
            <h3>Unreliable Operating Regions</h3>
          </div>
          <p className="text-xs text-slate-600">
            Based on the calibrated Trust Map (Spearman r = {scores.rank_correlation_spearman ?? 0.66}), the model is untrustworthy in:
          </p>
          <ul className="text-xs space-y-2 text-slate-700 list-disc list-inside">
            <li>
              <span className="font-semibold text-slate-900">High Temperature (T &gt; 72°C):</span> Safety interlock forces heater off during probe training. Evaporation non-linearity triggers at 70°C (0.01*(T-70)^2).
            </li>
            <li>
              <span className="font-semibold text-slate-900">Low Tank Level (L &lt; 20%):</span> Reduced thermal mass causes extreme heating sensitivity.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Red Cell Error:</span> Untrusted cells exhibit true errors exceeding {((thresholds.critical_error_threshold ?? 0.41)).toFixed(2)} °C.
            </li>
          </ul>
        </div>

        {/* Model & Structural Limits */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center gap-2 font-bold text-sm text-navy-900 border-b pb-2">
            <Flame className="w-4 h-4 text-orange-500" />
            <h3>Physical Dynamic Limits</h3>
          </div>
          <p className="text-xs text-slate-600">
            Assumptions baked into the 2-layer MLP structural equation models:
          </p>
          <ul className="text-xs space-y-2 text-slate-700 list-disc list-inside">
            <li>
              <span className="font-semibold text-slate-900">Additive Gaussian Shocks:</span> Residual noise is assumed conditionally Gaussian. Extreme Bernoulli or heavy-tailed shocks degrade abduction accuracy.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Discrete 1-Min Time Steps:</span> Sub-minute thermal oscillations or pipe wave acoustics are lumped into process noise.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Markovian Transition:</span> Dynamics assume (s_t, a_t) contains all sufficient state history.
            </li>
          </ul>
        </div>

        {/* What Breaks Assumptions */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-3">
          <div className="flex items-center gap-2 font-bold text-sm text-amber-700 border-b pb-2">
            <AlertTriangle className="w-4 h-4" />
            <h3>Assumptions Invalidation Risk</h3>
          </div>
          <p className="text-xs text-slate-600">
            Conditions under which KAARAN's causal guarantees collapse:
          </p>
          <ul className="text-xs space-y-2 text-slate-700 list-disc list-inside">
            <li>
              <span className="font-semibold text-slate-900">Correlated Probe Interventions:</span> If probe experiments correlate with ambient temperature U, identification is lost.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Mechanism Non-Autonomy:</span> If intervening on valve causes actuator physical failure or alters pipe geometry.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Sensor Drift:</span> Uncalibrated thermocouples violate the noise abduction invertibility condition.
            </li>
          </ul>
        </div>
      </div>

      {/* Safe Operating Envelope Recommendation */}
      <div className="bg-emerald-50 border border-emerald-200 p-6 rounded-xl">
        <h3 className="text-sm font-bold text-emerald-950 flex items-center gap-2 mb-2">
          <CheckCircle className="w-4 h-4 text-emerald-600" />
          <span>Recommended Safe Operational Envelope for Autonomous Deployment</span>
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono text-emerald-900 mt-3">
          <div className="bg-white p-3 rounded-lg border border-emerald-100">
            <span className="text-[10px] uppercase text-emerald-600 block">Temperature Window</span>
            <span className="font-bold text-sm">35.0 °C – 68.0 °C</span>
          </div>
          <div className="bg-white p-3 rounded-lg border border-emerald-100">
            <span className="text-[10px] uppercase text-emerald-600 block">Level Window</span>
            <span className="font-bold text-sm">30.0 % – 80.0 %</span>
          </div>
          <div className="bg-white p-3 rounded-lg border border-emerald-100">
            <span className="text-[10px] uppercase text-emerald-600 block">Valve Range</span>
            <span className="font-bold text-sm">15 % – 85 %</span>
          </div>
          <div className="bg-white p-3 rounded-lg border border-emerald-100">
            <span className="text-[10px] uppercase text-emerald-600 block">Heater Duty</span>
            <span className="font-bold text-sm">0.0 – 0.85 duty</span>
          </div>
        </div>
      </div>
    </div>
  );
};
