"use client";

import React from "react";
import { Info, Network, CheckCircle, ShieldAlert, Cpu } from "lucide-react";

interface SpecTabProps {
  specData: any;
}

export const SpecTab: React.FC<SpecTabProps> = ({ specData }) => {
  const isMock = specData?.mock === true;

  return (
    <div className="space-y-6">
      {/* Intro banner */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <span>Continuous Heated Process Tank System</span>
            <span className="text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-mono">
              spec/kaaran_spec.yaml
            </span>
          </h2>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl">
            A continuous fluid dynamics & thermodynamics industrial system operating under unobserved
            ambient confounding <span className="font-semibold text-rose-600">U ~ N(0, 1)</span> and
            heteroscedastic process disturbances.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start md:self-auto">
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>Deterministic SCM</span>
          </span>
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-blue-50 text-blue-700 border border-blue-200">
            1 Step = 1 min
          </span>
        </div>
      </div>

      {/* Grid of Spec Columns */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* State Space */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 text-navy-900 font-bold mb-3 border-b pb-2">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-600"></span>
            <h3>State Space (s)</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="font-bold text-slate-800">Temperature (T)</div>
              <div className="text-xs text-slate-500 mt-0.5">Range: 0.0 – 120.0 °C</div>
              <div className="text-xs text-slate-500">Init: U(30.0, 60.0) °C</div>
              <div className="text-xs font-semibold text-rose-600 mt-1">Safety Trip: 72.0 °C</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="font-bold text-slate-800">Tank Level (L)</div>
              <div className="text-xs text-slate-500 mt-0.5">Range: 0.0 – 100.0 %</div>
              <div className="text-xs text-slate-500">Init: U(30.0, 80.0) %</div>
              <div className="text-xs text-slate-600 mt-1">Min effective depth: 15 %</div>
            </div>
          </div>
        </div>

        {/* Action Space */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 text-navy-900 font-bold mb-3 border-b pb-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-600"></span>
            <h3>Action Space (a)</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="font-bold text-slate-800">Inflow Valve (v)</div>
              <div className="text-xs text-slate-500 mt-0.5">Range: 0 – 100 % opening</div>
              <div className="text-xs text-slate-600 mt-1">Introduces cool fluid at 15 °C</div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="font-bold text-slate-800">Heater Duty (h)</div>
              <div className="text-xs text-slate-500 mt-0.5">Range: 0.0 – 1.0 duty cycle</div>
              <div className="text-xs text-slate-600 mt-1">Base rate 6.0 °C/min scaled by 50/max(L,15)</div>
            </div>
          </div>
        </div>

        {/* Unobserved Confounder & Noise */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center gap-2 text-navy-900 font-bold mb-3 border-b pb-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
            <h3>Hidden Latents & Noise</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-rose-50/50 rounded-lg border border-rose-100">
              <div className="font-bold text-rose-900">Latent Confounder U ~ N(0, 1)</div>
              <div className="text-xs text-rose-700 mt-0.5 font-medium">NEVER shown to the model</div>
              <div className="text-xs text-slate-600 mt-1">
                Operator reacts to U, while U directly drives temperature shock (-1.5*U).
              </div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="font-bold text-slate-800">Process Disturbances</div>
              <div className="text-xs text-slate-600 mt-0.5">eps_T, eps_L ~ N(0, 1)</div>
              <div className="text-xs text-slate-500 mt-1">
                Heteroscedastic thermal noise: (0.15 + 0.008*v)*eps_T
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Causal Graph Visualization */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex items-center justify-between mb-4 border-b pb-3">
          <div>
            <h3 className="text-base font-bold text-navy-900 flex items-center gap-2">
              <Network className="w-5 h-5 text-blue-600" />
              <span>Structural Causal Graph (DAG)</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Notice the dashed edges from <span className="font-bold text-rose-600">hidden confounder U(t)</span> creating an unobserved backdoor path.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-semibold">
            <span className="flex items-center gap-1.5 text-slate-700">
              <span className="w-4 h-0.5 bg-slate-800 inline-block"></span>
              <span>Observed / Physical</span>
            </span>
            <span className="flex items-center gap-1.5 text-rose-600">
              <span className="w-4 h-0.5 border-t-2 border-dashed border-rose-500 inline-block"></span>
              <span>Unobserved Confounder / Noise</span>
            </span>
          </div>
        </div>

        {/* SVG DAG */}
        <div className="w-full overflow-x-auto py-4 flex justify-center bg-slate-50/50 rounded-lg border border-slate-100">
          <svg className="w-[720px] h-[340px]" viewBox="0 0 720 340">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
              </marker>
              <marker id="arrow-dashed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#e11d48" />
              </marker>
            </defs>

            {/* Nodes */}
            {/* Latents top row */}
            <g transform="translate(180, 40)">
              <rect x="-60" y="-20" width="120" height="40" rx="20" fill="#ffe4e6" stroke="#f43f5e" strokeWidth="2" strokeDasharray="4 4" />
              <text textAnchor="middle" y="5" fontSize="13" fontWeight="bold" fill="#9f1239">Confounder U(t)</text>
            </g>

            <g transform="translate(540, 40)">
              <rect x="-65" y="-20" width="130" height="40" rx="20" fill="#ffe4e6" stroke="#f43f5e" strokeWidth="2" strokeDasharray="4 4" />
              <text textAnchor="middle" y="5" fontSize="13" fontWeight="bold" fill="#9f1239">Disturbance eps(t)</text>
            </g>

            {/* Time t states & actions */}
            <g transform="translate(80, 160)">
              <rect x="-50" y="-22" width="100" height="44" rx="8" fill="#eff6ff" stroke="#3b82f6" strokeWidth="2" />
              <text textAnchor="middle" y="5" fontSize="13" fontWeight="bold" fill="#1e3a8a">T(t), L(t)</text>
            </g>

            <g transform="translate(240, 160)">
              <rect x="-50" y="-22" width="100" height="44" rx="8" fill="#eef2ff" stroke="#6366f1" strokeWidth="2" />
              <text textAnchor="middle" y="5" fontSize="13" fontWeight="bold" fill="#312e81">Heater h(t)</text>
            </g>

            <g transform="translate(400, 160)">
              <rect x="-50" y="-22" width="100" height="44" rx="8" fill="#eef2ff" stroke="#6366f1" strokeWidth="2" />
              <text textAnchor="middle" y="5" fontSize="13" fontWeight="bold" fill="#312e81">Valve v(t)</text>
            </g>

            {/* Next state t+1 */}
            <g transform="translate(600, 260)">
              <rect x="-70" y="-25" width="140" height="50" rx="10" fill="#f0fdf4" stroke="#16a34a" strokeWidth="2.5" />
              <text textAnchor="middle" y="-2" fontSize="13" fontWeight="bold" fill="#14532d">T(t+1), L(t+1)</text>
              <text textAnchor="middle" y="16" fontSize="10" fontWeight="semibold" fill="#166534">Next Physical State</text>
            </g>

            {/* Edges */}
            {/* U -> h (dashed) */}
            <path d="M 180 60 L 230 135" stroke="#f43f5e" strokeWidth="2" strokeDasharray="5 5" fill="none" markerEnd="url(#arrow-dashed)" />
            {/* U -> v (dashed) */}
            <path d="M 210 55 L 380 135" stroke="#f43f5e" strokeWidth="2" strokeDasharray="5 5" fill="none" markerEnd="url(#arrow-dashed)" />
            {/* U -> T(t+1) (dashed) */}
            <path d="M 220 50 Q 450 60 560 235" stroke="#f43f5e" strokeWidth="2" strokeDasharray="5 5" fill="none" markerEnd="url(#arrow-dashed)" />
            {/* eps -> T(t+1) (dashed) */}
            <path d="M 540 60 L 590 230" stroke="#f43f5e" strokeWidth="2" strokeDasharray="5 5" fill="none" markerEnd="url(#arrow-dashed)" />

            {/* State -> Action (policy feedback) */}
            <path d="M 130 160 L 185 160" stroke="#64748b" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />

            {/* Physical dynamics -> Next State */}
            <path d="M 110 185 Q 250 290 525 270" stroke="#0284c7" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
            <path d="M 260 185 Q 380 270 525 265" stroke="#0284c7" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
            <path d="M 420 185 Q 490 225 530 250" stroke="#0284c7" strokeWidth="2" fill="none" markerEnd="url(#arrow)" />
          </svg>
        </div>
      </div>

      {/* Dynamics Equations & Behavioral Assumptions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm font-mono text-xs">
          <h4 className="font-sans font-bold text-sm text-navy-900 mb-2 border-b pb-1">
            Physical Transition Equations (Simulator Truth)
          </h4>
          <div className="p-3 bg-slate-900 text-slate-100 rounded-lg space-y-2 overflow-x-auto">
            <div><span className="text-blue-400 font-bold">L'</span> = clip(L + 0.06*v - 0.05*L + 0.3*eps_L, 0, 100)</div>
            <div><span className="text-emerald-400">heat</span> = 6.0*h * (50 / max(L, 15))</div>
            <div><span className="text-cyan-400">cool</span> = 0.002*v * (T - 15)</div>
            <div><span className="text-yellow-400">loss</span> = 0.02 * (T - 25)</div>
            <div><span className="text-rose-400">evap</span> = 0.01 * max(T - 70, 0)^2</div>
            <div>
              <span className="text-blue-400 font-bold">T'</span> = clip(T + heat - cool - loss - evap
              <span className="text-rose-400 font-bold"> - 1.5*U</span> + (0.15+0.008*v)*eps_T, 0, 120)
            </div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm text-xs">
          <h4 className="font-bold text-sm text-navy-900 mb-2 border-b pb-1">
            Causal Identification & Structural Assumptions
          </h4>
          <ul className="space-y-2 text-slate-700 list-disc list-inside">
            <li>
              <span className="font-semibold text-slate-900">Confounded Observational Logs:</span> The operator reacts to ambient condition <span className="font-mono text-rose-600">U</span>:
              <div className="font-mono text-[11px] bg-slate-50 p-1.5 rounded mt-1 border">
                h = clip(0.55 + 0.25*U + 0.02*(55-T) + 0.05*n_h, 0, 1)
              </div>
            </li>
            <li>
              <span className="font-semibold text-slate-900">Randomized Probe Identification:</span> Probe experiments draw actions randomly independent of U, severing the <span className="font-mono text-rose-600">U -&gt; a</span> edge.
            </li>
            <li>
              <span className="font-semibold text-slate-900">Counterfactual Noise Invariance:</span> Process disturbances <span className="font-mono">eps_T, eps_L</span> are abducted via standardized residuals and preserved during replay.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
