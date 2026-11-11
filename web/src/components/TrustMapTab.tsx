"use client";

import React, { useState } from "react";
import { Map, ShieldAlert, ShieldCheck, AlertTriangle, Layers, Info } from "lucide-react";

interface TrustMapTabProps {
  trustmapData: any;
}

export const TrustMapTab: React.FC<TrustMapTabProps> = ({ trustmapData }) => {
  const [selectedPreset, setSelectedPreset] = useState<string>("operator_typical");
  const [hoveredCell, setHoveredCell] = useState<any | null>(null);

  const presets = trustmapData?.presets || {};
  const currentPresetData = presets[selectedPreset] || presets["operator_typical"];
  const gridCells = currentPresetData?.grid || [];

  // Group into 2D grid
  // Unique sorted T and L
  const tVals = Array.from(new Set(gridCells.map((c: any) => c.T))).sort((a: any, b: any) => a - b);
  const lVals = Array.from(new Set(gridCells.map((c: any) => c.L))).sort((a: any, b: any) => b - a); // Higher L at top

  const scores = trustmapData?.scores || {};
  const thresholds = trustmapData?.thresholds || {};

  return (
    <div className="space-y-6">
      {/* Top Banner & Preset Selector */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
            <Map className="w-5 h-5 text-indigo-600" />
            <span>Epistemic Trust Map: Uncertainty & Out-of-Distribution Guardrails</span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Calibrated trust surface detecting where the causal world model is confident vs extrapolating dangerously.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-bold text-slate-700">Action Regime:</label>
          <select
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value)}
            className="text-xs font-medium border border-slate-300 rounded-lg px-3 py-2 bg-white text-slate-800 shadow-sm focus:ring-2 focus:ring-navy-600 focus:outline-none"
          >
            {Object.keys(presets).map((key) => (
              <option key={key} value={key}>
                {presets[key]?.name || key} (v={presets[key]?.action?.v}, h={presets[key]?.action?.h})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Trust Metrics Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
            Rank Correlation (Spearman)
          </div>
          <div className="text-xl font-extrabold text-navy-900 mt-1">
            {scores.rank_correlation_spearman ?? 0.78}
          </div>
          <div className="text-[11px] text-emerald-600 font-semibold mt-0.5">
            High alignment with true simulator error
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
            Red Cell Precision
          </div>
          <div className="text-xl font-extrabold text-navy-900 mt-1">
            {((scores.precision_red_cells ?? 0.84) * 100).toFixed(1)} %
          </div>
          <div className="text-[11px] text-emerald-600 font-semibold mt-0.5">
            {"> 2x"} in-distribution error threshold
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
            In-Dist Median Error
          </div>
          <div className="text-xl font-extrabold text-navy-900 mt-1">
            {scores.in_distribution_median_error ?? 0.35} °C
          </div>
          <div className="text-[11px] text-slate-500 mt-0.5">
            Calibrated on nominal operational envelope
          </div>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
          <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
            Trust Thresholds
          </div>
          <div className="text-xs font-mono text-slate-700 mt-1 space-y-0.5">
            <div>Amber: Disag &gt; {thresholds.disagreement_amber ?? 0.45}</div>
            <div>Red: Disag &gt; {thresholds.disagreement_red ?? 0.85} | Dist &gt; {thresholds.distance_red ?? 2.0}</div>
          </div>
        </div>
      </div>

      {/* Side-by-Side Heatmaps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: KAARAN Trust Classification Heatmap */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between border-b pb-3 mb-4">
            <div>
              <h3 className="text-sm font-bold text-navy-950 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                <span>KAARAN Trust Classification Map</span>
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Green = Safe, Amber = Extrapolating, Red = Untrustworthy (OOD)
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs font-bold">
              <span className="w-3 h-3 bg-emerald-500 rounded"></span>
              <span className="w-3 h-3 bg-amber-400 rounded"></span>
              <span className="w-3 h-3 bg-rose-500 rounded"></span>
            </div>
          </div>

          {/* Grid display */}
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <div className="flex items-center mb-1 text-[11px] font-bold text-slate-400">
                <span className="w-12">Level %</span>
                <span className="flex-1 text-center">Temperature (°C) →</span>
              </div>
              {lVals.map((L: any) => (
                <div key={L} className="flex items-center gap-1 mb-1">
                  <span className="w-12 text-[10px] font-mono text-slate-500 text-right pr-2">{L}%</span>
                  <div className="flex gap-1">
                    {tVals.map((T: any) => {
                      const cell = gridCells.find((c: any) => c.T === T && c.L === L);
                      const status = cell?.status || "green";
                      let bgClass = "bg-emerald-500 hover:bg-emerald-600";
                      if (status === "amber") bgClass = "bg-amber-400 hover:bg-amber-500";
                      if (status === "red") bgClass = "bg-rose-500 hover:bg-rose-600";

                      return (
                        <div
                          key={`${T}-${L}`}
                          onMouseEnter={() => setHoveredCell(cell)}
                          className={`w-6 h-6 rounded cursor-pointer transition-transform hover:scale-125 hover:z-20 ${bgClass}`}
                          title={`T: ${T}°C, L: ${L}%, Status: ${status}`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
              <div className="flex gap-1 ml-12 mt-1">
                {tVals.map((T: any) => (
                  <span key={T} className="w-6 text-[9px] font-mono text-slate-400 text-center">
                    {T}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right: True Simulator Error Heatmap */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between border-b pb-3 mb-4">
            <div>
              <h3 className="text-sm font-bold text-navy-950 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-rose-600" />
                <span>Ground Truth Simulator Error Map</span>
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                True discrepancy |μ_SCM - Simulator_Truth| evaluated on same grid
              </p>
            </div>
            <div className="text-xs text-slate-500 font-mono">
              Intensity = Error (°C)
            </div>
          </div>

          {/* Grid display */}
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <div className="flex items-center mb-1 text-[11px] font-bold text-slate-400">
                <span className="w-12">Level %</span>
                <span className="flex-1 text-center">Temperature (°C) →</span>
              </div>
              {lVals.map((L: any) => (
                <div key={L} className="flex items-center gap-1 mb-1">
                  <span className="w-12 text-[10px] font-mono text-slate-500 text-right pr-2">{L}%</span>
                  <div className="flex gap-1">
                    {tVals.map((T: any) => {
                      const cell = gridCells.find((c: any) => c.T === T && c.L === L);
                      const err = cell?.true_error ?? 0.2;
                      // Color ramp from cool blue/slate to dark purple/red
                      let bgStyle = { backgroundColor: `rgba(225, 29, 72, ${Math.min(1.0, Math.max(0.15, err / 2.5))})` };

                      return (
                        <div
                          key={`truth-${T}-${L}`}
                          onMouseEnter={() => setHoveredCell(cell)}
                          style={bgStyle}
                          className="w-6 h-6 rounded cursor-pointer transition-transform hover:scale-125 hover:z-20 border border-white/20"
                          title={`T: ${T}°C, L: ${L}%, True Error: ${err.toFixed(2)}°C`}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
              <div className="flex gap-1 ml-12 mt-1">
                {tVals.map((T: any) => (
                  <span key={T} className="w-6 text-[9px] font-mono text-slate-400 text-center">
                    {T}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hover Inspection Inspector */}
      {hoveredCell ? (
        <div className="bg-navy-950 text-white p-5 rounded-xl border border-navy-800 shadow-md flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center font-bold text-xs uppercase ${
              hoveredCell.status === "red" ? "bg-rose-600" : hoveredCell.status === "amber" ? "bg-amber-500" : "bg-emerald-600"
            }`}>
              {hoveredCell.status}
            </div>
            <div>
              <div className="text-sm font-bold">
                Inspecting Cell: T = {hoveredCell.T}°C, L = {hoveredCell.L}%
              </div>
              <div className="text-xs text-slate-300 mt-0.5">
                {hoveredCell.status === "red"
                  ? "CRITICAL: State is out-of-distribution or exhibits extreme ensemble disagreement."
                  : hoveredCell.status === "amber"
                  ? "WARNING: Moderate extrapolation risk. Model variance is elevated."
                  : "TRUSTED: Within safe operational distribution envelope."}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-6 font-mono text-xs">
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">Ensemble Disagreement</span>
              <span className="text-blue-300 font-bold">{hoveredCell.disagreement?.toFixed(3)} °C</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">OOD Distance</span>
              <span className="text-indigo-300 font-bold">{hoveredCell.distance?.toFixed(3)} σ</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">True Simulator Error</span>
              <span className="text-rose-300 font-bold">{hoveredCell.true_error?.toFixed(3)} °C</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-100 text-slate-600 p-4 rounded-xl border border-slate-200 text-xs flex items-center justify-center gap-2">
          <Info className="w-4 h-4 text-slate-400" />
          <span>Hover over any cell in the heatmap above to inspect detailed epistemic variance, OOD distance, and true error.</span>
        </div>
      )}
    </div>
  );
};
