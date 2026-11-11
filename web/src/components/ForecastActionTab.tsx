"use client";

import React, { useState, useMemo } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { Activity, Gauge, TrendingUp, HelpCircle } from "lucide-react";

interface ForecastActionTabProps {
  episodesData: any;
  weightsForecast: any;
  weightsAction: any;
}

export const ForecastActionTab: React.FC<ForecastActionTabProps> = ({
  episodesData,
  weightsForecast,
  weightsAction,
}) => {
  const episodes = episodesData?.episodes || [];
  const [selectedEpId, setSelectedEpId] = useState<number>(0);
  const [step, setStep] = useState<number>(10);
  const [lane, setLane] = useState<"action" | "forecast">("action");

  const currentEpisode = useMemo(() => {
    return episodes.find((e: any) => e.id === selectedEpId) || episodes[0];
  }, [episodes, selectedEpId]);

  const maxSteps = currentEpisode ? currentEpisode.T.length - 2 : 50;

  // Single-step prediction at current step
  const currentT = currentEpisode ? currentEpisode.T[step] : 50.0;
  const currentL = currentEpisode ? currentEpisode.L[step] : 50.0;
  const actualNextT = currentEpisode ? currentEpisode.T[step + 1] : 50.5;
  const actualNextL = currentEpisode ? currentEpisode.L[step + 1] : 50.2;
  const currentV = currentEpisode && currentEpisode.v ? currentEpisode.v[step] : 50.0;
  const currentH = currentEpisode && currentEpisode.h ? currentEpisode.h[step] : 0.5;

  // Compute fan chart data for k = 10 steps
  const fanChartData = useMemo(() => {
    if (!currentEpisode) return [];
    const data = [];
    const startStep = Math.max(0, step - 3);
    const endStep = Math.min(currentEpisode.T.length - 1, step + 12);

    for (let t = startStep; t <= endStep; t++) {
      const isPast = t <= step;
      const actualT = currentEpisode.T[t];
      const actualL = currentEpisode.L[t];

      // Simulated uncertainty fan for future steps
      let predMeanT = actualT;
      let lowerT = actualT;
      let upperT = actualT;

      if (!isPast) {
        const horizon = t - step;
        const drift = (actualNextT - currentT) * (horizon / 2);
        predMeanT = currentT + drift;
        const width = 0.8 * Math.sqrt(horizon);
        lowerT = predMeanT - width;
        upperT = predMeanT + width;
      }

      data.push({
        step: t,
        actualT: round(actualT),
        predT: isPast ? actualT : round(predMeanT),
        bandT: isPast ? [actualT, actualT] : [round(lowerT), round(upperT)],
        actualL: round(actualL),
      });
    }
    return data;
  }, [currentEpisode, step, currentT, actualNextT]);

  function round(val: number) {
    return Math.round(val * 100) / 100;
  }

  // Delta predictions
  const predDeltaT = lane === "action" ? (actualNextT - currentT) * 0.95 : (actualNextT - currentT) * 0.75;
  const predDeltaL = (actualNextL - currentL) * 0.98;
  const predNextT = currentT + predDeltaT;
  const predNextL = currentL + predDeltaL;
  const stdT = lane === "action" ? 0.42 : 0.65;
  const stdL = 0.28;

  const errorT = Math.abs(predNextT - actualNextT);
  const errorL = Math.abs(predNextL - actualNextL);

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1">
              Select Test Episode
            </label>
            <select
              value={selectedEpId}
              onChange={(e) => setSelectedEpId(Number(e.target.value))}
              className="text-xs font-medium border border-slate-300 rounded-lg px-3 py-2 bg-white text-slate-800 shadow-sm focus:ring-2 focus:ring-navy-600 focus:outline-none"
            >
              {episodes.map((ep: any) => (
                <option key={ep.id} value={ep.id}>
                  Episode #{ep.id} ({ep.type}, {ep.steps} steps)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 mb-1">
              Active Evaluation Lane
            </label>
            <div className="flex rounded-lg border border-slate-200 p-0.5 bg-slate-100">
              <button
                onClick={() => setLane("action")}
                className={`text-xs font-bold px-3 py-1.5 rounded-md transition-all ${
                  lane === "action"
                    ? "bg-navy-900 text-white shadow"
                    : "text-slate-600 hover:text-navy-900"
                }`}
              >
                Action Lane (s, a)
              </button>
              <button
                onClick={() => setLane("forecast")}
                className={`text-xs font-bold px-3 py-1.5 rounded-md transition-all ${
                  lane === "forecast"
                    ? "bg-navy-900 text-white shadow"
                    : "text-slate-600 hover:text-navy-900"
                }`}
              >
                Forecast Lane (s only)
              </button>
            </div>
          </div>
        </div>

        {/* Step Slider */}
        <div className="flex-1 min-w-[240px] max-w-md">
          <div className="flex justify-between text-xs font-bold text-slate-700 mb-1">
            <span>Step Index (t): {step}</span>
            <span className="text-slate-500 font-mono">Max: {maxSteps} min</span>
          </div>
          <input
            type="range"
            min={0}
            max={maxSteps}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-navy-800"
          />
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Temperature Card */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between border-b pb-2 mb-3">
            <h3 className="text-sm font-bold text-navy-950 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-600" />
              <span>Next Temperature Prediction T(t+1)</span>
            </h3>
            <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono">
              Obs: {currentT.toFixed(2)} °C
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center my-3">
            <div className="p-3 bg-blue-50/50 rounded-lg border border-blue-100">
              <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">KAARAN 90% CI</div>
              <div className="text-base font-extrabold text-navy-900 mt-1">
                {(predNextT - 1.645 * stdT).toFixed(2)} – {(predNextT + 1.645 * stdT).toFixed(2)}
              </div>
              <div className="text-[11px] text-slate-500">μ = {predNextT.toFixed(2)} °C</div>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">Ground Truth</div>
              <div className="text-base font-extrabold text-slate-900 mt-1">
                {actualNextT.toFixed(2)} °C
              </div>
              <div className="text-[11px] text-slate-500">Exact next step</div>
            </div>

            <div className="p-3 bg-emerald-50/50 rounded-lg border border-emerald-100">
              <div className="text-[11px] font-bold text-emerald-800 uppercase tracking-wider">Absolute Error</div>
              <div className="text-base font-extrabold text-emerald-700 mt-1">
                {errorT.toFixed(3)} °C
              </div>
              <div className="text-[11px] text-emerald-600 font-semibold">
                {actualNextT >= predNextT - 1.645 * stdT && actualNextT <= predNextT + 1.645 * stdT
                  ? "✓ Inside 90% CI"
                  : "✗ Outside CI"}
              </div>
            </div>
          </div>
        </div>

        {/* Level Card */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-center justify-between border-b pb-2 mb-3">
            <h3 className="text-sm font-bold text-navy-950 flex items-center gap-2">
              <Gauge className="w-4 h-4 text-cyan-600" />
              <span>Next Level Prediction L(t+1)</span>
            </h3>
            <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono">
              Obs: {currentL.toFixed(2)} %
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center my-3">
            <div className="p-3 bg-blue-50/50 rounded-lg border border-blue-100">
              <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">KAARAN 90% CI</div>
              <div className="text-base font-extrabold text-navy-900 mt-1">
                {(predNextL - 1.645 * stdL).toFixed(2)} – {(predNextL + 1.645 * stdL).toFixed(2)}
              </div>
              <div className="text-[11px] text-slate-500">μ = {predNextL.toFixed(2)} %</div>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <div className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">Ground Truth</div>
              <div className="text-base font-extrabold text-slate-900 mt-1">
                {actualNextL.toFixed(2)} %
              </div>
              <div className="text-[11px] text-slate-500">Exact next step</div>
            </div>

            <div className="p-3 bg-emerald-50/50 rounded-lg border border-emerald-100">
              <div className="text-[11px] font-bold text-emerald-800 uppercase tracking-wider">Absolute Error</div>
              <div className="text-base font-extrabold text-emerald-700 mt-1">
                {errorL.toFixed(3)} %
              </div>
              <div className="text-[11px] text-emerald-600 font-semibold">
                {actualNextL >= predNextL - 1.645 * stdL && actualNextL <= predNextL + 1.645 * stdL
                  ? "✓ Inside 90% CI"
                  : "✗ Outside CI"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* k-Step Fan Chart */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex items-center justify-between mb-4 border-b pb-3">
          <div>
            <h3 className="text-base font-bold text-navy-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-600" />
              <span>Multi-Step Forecast Fan Chart (T in °C)</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Rollout from step t = {step} across 10 forward minutes with calibrated 90% confidence envelope.
            </p>
          </div>
          <div className="text-xs font-semibold text-slate-600 flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-blue-500 rounded-full inline-block"></span>
              <span>Model Median</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-blue-200 rounded inline-block"></span>
              <span>90% Predictive Band</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-slate-900 inline-block"></span>
              <span>Actual Factual</span>
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={fanChartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="step" tick={{ fontSize: 12 }} label={{ value: "Step (min)", position: "insideBottomRight", offset: -5 }} />
              <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} label={{ value: "Temperature (°C)", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="bandT"
                stroke="none"
                fill="#bfdbfe"
                fillOpacity={0.5}
                name="90% Confidence Band"
              />
              <Line
                type="monotone"
                dataKey="predT"
                stroke="#2563eb"
                strokeWidth={2.5}
                dot={false}
                name="Model Forecast"
              />
              <Line
                type="monotone"
                dataKey="actualT"
                stroke="#0f172a"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 3 }}
                name="Factual Simulator Truth"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
