"use client";

import React, { useState, useEffect } from "react";
import { Header } from "../components/Header";
import { SpecTab } from "../components/SpecTab";
import { ForecastActionTab } from "../components/ForecastActionTab";
import { InterveneTab } from "../components/InterveneTab";
import { WhatIfTab } from "../components/WhatIfTab";
import { TrustMapTab } from "../components/TrustMapTab";
import { ScorecardTab } from "../components/ScorecardTab";
import { LimitsTab } from "../components/LimitsTab";
import { GuidedDemoModal } from "../components/GuidedDemoModal";

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>("intervene");
  const [isDemoModalOpen, setIsDemoModalOpen] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Artifact state containers
  const [specData, setSpecData] = useState<any>(null);
  const [episodesData, setEpisodesData] = useState<any>(null);
  const [metricsData, setMetricsData] = useState<any>(null);
  const [calibrationData, setCalibrationData] = useState<any>(null);
  const [interventionsEval, setInterventionsEval] = useState<any>(null);
  const [counterfactualEval, setCounterfactualEval] = useState<any>(null);
  const [lowData, setLowData] = useState<any>(null);
  const [missingObsData, setMissingObsData] = useState<any>(null);
  const [confoundingData, setConfoundingData] = useState<any>(null);
  const [trustmapData, setTrustmapData] = useState<any>(null);
  const [dowhyData, setDowhyData] = useState<any>(null);
  const [weightsForecast, setWeightsForecast] = useState<any>(null);
  const [weightsAction, setWeightsAction] = useState<any>(null);

  useEffect(() => {
    async function loadArtifacts() {
      try {
        const fetchJson = async (filename: string) => {
          const res = await fetch(`/artifacts/${filename}`);
          if (!res.ok) {
            console.warn(`Artifact /artifacts/${filename} not found`);
            return null;
          }
          return res.json();
        };

        const [
          spec,
          episodes,
          metrics,
          calib,
          interv,
          cf,
          low,
          miss,
          conf,
          trust,
          dowhy,
          wf,
          wa,
        ] = await Promise.all([
          fetchJson("spec.json"),
          fetchJson("episodes_test.json"),
          fetchJson("metrics.json"),
          fetchJson("calibration.json"),
          fetchJson("interventions_eval.json"),
          fetchJson("counterfactual_eval.json"),
          fetchJson("lowdata.json"),
          fetchJson("missing_obs.json"),
          fetchJson("confounding.json"),
          fetchJson("trustmap.json"),
          fetchJson("dowhy.json"),
          fetchJson("weights_forecast.json"),
          fetchJson("weights_action.json"),
        ]);

        setSpecData(spec);
        setEpisodesData(episodes);
        setMetricsData(metrics);
        setCalibrationData(calib);
        setInterventionsEval(interv);
        setCounterfactualEval(cf);
        setLowData(low);
        setMissingObsData(miss);
        setConfoundingData(conf);
        setTrustmapData(trust);
        setDowhyData(dowhy);
        setWeightsForecast(wf);
        setWeightsAction(wa);
      } catch (err) {
        console.error("Error loading artifacts:", err);
      } finally {
        setLoading(false);
      }
    }

    loadArtifacts();
  }, []);

  const isMockData =
    specData?.mock === true ||
    metricsData?.mock === true ||
    interventionsEval?.mock === true;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isMockData={isMockData}
        onOpenGuidedDemo={() => setIsDemoModalOpen(true)}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {loading ? (
          <div className="flex flex-col items-center justify-center h-64 space-y-3">
            <div className="w-8 h-8 border-4 border-navy-800 border-t-transparent rounded-full animate-spin"></div>
            <div className="text-xs font-semibold text-slate-500">
              Loading KAARAN artifacts...
            </div>
          </div>
        ) : (
          <div>
            {activeTab === "spec" && <SpecTab specData={specData} />}
            {activeTab === "forecast_action" && (
              <ForecastActionTab
                episodesData={episodesData}
                weightsForecast={weightsForecast}
                weightsAction={weightsAction}
              />
            )}
            {activeTab === "intervene" && (
              <InterveneTab interventionsEval={interventionsEval} />
            )}
            {activeTab === "what_if" && (
              <WhatIfTab
                counterfactualEval={counterfactualEval}
                episodesData={episodesData}
              />
            )}
            {activeTab === "trust_map" && (
              <TrustMapTab trustmapData={trustmapData} />
            )}
            {activeTab === "scorecard" && (
              <ScorecardTab
                metricsData={metricsData}
                calibrationData={calibrationData}
                lowData={lowData}
                missingObsData={missingObsData}
                confoundingData={confoundingData}
                dowhyData={dowhyData}
              />
            )}
            {activeTab === "limits" && (
              <LimitsTab
                trustmapData={trustmapData}
                confoundingData={confoundingData}
              />
            )}
          </div>
        )}
      </main>

      {/* Guided Presentation Modal */}
      <GuidedDemoModal
        isOpen={isDemoModalOpen}
        onClose={() => setIsDemoModalOpen(false)}
        onSelectTab={(tab) => {
          setActiveTab(tab);
        }}
      />

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500 flex flex-wrap items-center justify-between gap-2">
          <div>
            <span className="font-bold text-navy-900">KAARAN</span> — Uncertainty-Aware Causal World Model for Industrial Process Control
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            PyTorch Deep Ensembles | Gymnasium Env | Client-Side TypeScript SCM Engine
          </div>
        </div>
      </footer>
    </div>
  );
}
