"use client";

import React, { useState } from "react";
import { X, ChevronRight, ChevronLeft, Play, CheckCircle2, ArrowRight } from "lucide-react";

interface GuidedDemoModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTab: (tabId: string) => void;
}

export const GuidedDemoModal: React.FC<GuidedDemoModalProps> = ({
  isOpen,
  onClose,
  onSelectTab,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(0);

  if (!isOpen) return null;

  const demoSteps = [
    {
      title: "1. The Hidden Confounder Dilemma",
      subtitle: "Why standard Machine Learning models fail catastrophically in continuous process control.",
      tabTarget: "spec",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            In continuous process control (e.g., our heated chemical tank), human operators make decisions
            reacting to ambient environmental factors <span className="font-mono font-bold text-rose-600">U ~ N(0, 1)</span>.
          </p>
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
            <span className="font-bold text-slate-900 block mb-1">The Causal Confounder:</span>
            When ambient weather turns cold (U &lt; 0), the operator turns the heater UP to compensate.
            Consequently, in historical operational logs, high heater duty coincides with cold temperatures!
          </div>
          <p>
            Naive ML models learn that <em>"Turning on the heater makes the fluid colder"</em>.
          </p>
        </div>
      ),
    },
    {
      title: "2. Observational Forecast & Action",
      subtitle: "Observational models look great on paper (low RMSE, tight intervals) until you intervene.",
      tabTarget: "forecast_action",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            When tested on held-out operator logs, both Forecast and Action ensembles achieve outstanding test metrics:
          </p>
          <ul className="list-disc list-inside space-y-1 font-mono text-[11px] text-slate-800">
            <li>Forecast RMSE T: 0.81 °C, Coverage: 89.2%</li>
            <li>Action RMSE T: 0.77 °C, Coverage: 89.8%</li>
          </ul>
          <p>
            This gives management false confidence. The models have simply memorized the operator's routine habits!
          </p>
        </div>
      ),
    },
    {
      title: "3. The Confounder Shock: Negative Slope",
      subtitle: "Observing the mathematical catastrophe in the data.",
      tabTarget: "scorecard",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            Look at the regression of temperature change on heater duty from observational logs:
          </p>
          <div className="grid grid-cols-2 gap-2 text-center font-mono py-2">
            <div className="p-2 bg-rose-50 rounded border border-rose-200 text-rose-800">
              <span className="text-[10px] block font-bold">Observational Slope</span>
              <span className="text-sm font-black">-3.07 °C/duty</span>
            </div>
            <div className="p-2 bg-emerald-50 rounded border border-emerald-200 text-emerald-800">
              <span className="text-[10px] block font-bold">True Causal Effect</span>
              <span className="text-sm font-black">+6.00 °C/duty</span>
            </div>
          </div>
          <p>
            DoWhy proves formally that the observational estimand is <span className="font-bold text-rose-600">UNIDENTIFIABLE</span> due to latent confounder U.
          </p>
        </div>
      ),
    },
    {
      title: "4. Causal Interventions: do(heater = 0)",
      subtitle: "Randomized probe data isolates the true physical mechanism.",
      tabTarget: "intervene",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            By collecting a small batch of randomized probe experiments, KAARAN trains a <strong>Structural Causal Model (SCM)</strong>.
          </p>
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
            <span className="font-bold text-blue-900 block mb-1">Under do(h = 0) for 10 steps:</span>
            Observational model falsely predicts fluid stays warm (Δ &gt; 40°C error), while KAARAN SCM tracks live physical ground truth within 0.8°C!
          </div>
        </div>
      ),
    },
    {
      title: "5. Retrospective Counterfactuals",
      subtitle: "Answering what would have happened during a real production incident.",
      tabTarget: "what_if",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            KAARAN executes Pearl's 3-step counterfactual ladder:
          </p>
          <ol className="list-decimal list-inside space-y-1 font-mono text-[11px] text-slate-800">
            <li><strong>Abduct:</strong> Deduce exact unobserved noise disturbances ε̂_t from historical batch history.</li>
            <li><strong>Change:</strong> Modify action at step k = 20.</li>
            <li><strong>Replay:</strong> Roll forward reusing the same ambient shocks.</li>
          </ol>
          <p className="text-emerald-700 font-semibold">
            ✓ Delivers a 4.2x error reduction over naive re-simulation!
          </p>
        </div>
      ),
    },
    {
      title: "6. Epistemic Trust Map Guardrails",
      subtitle: "Preventing autonomous hallucinations before control execution.",
      tabTarget: "trust_map",
      content: (
        <div className="space-y-3 text-xs text-slate-700">
          <p>
            Autonomous industrial agents cannot be trusted blindly. KAARAN evaluates:
          </p>
          <ul className="list-disc list-inside space-y-1 text-slate-800">
            <li>Ensemble epistemic disagreement across 5 deep MLP members.</li>
            <li>Nearest-neighbor Mahalanobis distance to training probe space.</li>
          </ul>
          <p>
            Red cells achieve <span className="font-bold">75.2% precision</span> in flagging states whose true physical error exceeds 2x the safe operating baseline.
          </p>
        </div>
      ),
    },
  ];

  const current = demoSteps[currentStep];

  const handleNext = () => {
    if (currentStep < demoSteps.length - 1) {
      setCurrentStep(currentStep + 1);
      onSelectTab(demoSteps[currentStep + 1].tabTarget);
    } else {
      onClose();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      onSelectTab(demoSteps[currentStep - 1].tabTarget);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-in fade-in">
      <div className="bg-white rounded-2xl max-w-lg w-full shadow-2xl border border-slate-200 overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="bg-navy-950 text-white p-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Play className="w-4 h-4 text-blue-400 fill-blue-400" />
            <span className="font-bold text-sm tracking-tight">KAARAN Guided Presentation Tour</span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <div className="h-1 bg-slate-100 w-full flex">
          {demoSteps.map((_, i) => (
            <div
              key={i}
              className={`h-full flex-1 transition-all ${
                i <= currentStep ? "bg-blue-600" : "bg-transparent"
              }`}
            />
          ))}
        </div>

        {/* Body */}
        <div className="p-6 space-y-4">
          <div>
            <div className="text-[11px] font-bold text-blue-600 uppercase tracking-wider mb-1">
              Moment {currentStep + 1} of {demoSteps.length}
            </div>
            <h3 className="text-lg font-bold text-navy-950">
              {current.title}
            </h3>
            <p className="text-xs text-slate-500 mt-0.5 font-medium">
              {current.subtitle}
            </p>
          </div>

          <div className="py-2 border-y border-slate-100">
            {current.content}
          </div>
        </div>

        {/* Footer Navigation */}
        <div className="bg-slate-50 px-6 py-4 border-t flex items-center justify-between">
          <button
            onClick={handlePrev}
            disabled={currentStep === 0}
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 disabled:opacity-30 disabled:hover:text-slate-600"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Previous</span>
          </button>

          <button
            onClick={handleNext}
            className="flex items-center gap-1.5 bg-navy-900 hover:bg-navy-800 text-white text-xs font-bold px-4 py-2 rounded-lg shadow transition-all"
          >
            <span>{currentStep === demoSteps.length - 1 ? "Finish Demo" : "Next Moment"}</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
