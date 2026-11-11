"use client";

import React from "react";
import { 
  Sliders, 
  Activity, 
  GitFork, 
  Map, 
  Table2, 
  AlertTriangle, 
  FileText,
  PlayCircle,
  Database
} from "lucide-react";

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isMockData: boolean;
  onOpenGuidedDemo: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  isMockData,
  onOpenGuidedDemo,
}) => {
  const navItems = [
    { id: "spec", label: "Spec & Graph", icon: FileText },
    { id: "forecast_action", label: "Forecast & Action", icon: Activity },
    { id: "intervene", label: "Intervene do(a)", icon: Sliders },
    { id: "what_if", label: "What-If (Counterfactual)", icon: GitFork },
    { id: "trust_map", label: "Trust Map", icon: Map },
    { id: "scorecard", label: "Scorecard & Demos", icon: Table2 },
    { id: "limits", label: "Limitations", icon: AlertTriangle },
  ];

  return (
    <header className="sticky top-0 z-30 bg-white border-b border-slate-200 shadow-sm">
      {isMockData && (
        <div className="bg-amber-500 text-white text-xs font-semibold px-4 py-1.5 flex items-center justify-center gap-2">
          <Database className="w-4 h-4" />
          <span>PRE-TRAINING MOCK DATA ACTIVE — Model training pipeline is executing in background.</span>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          <div>
            <div className="flex items-center gap-3">
              <span className="text-2xl font-black tracking-tight text-navy-900 bg-navy-100 text-navy-800 px-2.5 py-0.5 rounded border border-navy-300">
                KAARAN
              </span>
              <span className="text-xs uppercase tracking-wider font-bold text-navy-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                Causal World Model
              </span>
            </div>
            <p className="text-xs font-medium text-slate-500 mt-1">
              Ground truth comes from the simulator and is <span className="font-bold text-navy-900">never shown to the model</span>.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onOpenGuidedDemo}
              className="flex items-center gap-2 bg-navy-800 hover:bg-navy-900 text-white font-medium text-xs px-3.5 py-2 rounded-lg shadow-sm transition-all border border-navy-700 hover:shadow"
            >
              <PlayCircle className="w-4 h-4 text-blue-400" />
              <span>Guided Demo (6 Moments)</span>
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 overflow-x-auto border-t border-slate-100 pt-1 pb-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-md transition-colors whitespace-nowrap ${
                  isActive
                    ? "bg-navy-900 text-white shadow-sm"
                    : "text-slate-600 hover:text-navy-900 hover:bg-slate-100"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-blue-300" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
