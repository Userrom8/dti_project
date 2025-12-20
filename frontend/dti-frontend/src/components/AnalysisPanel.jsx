import {
  Activity,
  ThumbsUp,
  ThumbsDown,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";

export default function AnalysisPanel({ affinity, drugName }) {
  // Determine Strength (Assuming pKd/pIC50 scale: <5 weak, 5-7 moderate, >7 strong)
  // You can adjust these thresholds based on your specific model's output range.
  let strength = "Moderate";
  let color = "text-yellow-400";
  let bgColor = "bg-yellow-400/10";
  let width = "50%";

  if (affinity > 7.0) {
    strength = "Strong";
    color = "text-green-400";
    bgColor = "bg-green-400/10";
    width = "85%";
  } else if (affinity < 5.0) {
    strength = "Weak";
    color = "text-red-400";
    bgColor = "bg-red-400/10";
    width = "25%";
  }

  return (
    <div className="space-y-6 mt-6 pt-6 border-t border-slate-800">
      {/* --- Score Interpretation --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-xl border border-slate-800 ${bgColor}`}>
          <div className="flex items-center gap-2 mb-2">
            <Activity className={`w-5 h-5 ${color}`} />
            <h4 className="font-semibold text-slate-200">Binding Strength</h4>
          </div>
          <div className={`text-2xl font-bold ${color} mb-1`}>
            {strength} Interaction
          </div>
          <div className="w-full bg-slate-800 h-2 rounded-full mt-3 overflow-hidden">
            <div
              className={`h-full ${color.replace(
                "text",
                "bg"
              )} transition-all duration-1000 ease-out`}
              style={{ width: width }}
            />
          </div>
          <p className="text-xs text-slate-400 mt-2">
            Based on predicted affinity score ({affinity.toFixed(2)})
          </p>
        </div>

        {/* --- Contextual Hints --- */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/50">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            <h4 className="font-semibold text-slate-200">
              Pharmacological Context
            </h4>
          </div>
          <p className="text-sm text-slate-400 leading-relaxed">
            {strength === "Strong"
              ? "High binding affinity typically suggests the drug can effectively occupy the target receptor even at low concentrations, potentially reducing the required dosage."
              : strength === "Weak"
              ? "Low affinity suggests the drug may detach easily from the target or require very high concentrations to be effective, increasing the risk of off-target side effects."
              : "Moderate affinity indicates a stable interaction, but optimization might be needed to improve potency or selectivity."}
          </p>
        </div>
      </div>

      {/* --- Pros & Cons (Generic/Template) --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Potential Advantages */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2 mb-3 text-green-400">
            <ThumbsUp className="w-4 h-4" />
            <span className="font-semibold text-sm uppercase tracking-wide">
              Potential Advantages
            </span>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {strength === "Strong" ? (
              <>
                <li className="flex gap-2">
                  <span className="text-green-500">•</span> High potency likely
                  at low doses
                </li>
                <li className="flex gap-2">
                  <span className="text-green-500">•</span> Prolonged receptor
                  residency time
                </li>
                <li className="flex gap-2">
                  <span className="text-green-500">•</span> Strong thermodynamic
                  stability
                </li>
              </>
            ) : (
              <>
                <li className="flex gap-2">
                  <span className="text-green-500">•</span> Easier to reverse
                  (less toxic potential)
                </li>
                <li className="flex gap-2">
                  <span className="text-green-500">•</span> Lower risk of
                  irreversible binding
                </li>
              </>
            )}
          </ul>
        </div>

        {/* Potential Disadvantages */}
        <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-2 mb-3 text-red-400">
            <ThumbsDown className="w-4 h-4" />
            <span className="font-semibold text-sm uppercase tracking-wide">
              Challenges
            </span>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {strength === "Strong" ? (
              <>
                <li className="flex gap-2">
                  <span className="text-red-500">•</span> Risk of off-target
                  toxicity if not selective
                </li>
                <li className="flex gap-2">
                  <span className="text-red-500">•</span> Harder to clear from
                  system
                </li>
              </>
            ) : (
              <>
                <li className="flex gap-2">
                  <span className="text-red-500">•</span> High dose may be
                  required
                </li>
                <li className="flex gap-2">
                  <span className="text-red-500">•</span> Likely low therapeutic
                  efficacy
                </li>
                <li className="flex gap-2">
                  <span className="text-red-500">•</span> Rapid dissociation
                  from target
                </li>
              </>
            )}
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-900/10 border border-blue-900/30 text-xs text-blue-200/70">
        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
        <p>
          <strong>Note:</strong> This analysis is based solely on the predicted
          binding affinity score. Real-world efficacy depends on ADMET
          properties (Absorption, Distribution, Metabolism, Excretion, Toxicity)
          which are not calculated by this specific model.
        </p>
      </div>
    </div>
  );
}
