import { LAYER_WEIGHTS } from "../../constants";

export default function WeightPreview({ layers }) {
  const enabled = Object.entries(LAYER_WEIGHTS).filter(([k]) => layers[k] !== false);
  const total   = enabled.reduce((s, [, w]) => s + w, 0);
  const pct     = total > 0 ? Math.round(total * 100) : 0;

  return (
    <div className="mt-4 p-3 bg-slate-900 rounded-lg border border-slate-700">
      <div className="text-xs text-slate-400 mb-2">Fusion weight coverage</div>
      <div className="flex items-center gap-2">
        <div className="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
          <div
            className="h-2 rounded-full bg-indigo-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-xs font-mono text-slate-300 w-8 text-right">{pct}%</span>
      </div>
      <div className="mt-1 text-xs text-slate-600">{enabled.map(([k]) => k).join(" · ")}</div>
    </div>
  );
}
