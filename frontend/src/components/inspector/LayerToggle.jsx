import ScoreBar from "../common/ScoreBar";

export default function LayerToggle({ layerKey, meta, enabled, onChange, liveResult }) {
  const layerData  = liveResult?.layers?.[layerKey];
  const hasResult  = !!layerData;
  const score      = layerData?.score;
  const decision   = layerData?.decision;

  const dotCls = hasResult
    ? decision === "BLOCK"
      ? "bg-red-500 animate-pulse"
      : decision === "SUSPICIOUS"
        ? "bg-yellow-400"
        : "bg-green-500"
    : "bg-slate-600";

  return (
    <div className={`flex items-start gap-3 py-3 border-b border-slate-700/50 last:border-0 ${!enabled ? "opacity-60" : ""}`}>
      <div className="mt-0.5">
        <span className={`w-2.5 h-2.5 rounded-full inline-block ${dotCls}`} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className="text-xs text-slate-500 font-mono">{meta.label}</span>
          <span className="text-sm font-medium text-slate-200">{meta.name}</span>
        </div>
        <div className="text-xs text-slate-500 mt-0.5">{meta.desc}</div>
        {hasResult && score !== undefined && (
          <div className="mt-1 flex items-center gap-2">
            <ScoreBar score={score} mini />
            <span className="text-xs font-mono text-slate-400">{(score || 0).toFixed(3)}</span>
          </div>
        )}
      </div>
      <button
        onClick={() => onChange(layerKey, !enabled)}
        className={`relative w-10 h-5 rounded-full transition-colors duration-200 flex-shrink-0 mt-0.5 ${enabled ? "bg-indigo-600" : "bg-slate-700"}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${enabled ? "translate-x-5" : "translate-x-0"}`}
        />
      </button>
    </div>
  );
}
