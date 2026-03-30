import ScoreBar from "../common/ScoreBar";
import { actionColor, actionBg } from "../../utils/helpers";

export default function FinalCard({ result }) {
  if (!result) return null;
  const { final_action, final_score, latency_ms, request_id, layers } = result;
  const weights = layers?.layer3?.weights || {};

  return (
    <div className={`rounded-lg border-2 p-4 ${actionBg(final_action)}`}>
      <div className="flex items-center justify-between mb-3">
        <span className="font-bold text-base text-slate-100">FINAL DECISION</span>
        <span className="text-xs text-slate-400 font-mono">{latency_ms} ms</span>
      </div>
      <div className={`text-3xl font-black mb-3 ${actionColor(final_action)}`}>{final_action}</div>
      <ScoreBar score={final_score} />
      <div className="mt-3 text-xs space-y-1">
        <div className="flex justify-between">
          <span className="text-slate-400">Fused score</span>
          <span className="font-mono text-slate-200">{(final_score || 0).toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-400">Request ID</span>
          <span className="font-mono text-slate-400 text-xs truncate max-w-[180px]">{request_id}</span>
        </div>
        {result.short_circuited && (
          <div className="flex justify-between">
            <span className="text-slate-400">Short-circuit at</span>
            <span className="text-red-400 font-mono">{result.short_circuit_at}</span>
          </div>
        )}
      </div>
      {Object.keys(weights).filter((k) => weights[k] > 0).length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-700">
          <div className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Active weights</div>
          <div className="grid grid-cols-2 gap-x-4 text-xs">
            {Object.entries(weights)
              .filter(([, w]) => w > 0)
              .map(([k, w]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-slate-500">{k}</span>
                  <span className="font-mono text-slate-300">{(w * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
