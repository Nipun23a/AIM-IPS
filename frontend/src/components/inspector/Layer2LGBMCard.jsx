import ScoreBar from "../common/ScoreBar";
import Badge from "../common/Badge";
import { decisionBorderColor, decisionDot } from "../../utils/helpers";

export default function Layer2LGBMCard({ data }) {
  if (!data) return null;
  const { enabled, ran, skipped_reason, score, label, all_probs } = data;
  const skipped          = !enabled;
  const wasShortCircuited = !ran && skipped_reason === "short_circuited";
  const wasCNNGated       = !ran && skipped_reason === "cnn_gate_clean";
  const decision          = !ran ? "CLEAN" : score > 0.5 ? "BLOCK" : score > 0.25 ? "SUSPICIOUS" : "CLEAN";
  const topProbs          = all_probs && Object.keys(all_probs).length > 0
    ? Object.entries(all_probs).sort((a, b) => b[1] - a[1]).slice(0, 4)
    : [];

  const borderCls = wasCNNGated
    ? "border-green-800"
    : wasShortCircuited || !ran
      ? "border-slate-600"
      : decisionBorderColor(decision, enabled);

  return (
    <div className={`rounded-lg border-l-4 bg-slate-800/60 border border-slate-700/50 p-3 ${borderCls} ${skipped ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm text-slate-200">Layer 2a — LightGBM</span>
        <div className="flex gap-1 items-center">
          {skipped && <Badge text="DISABLED" action="DISABLED" />}
          {wasShortCircuited && !skipped && <Badge text="SKIPPED"  action="SKIPPED" />}
          {wasCNNGated       && !skipped && <Badge text="CNN GATE" action="ALLOW"   />}
          {ran               && !skipped && <Badge text={decision} action={decision === "BLOCK" ? "BLOCK" : decision === "SUSPICIOUS" ? "DELAY" : "ALLOW"} />}
          {!ran && skipped_reason && !["short_circuited", "cnn_gate_clean"].includes(skipped_reason) && (
            <Badge text={skipped_reason} action="DISABLED" />
          )}
        </div>
      </div>

      {wasShortCircuited && !skipped ? (
        <div className="text-xs text-slate-500 italic">Short-circuited by Layer 0/1</div>
      ) : wasCNNGated && !skipped ? (
        <div className="flex items-center gap-1.5 text-xs text-green-600 mt-1">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block" />
          CNN gate passed clean — LightGBM not needed
        </div>
      ) : !ran ? (
        <div className="text-xs text-slate-500 italic">{skipped_reason || "not run"}</div>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-2">
            <span className={`w-2 h-2 rounded-full ${decisionDot(decision, enabled)}`} />
            <span className="text-xs text-slate-400">{label || "—"}</span>
            <span className="ml-auto text-xs text-slate-400">score=</span>
            <span className="text-xs font-mono text-slate-200">{(score || 0).toFixed(3)}</span>
            <ScoreBar score={score} mini />
          </div>
          <ScoreBar score={score} />
          {topProbs.length > 0 && (
            <div className="mt-2 grid grid-cols-2 gap-x-4 text-xs">
              {topProbs.map(([cls, prob]) => (
                <div key={cls} className="flex justify-between">
                  <span className="text-slate-500">{cls}</span>
                  <span className={`font-mono ${prob > 0.5 ? "text-red-400" : "text-slate-300"}`}>
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {skipped && ran && (
        <div className="mt-1 text-xs text-slate-500 italic">
          Score not counted in fusion (disabled). Would have scored: {(score || 0).toFixed(3)}
        </div>
      )}
    </div>
  );
}
