import ScoreBar from "../common/ScoreBar";
import Badge from "../common/Badge";
import { decisionBorderColor, decisionDot } from "../../utils/helpers";

export default function NetworkCard({ data }) {
  if (!data) return null;
  const { enabled, found, score, attack_type, lgbm_score, ensemble_score } = data;
  const skipped  = !enabled;
  const decision = found
    ? (score > 0.5 ? "BLOCK" : score > 0.25 ? "SUSPICIOUS" : "CLEAN")
    : "CLEAN";

  return (
    <div className={`rounded-lg border-l-4 bg-slate-800/60 border border-slate-700/50 p-3 ${decisionBorderColor(decision, enabled)} ${skipped ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm text-slate-200">Network — Redis Score</span>
        <div className="flex gap-1 items-center">
          {skipped            && <Badge text="DISABLED" action="DISABLED" />}
          {!skipped && !found && <Badge text="NO DATA"  action="ALLOW"    />}
          {!skipped && found  && <Badge text={decision} action={decision === "BLOCK" ? "BLOCK" : decision === "SUSPICIOUS" ? "DELAY" : "ALLOW"} />}
        </div>
      </div>
      <div className="flex items-center gap-2 mb-2">
        <span className={`w-2 h-2 rounded-full ${found && enabled ? decisionDot(decision, enabled) : "bg-slate-600"}`} />
        <span className="text-xs text-slate-400">{found ? (attack_type || "unknown") : "no threat data"}</span>
        <span className="ml-auto text-xs text-slate-400">score=</span>
        <span className="text-xs font-mono text-slate-200">{(score || 0).toFixed(3)}</span>
        <ScoreBar score={score} mini />
      </div>
      <ScoreBar score={score} />
      <div className="mt-2 text-xs text-slate-500 space-y-0.5">
        {!found && <div className="italic">First request or no network activity — no score in Redis yet for this IP</div>}
        {found && lgbm_score != null && (
          <div>
            net_lgbm: <span className="text-slate-300">{lgbm_score.toFixed(3)}</span>
            {ensemble_score != null && (
              <span className="ml-3">ensemble: <span className="text-slate-300">{ensemble_score.toFixed(3)}</span></span>
            )}
          </div>
        )}
      </div>
      {skipped && <div className="mt-1 text-xs text-slate-500 italic">Score not counted in fusion (disabled)</div>}
    </div>
  );
}
