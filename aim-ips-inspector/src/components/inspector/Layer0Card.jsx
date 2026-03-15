import ScoreBar from "../common/ScoreBar";
import Badge from "../common/Badge";
import { decisionBorderColor, decisionDot } from "../../utils/helpers";

export default function Layer0Card({ data }) {
  if (!data) return null;
  const { enabled, decision, score, label, reason, detail } = data;
  const skipped = !enabled;

  return (
    <div className={`rounded-lg border-l-4 bg-slate-800/60 border border-slate-700/50 p-3 ${decisionBorderColor(decision, enabled)} ${skipped ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm text-slate-200">Layer 0 — Static Firewall</span>
        <div className="flex gap-1 items-center">
          {skipped && <Badge text="DISABLED" action="DISABLED" />}
          {!skipped && <Badge text={decision} action={decision === "BLOCK" ? "BLOCK" : decision === "SUSPICIOUS" ? "DELAY" : "ALLOW"} />}
        </div>
      </div>
      <div className="flex items-center gap-2 mb-2">
        <span className={`w-2 h-2 rounded-full ${decisionDot(decision, enabled)}`} />
        <span className="text-xs text-slate-400">{label}</span>
        <span className="ml-auto text-xs text-slate-400">score=</span>
        <span className="text-xs font-mono text-slate-200">{(score || 0).toFixed(3)}</span>
        <ScoreBar score={score} mini />
      </div>
      <ScoreBar score={score} />
      <div className="mt-2 text-xs text-slate-500 space-y-0.5">
        <div>
          blacklisted:{" "}
          <span className={detail?.blacklisted ? "text-red-400" : "text-green-500"}>{String(!!detail?.blacklisted)}</span>
          <span className="mx-2">
            rate_limited:{" "}
            <span className={detail?.rate_limited ? "text-red-400" : "text-green-500"}>{String(!!detail?.rate_limited)}</span>
          </span>
          {detail?.rate_count !== undefined && (
            <span>count: <span className="text-slate-300">{detail.rate_count}</span></span>
          )}
        </div>
        {detail?.firewall_match && (
          <div>firewall match: <span className="text-yellow-400">{detail.firewall_match}</span></div>
        )}
        {reason && reason !== "Normal traffic" && (
          <div className="text-slate-500 truncate">{reason}</div>
        )}
      </div>
      {skipped && <div className="mt-1 text-xs text-slate-500 italic">Score not counted in fusion (disabled)</div>}
    </div>
  );
}
