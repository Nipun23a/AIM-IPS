import ScoreBar from "../common/ScoreBar";
import Badge from "../common/Badge";
import { decisionBorderColor, decisionDot } from "../../utils/helpers";

export default function Layer1Card({ data, shortCircuitAt }) {
  if (!data) return null;
  const { enabled, decision, score, label, pattern, confidence, group, matched_in, matched_text } = data;
  const skipped = !enabled;

  return (
    <div className={`rounded-lg border-l-4 bg-slate-800/60 border border-slate-700/50 p-3 ${decisionBorderColor(decision, enabled)} ${skipped ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm text-slate-200">Layer 1 — Regex Filter</span>
        <div className="flex gap-1 items-center">
          {skipped && <Badge text="DISABLED" action="DISABLED" />}
          {!skipped && <Badge text={decision} action={decision === "BLOCK" ? "BLOCK" : decision === "SUSPICIOUS" ? "DELAY" : "ALLOW"} />}
          {shortCircuitAt === "layer1" && !skipped && <Badge text="TRIGGERED SC" action="BLOCK" />}
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
      {pattern && (
        <div className="mt-2 text-xs space-y-0.5">
          <div><span className="text-slate-500">pattern: </span><span className="text-yellow-300">{pattern}</span></div>
          <div>
            <span className="text-slate-500">group: </span><span className="text-slate-300">{group}</span>
            <span className="text-slate-500 ml-3">confidence: </span><span className="text-slate-300">{(confidence || 0).toFixed(2)}</span>
          </div>
          {matched_in   && <div><span className="text-slate-500">matched in: </span><span className="text-slate-400">{matched_in}</span></div>}
          {matched_text && <div className="truncate"><span className="text-slate-500">matched text: </span><span className="font-mono text-red-300">{matched_text}</span></div>}
        </div>
      )}
      {skipped && <div className="mt-1 text-xs text-slate-500 italic">Score not counted in fusion (disabled)</div>}
    </div>
  );
}
