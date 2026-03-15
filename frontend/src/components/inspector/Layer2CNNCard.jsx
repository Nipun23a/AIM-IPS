import ScoreBar from "../common/ScoreBar";
import Badge from "../common/Badge";
import { decisionBorderColor, decisionDot } from "../../utils/helpers";

export default function Layer2CNNCard({ data }) {
  if (!data) return null;
  const { enabled, ran, skipped_reason, score, is_anomaly, recon_score, maha_score } = data;
  const skipped          = !enabled;
  const wasShortCircuited = !ran && skipped_reason === "short_circuited";
  const decision          = !ran ? "CLEAN" : is_anomaly ? "BLOCK" : score > 0.25 ? "SUSPICIOUS" : "CLEAN";

  const borderCls = wasShortCircuited || !ran
    ? "border-slate-600"
    : decisionBorderColor(decision, enabled);

  return (
    <div className={`rounded-lg border-l-4 bg-slate-800/60 border border-slate-700/50 p-3 ${borderCls} ${skipped ? "opacity-60" : ""}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-semibold text-sm text-slate-200">Layer 2b — CNN Autoencoder</span>
        <div className="flex gap-1 items-center">
          {skipped && <Badge text="DISABLED" action="DISABLED" />}
          {wasShortCircuited && !skipped && <Badge text="SKIPPED" action="SKIPPED" />}
          {ran && !skipped && <Badge text={is_anomaly ? "ANOMALY" : "NORMAL"} action={is_anomaly ? "BLOCK" : "ALLOW"} />}
          {!ran && skipped_reason && skipped_reason !== "short_circuited" && (
            <Badge text={skipped_reason} action="DISABLED" />
          )}
        </div>
      </div>

      {wasShortCircuited && !skipped ? (
        <div className="text-xs text-slate-500 italic">Short-circuited by Layer 0/1</div>
      ) : !ran ? (
        <div className="text-xs text-slate-500 italic">{skipped_reason || "not run"}</div>
      ) : (
        <>
          <div className="flex items-center gap-2 mb-2">
            <span className={`w-2 h-2 rounded-full ${decisionDot(decision, enabled)}`} />
            <span className="text-xs text-slate-400">{is_anomaly ? "zero-day anomaly" : "normal"}</span>
            <span className="ml-auto text-xs text-slate-400">score=</span>
            <span className="text-xs font-mono text-slate-200">{(score || 0).toFixed(3)}</span>
            <ScoreBar score={score} mini />
          </div>
          <ScoreBar score={score} />
          {recon_score != null && (
            <div className="mt-2 text-xs">
              <span className="text-slate-500">recon: </span>
              <span className="text-slate-300 font-mono">{recon_score.toFixed(4)}</span>
              {maha_score != null && (
                <span className="ml-3">
                  <span className="text-slate-500">maha: </span>
                  <span className="text-slate-300 font-mono">{maha_score.toFixed(4)}</span>
                </span>
              )}
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
