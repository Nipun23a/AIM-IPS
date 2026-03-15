import { useState } from "react";
import ActionPill from "../common/ActionPill";
import { fmtTime, scoreTextColor, scoreBarFillColor } from "../../utils/helpers";

const OAI_KEY = process.env.REACT_APP_OPENAI_API_KEY || "";

export default function EventDetailModal({ event, onClose }) {
  const [explainOut,     setExplainOut]     = useState("");
  const [explainLoading, setExplainLoading] = useState(false);

  const explainAttack = async () => {
    const key = localStorage.getItem("oai_key") || OAI_KEY;
    if (!key) { alert("Enter your OpenAI API key in the AI Threat Analysis panel first"); return; }
    setExplainOut(""); setExplainLoading(true);
    const e      = event;
    const prompt = `You are a cybersecurity expert. Analyse this IPS event in 3 short paragraphs:
1. What attack type this likely is
2. Why the IPS scored it ${(e.final_score || 0).toFixed(3)} (action: ${e.action})
3. What the attacker was attempting

Event: IP=${e.ip}, ${e.method} ${e.path}, Label=${e.best_label || e.block_reason || "?"}, Action=${e.action}, Score=${(e.final_score || 0).toFixed(3)}, NetworkScore=${(e.network_score || 0).toFixed(3)}, ShortCircuited=${e.short_circuited}
Scores: ${JSON.stringify(e.scores || {})}
Max 150 words.`;
    try {
      const r = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: prompt }], max_tokens: 300 }),
      });
      const d = await r.json();
      setExplainOut(d.error ? `Error: ${d.error.message}` : d.choices[0].message.content);
    } catch (err) {
      setExplainOut("Error: " + err.message);
    }
    setExplainLoading(false);
  };

  const ls = event.layer_scores || [];
  const sc = event.scores || {};
  const colorMap = {
    layer0_firewall: "bg-blue-500",
    layer1_regex:    "bg-yellow-500",
    layer2_lgbm:     "bg-orange-500",
    layer2_cnn:      "bg-purple-500",
    network:         "bg-cyan-500",
  };

  return (
    <div
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-5 border-b border-slate-800">
          <h2 className="font-bold text-white text-lg">Event Detail</h2>
          <button onClick={onClose} className="text-slate-500 hover:text-white text-2xl leading-none">×</button>
        </div>
        <div className="p-5 space-y-5">
          {/* Request info */}
          <div className="bg-slate-800 rounded-xl p-4 grid grid-cols-2 gap-3 text-sm">
            <div><span className="text-slate-500">IP: </span><span className="font-mono text-blue-300">{event.ip}</span></div>
            <div><span className="text-slate-500">Method: </span><span className="text-white">{event.method}</span></div>
            <div className="col-span-2"><span className="text-slate-500">Path: </span><span className="font-mono text-slate-300">{event.path}</span></div>
            <div><span className="text-slate-500">Action: </span><ActionPill action={event.action} /></div>
            <div><span className="text-slate-500">Label: </span><span className="text-slate-300">{event.best_label || event.block_reason || "—"}</span></div>
            <div><span className="text-slate-500">Time: </span><span className="text-slate-300 font-mono text-xs">{fmtTime(event.timestamp)}</span></div>
            {event.latency_ms != null && <div><span className="text-slate-500">Latency: </span><span className="text-purple-400">{event.latency_ms}ms</span></div>}
            {event.short_circuited && (
              <div className="col-span-2">
                <span className="text-xs bg-red-900/50 text-red-300 px-2 py-0.5 rounded border border-red-800">⚡ Short-circuited</span>
              </div>
            )}
          </div>

          {/* Layer scores */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">Per-Layer Scores</h3>
            <div className="space-y-2">
              {ls.length > 0 ? (
                ls.map((l, i) => {
                  const pct = Math.min(100, Math.round((l.score || 0) * 100));
                  return (
                    <div key={i}>
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>{l.layer} <span className="text-slate-600">— {l.label}{l.triggered ? " ⚡" : ""}</span></span>
                        <span className={scoreTextColor(l.score)}>{(l.score || 0).toFixed(3)}</span>
                      </div>
                      <div className="bg-slate-700 rounded-full h-2">
                        <div className={`h-2 rounded-full ${colorMap[l.layer] || "bg-slate-500"}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })
              ) : (
                [["Regex Score", sc.regex_conf, "bg-yellow-500"], ["Network Score", event.network_score, "bg-cyan-500"], ["LGBM Score", sc.app_lgbm_score, "bg-orange-500"], ["CNN Score", sc.deep_anomaly, "bg-purple-500"]].map(([lbl, val, col]) => {
                  const pct = Math.min(100, Math.round((val || 0) * 100));
                  return (
                    <div key={lbl}>
                      <div className="flex justify-between text-xs text-slate-400 mb-1">
                        <span>{lbl}</span><span>{(val || 0).toFixed(3)}</span>
                      </div>
                      <div className="bg-slate-700 rounded-full h-2">
                        <div className={`h-2 rounded-full ${col}`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })
              )}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300 font-semibold">Final Score</span>
                  <span className={`font-semibold ${scoreTextColor(event.final_score)}`}>{(event.final_score || 0).toFixed(3)}</span>
                </div>
                <div className="bg-slate-700 rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full ${scoreBarFillColor(event.final_score)}`}
                    style={{ width: `${Math.min(100, Math.round((event.final_score || 0) * 100))}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Explain button */}
          <div>
            <button
              onClick={explainAttack}
              disabled={explainLoading}
              className="w-full py-2.5 bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600 disabled:opacity-50 text-white rounded-xl font-medium text-sm"
            >
              {explainLoading ? "⏳ Asking GPT-4o-mini…" : "✨ Explain This Attack (GPT-4o-mini)"}
            </button>
            {explainOut && (
              <div className="mt-3 bg-slate-800 rounded-xl p-4 text-sm text-slate-300 whitespace-pre-wrap">{explainOut}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
