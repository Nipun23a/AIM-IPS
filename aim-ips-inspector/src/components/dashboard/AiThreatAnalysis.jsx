import { useState } from "react";

const OAI_KEY = process.env.REACT_APP_OPENAI_API_KEY || "";

export default function AiThreatAnalysis({ stats, events }) {
  const [oaiKey,    setOaiKey]    = useState(() => localStorage.getItem("oai_key") || OAI_KEY || "");
  const [aiOut,     setAiOut]     = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const handleKey = (v) => { setOaiKey(v); localStorage.setItem("oai_key", v); };

  const runThreatSummary = async (mode) => {
    const key = oaiKey || localStorage.getItem("oai_key");
    if (!key) { alert("Enter your OpenAI API key first"); return; }
    setAiOut(""); setAiLoading(true);

    const statsSnap = stats
      ? `Total: ${stats.total_requests}, Blocked: ${stats.blocked_attacks}, Block Rate: ${stats.block_rate_pct}%, Avg Latency: ${stats.avg_latency_ms}ms\nAttack types: ${JSON.stringify(stats.attack_types)}\nTop IPs: ${JSON.stringify((stats.top_ips || []).slice(0, 5))}`
      : "";
    const recent = events.slice(0, 20).map((e) => `${e.ip}(${e.action}:${e.best_label || "?"})`).join(", ");

    const prompt = mode === "summary"
      ? `SOC analyst — summarise today's AIM-IPS threat landscape in 3-4 bullets:\n\n${statsSnap}\nRecent events: ${recent}\n\nMax 120 words.`
      : `SOC analyst — recommend 3-5 specific security actions based on AIM-IPS data:\n\n${statsSnap}\nRecent events: ${recent}\n\nNumbered list. Max 150 words.`;

    try {
      const r = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: prompt }], max_tokens: 350 }),
      });
      const d = await r.json();
      setAiOut(d.error ? `Error: ${d.error.message}` : d.choices[0].message.content);
    } catch (err) {
      setAiOut("Error: " + err.message);
    }
    setAiLoading(false);
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <h2 className="font-semibold text-white text-sm">AI Threat Analysis</h2>
        <input
          type="password"
          value={oaiKey}
          onChange={(e) => handleKey(e.target.value)}
          placeholder="OpenAI API Key"
          className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 w-64"
        />
        <div className="flex gap-2 ml-auto">
          <button
            onClick={() => runThreatSummary("summary")}
            disabled={aiLoading}
            className="px-3 py-1.5 bg-purple-800 hover:bg-purple-700 disabled:opacity-50 text-white rounded-lg text-xs font-medium"
          >
            📊 Summarise Today's Threats
          </button>
          <button
            onClick={() => runThreatSummary("actions")}
            disabled={aiLoading}
            className="px-3 py-1.5 bg-indigo-800 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg text-xs font-medium"
          >
            🔧 Recommend Actions
          </button>
        </div>
      </div>
      {aiLoading && <div className="text-slate-500 text-sm">⏳ Generating analysis…</div>}
      {aiOut && (
        <div className="bg-slate-900 rounded-xl p-4 text-sm text-slate-300 whitespace-pre-wrap">{aiOut}</div>
      )}
    </div>
  );
}
