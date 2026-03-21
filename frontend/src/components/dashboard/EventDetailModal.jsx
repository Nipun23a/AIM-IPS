import { useState, useEffect } from "react";
import ActionPill from "../common/ActionPill";
import { fmtTime, scoreTextColor, scoreBarFillColor } from "../../utils/helpers";
import { displayLabel } from "../../constants";

const OAI_KEY = import.meta.env.VITE_OPENAI_API_KEY || "";

/* ── IP Intelligence Card ────────────────────────────────────────────── */
function IpIntelCard({ ip }) {
  const [geo,     setGeo]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(false);

  useEffect(() => {
    if (!ip || ip === "127.0.0.1" || ip.startsWith("192.168.") || ip.startsWith("10.")) {
      setLoading(false);
      return;
    }
    setLoading(true);
    fetch(`/api/geoip/${ip}`)
      .then((r) => r.json())
      .then((d) => { setGeo(d.success !== false ? d : null); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, [ip]);

  if (loading) return (
    <div className="bg-slate-800 rounded-xl p-4 flex items-center gap-2 text-xs text-slate-500">
      <span className="w-3 h-3 rounded-full border border-t-blue-400 border-blue-400/30 animate-spin inline-block" />
      Looking up IP intelligence…
    </div>
  );

  if (error || !geo) return (
    <div className="bg-slate-800 rounded-xl p-4 text-xs text-slate-600">
      IP intelligence unavailable (private/loopback address or lookup failed)
    </div>
  );

  const cc = (geo.country_code || "").toLowerCase();
  const flagUrl = cc ? `https://flagcdn.com/24x18/${cc}.png` : null;

  const rows = [
    { label: "Country",      value: geo.country,                     icon: "🌍" },
    { label: "City / Region",value: [geo.city, geo.region].filter(Boolean).join(", ") || "—", icon: "📍" },
    { label: "ISP",          value: geo.connection?.isp || "—",      icon: "🏢" },
    { label: "Organisation", value: geo.connection?.org || "—",      icon: "🏛️" },
    { label: "ASN",          value: geo.connection?.asn ? `AS${geo.connection.asn}` : "—", icon: "🔌" },
    { label: "Domain",       value: geo.connection?.domain || "—",   icon: "🌐" },
    { label: "Timezone",     value: geo.timezone?.id ? `${geo.timezone.id} (${geo.timezone.utc})` : "—", icon: "🕐" },
    { label: "Coordinates",  value: (geo.latitude && geo.longitude) ? `${geo.latitude.toFixed(4)}, ${geo.longitude.toFixed(4)}` : "—", icon: "📡" },
    { label: "IP Type",      value: geo.type || "—",                  icon: "🔢" },
    { label: "Continent",    value: geo.continent || "—",             icon: "🗺️" },
  ].filter((r) => r.value && r.value !== "—");

  // simple threat hints based on connection org
  const orgLower = (geo.connection?.org || "").toLowerCase();
  const ispLower = (geo.connection?.isp || "").toLowerCase();
  const isTor    = orgLower.includes("tor") || ispLower.includes("tor");
  const isVpn    = orgLower.includes("vpn") || ispLower.includes("vpn") || orgLower.includes("nordvpn") || orgLower.includes("expressvpn") || orgLower.includes("mullvad");
  const isHosting = orgLower.includes("hosting") || orgLower.includes("digitalocean") || orgLower.includes("linode") || orgLower.includes("vultr") || orgLower.includes("hetzner") || orgLower.includes("amazon") || orgLower.includes("google") || orgLower.includes("microsoft") || orgLower.includes("ovh");
  const isProxy  = orgLower.includes("proxy") || ispLower.includes("proxy");

  const tags = [
    isTor     && { label: "Tor Exit Node",   color: "bg-red-500/20 text-red-400 border-red-500/40" },
    isVpn     && { label: "VPN Provider",    color: "bg-orange-500/20 text-orange-400 border-orange-500/40" },
    isProxy   && { label: "Proxy Service",   color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40" },
    isHosting && { label: "Hosting / Cloud", color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40" },
  ].filter(Boolean);

  return (
    <div className="bg-slate-800 rounded-xl p-4">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        {flagUrl && (
          <img src={flagUrl} alt={geo.country} width={24} height={18}
            className="rounded-sm border border-slate-700 flex-shrink-0"
            onError={(e) => { e.target.style.display = "none"; }}
          />
        )}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-white">{geo.country || "Unknown"}</div>
          <div className="text-xs text-slate-500 font-mono">{ip}</div>
        </div>
        {tags.length > 0 && (
          <div className="flex flex-wrap gap-1 justify-end">
            {tags.map((t) => (
              <span key={t.label} className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${t.color}`}>
                {t.label}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Detail grid */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {rows.map(({ label, value, icon }) => (
          <div key={label} className="flex items-start gap-1.5 min-w-0">
            <span className="text-xs mt-0.5 flex-shrink-0">{icon}</span>
            <div className="min-w-0">
              <div className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</div>
              <div className="text-xs text-slate-300 truncate" title={value}>{value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Main Modal ──────────────────────────────────────────────────────── */
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

Event: IP=${e.ip}, ${e.method} ${e.path}, Label=${displayLabel(e.best_label || e.block_reason || "?")}, Action=${e.action}, Score=${(e.final_score || 0).toFixed(3)}, NetworkScore=${(e.network_score || 0).toFixed(3)}, ShortCircuited=${e.short_circuited}
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
            <div><span className="text-slate-500">Label: </span><span className="text-slate-300">{displayLabel(event.best_label || event.block_reason || "") || "—"}</span></div>
            <div><span className="text-slate-500">Time: </span><span className="text-slate-300 font-mono text-xs">{fmtTime(event.timestamp)}</span></div>
            {event.latency_ms != null && <div><span className="text-slate-500">Latency: </span><span className="text-purple-400">{event.latency_ms}ms</span></div>}
            {event.short_circuited && (
              <div className="col-span-2">
                <span className="text-xs bg-red-900/50 text-red-300 px-2 py-0.5 rounded border border-red-800">⚡ Short-circuited</span>
              </div>
            )}
          </div>

          {/* IP Intelligence */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 inline-block" />
              IP Intelligence
            </h3>
            <IpIntelCard ip={event.ip} />
          </div>

          {/* Layer scores */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 inline-block" />
              Per-Layer Scores
            </h3>
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
