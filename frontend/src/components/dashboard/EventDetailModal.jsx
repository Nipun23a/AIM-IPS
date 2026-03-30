import { useState, useEffect, useCallback } from "react";
import ActionPill from "../common/ActionPill";
import { fmtTime, scoreTextColor, scoreBarFillColor } from "../../utils/helpers";
import { displayLabel } from "../../constants";

const OAI_KEY = import.meta.env.VITE_OPENAI_API_KEY || "";

/* ── Severity badge ──────────────────────────────────────────── */
const SEVERITY_STYLE = {
  CRITICAL: "bg-red-500/20 text-red-400 border-red-500/40",
  HIGH:     "bg-orange-500/20 text-orange-400 border-orange-500/40",
  MEDIUM:   "bg-yellow-500/20 text-yellow-400 border-yellow-500/40",
  LOW:      "bg-green-500/20 text-green-400 border-green-500/40",
};
function SeverityBadge({ severity }) {
  const cls = SEVERITY_STYLE[severity] || SEVERITY_STYLE.MEDIUM;
  return (
    <span className={`text-xs font-bold px-2.5 py-1 rounded-full border uppercase tracking-wide ${cls}`}>
      {severity}
    </span>
  );
}

/* ── IP Intelligence Card ────────────────────────────────────── */
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

  const orgLower = (geo.connection?.org || "").toLowerCase();
  const ispLower = (geo.connection?.isp || "").toLowerCase();
  const isTor     = orgLower.includes("tor") || ispLower.includes("tor");
  const isVpn     = ["vpn","nordvpn","expressvpn","mullvad"].some(v => orgLower.includes(v) || ispLower.includes(v));
  const isHosting = ["hosting","digitalocean","linode","vultr","hetzner","amazon","google","microsoft","ovh"].some(v => orgLower.includes(v));
  const isProxy   = orgLower.includes("proxy") || ispLower.includes("proxy");

  const tags = [
    isTor     && { label: "Tor Exit Node",   color: "bg-red-500/20 text-red-400 border-red-500/40" },
    isVpn     && { label: "VPN Provider",    color: "bg-orange-500/20 text-orange-400 border-orange-500/40" },
    isProxy   && { label: "Proxy Service",   color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/40" },
    isHosting && { label: "Hosting / Cloud", color: "bg-cyan-500/20 text-cyan-400 border-cyan-500/40" },
  ].filter(Boolean);

  return (
    <div className="bg-slate-800 rounded-xl p-4">
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

/* ── AI Analysis Card ────────────────────────────────────────── */
function AIAnalysisCard({ event }) {
  const [analysis,  setAnalysis]  = useState(null);
  const [polling,   setPolling]   = useState(false);
  const [triggered, setTriggered] = useState(false);

  const eventId = event.request_id || event.event_id || "";

  // Auto-fetch on mount — may already be ready if score was high
  useEffect(() => {
    if (!eventId) return;
    let cancelled = false;
    let pollTimer = null;

    const fetchResult = async () => {
      try {
        const r = await fetch(`/api/ai-analysis/${eventId}`);
        const d = await r.json();
        if (cancelled) return;

        if (d.status === "complete" || d.status === "error") {
          setAnalysis(d);
          setPolling(false);
        } else if (d.status === "pending") {
          setPolling(true);
          pollTimer = setTimeout(fetchResult, 3000);
        }
        // "not_found" — do nothing, wait for user to trigger manually
      } catch {
        if (!cancelled) setPolling(false);
      }
    };

    fetchResult();
    return () => {
      cancelled = true;
      clearTimeout(pollTimer);
    };
  }, [eventId]);

  const triggerAnalysis = useCallback(async () => {
    setTriggered(true);
    setPolling(true);

    const scores = event.scores || {};
    try {
      await fetch("/api/ai-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id:    eventId,
          ip_address:  event.ip   || "",
          method:      event.method || "GET",
          url:         event.path  || "/",
          payload:     event.body  || "",
          attack_type: event.best_label || "unknown",
          action_taken: event.action || "BLOCK",
          detection_scores: {
            lgbm_score:                scores.app_lgbm_score   || 0,
            cnn_mahalanobis:           scores.deep_anomaly     || 0,
            fusion_score:              scores.final_risk       || event.final_score || 0,
            correlation_amplification: 1.0,
            final_score:               event.final_score || 0,
          },
          shap_explanation:   {},
          correlation_context: { network_score: event.network_score || 0 },
        }),
      });
    } catch { /* ignore */ }

    // Start polling
    const poll = async () => {
      try {
        const r = await fetch(`/api/ai-analysis/${eventId}`);
        const d = await r.json();
        if (d.status === "complete" || d.status === "error") {
          setAnalysis(d);
          setPolling(false);
        } else {
          setTimeout(poll, 3000);
        }
      } catch { setPolling(false); }
    };
    setTimeout(poll, 2000);
  }, [event, eventId]);

  /* ── Pending / polling ── */
  if (polling && !analysis) {
    return (
      <div className="bg-slate-800 rounded-xl p-5 flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin flex-shrink-0" />
        <div>
          <div className="text-sm text-slate-300 font-medium">AI analysis in progress…</div>
          <div className="text-xs text-slate-500 mt-0.5">Claude is analysing this threat · typically 5-10s</div>
        </div>
      </div>
    );
  }

  /* ── Not triggered ── */
  if (!analysis && !triggered) {
    return (
      <div className="bg-slate-800/60 rounded-xl border border-dashed border-slate-700 p-5 text-center">
        <div className="text-slate-400 text-sm mb-1">AI deep analysis not yet run</div>
        <div className="text-slate-600 text-xs mb-4">
          Auto-runs for score ≥ 0.65. Click below to run manually.
        </div>
        <button
          onClick={triggerAnalysis}
          className="px-5 py-2 bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600 text-white text-sm font-medium rounded-xl transition-all"
        >
          Run AI Analysis
        </button>
      </div>
    );
  }

  /* ── Error ── */
  if (analysis?.status === "error") {
    return (
      <div className="bg-red-900/20 border border-red-800/40 rounded-xl p-4 text-xs text-red-400">
        Analysis failed: {analysis.error || "Unknown error"}
      </div>
    );
  }

  if (!analysis) return null;

  /* ── Full analysis ── */
  const tc    = analysis.threat_classification || {};
  const rec   = analysis.mitigation_recommendations || {};
  const intel = analysis.threat_intelligence || {};
  const mitre = tc.mitre_technique || {};

  const honeyCount  = intel.honeydb?.count || 0;
  const abuseScore  = intel.abuseipdb?.abuse_confidence_score || 0;

  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-2">
        {tc.severity && <SeverityBadge severity={tc.severity} />}
        {tc.is_novel_variant && (
          <span className="text-xs font-bold px-2.5 py-1 rounded-full border bg-purple-500/20 text-purple-400 border-purple-500/40 uppercase tracking-wide">
            Novel Variant
          </span>
        )}
        <span className="text-sm font-semibold text-slate-200 ml-1">
          {tc.attack_type || "Unknown Attack"}
        </span>
        {tc.owasp_category && (
          <span className="text-xs text-slate-500 ml-auto">{tc.owasp_category}</span>
        )}
      </div>

      {/* Analyst summary */}
      {analysis.analyst_summary && (
        <div className="bg-slate-800/80 rounded-xl p-4 border border-slate-700/40">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-1.5">Analyst Summary</div>
          <p className="text-sm text-slate-300 leading-relaxed">{analysis.analyst_summary}</p>
        </div>
      )}

      {/* MITRE + Kill Chain */}
      {mitre.id && (
        <div className="flex items-center gap-3 bg-slate-800/60 rounded-xl px-4 py-3 border border-slate-700/40">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-red-900/40 border border-red-800/40 flex items-center justify-center text-xs font-bold font-mono text-red-400">
            {mitre.id.split(".")[0]}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-semibold text-slate-200">{mitre.name}</div>
            <div className="text-[10px] text-slate-500">
              {mitre.id} · {mitre.tactic}
              {tc.kill_chain_phase ? ` · Kill Chain: ${tc.kill_chain_phase}` : ""}
            </div>
          </div>
          {mitre.url && (
            <a href={mitre.url} target="_blank" rel="noreferrer"
              className="text-[10px] text-blue-400 hover:text-blue-300 flex-shrink-0">
              ATT&CK ↗
            </a>
          )}
        </div>
      )}

      {/* Threat intel row */}
      {(honeyCount > 0 || abuseScore > 0 || intel.global_prevalence) && (
        <div className="grid grid-cols-3 gap-2">
          {honeyCount > 0 && (
            <div className="bg-slate-800/60 rounded-xl p-3 text-center border border-slate-700/40">
              <div className="text-lg font-black text-orange-400">{honeyCount}</div>
              <div className="text-[10px] text-slate-500">HoneyDB hits</div>
            </div>
          )}
          {abuseScore > 0 && (
            <div className="bg-slate-800/60 rounded-xl p-3 text-center border border-slate-700/40">
              <div className={`text-lg font-black ${abuseScore > 75 ? "text-red-400" : abuseScore > 40 ? "text-orange-400" : "text-yellow-400"}`}>
                {abuseScore}%
              </div>
              <div className="text-[10px] text-slate-500">AbuseIPDB score</div>
            </div>
          )}
          {intel.global_prevalence && (
            <div className="bg-slate-800/60 rounded-xl p-3 text-center border border-slate-700/40">
              <div className={`text-sm font-bold capitalize ${
                intel.global_prevalence === "widespread" ? "text-red-400" :
                intel.global_prevalence === "emerging"   ? "text-orange-400" : "text-green-400"
              }`}>{intel.global_prevalence}</div>
              <div className="text-[10px] text-slate-500">Global prevalence</div>
            </div>
          )}
        </div>
      )}

      {/* Root cause */}
      {analysis.root_cause_analysis && (
        <details className="bg-slate-800/60 rounded-xl border border-slate-700/40 overflow-hidden">
          <summary className="px-4 py-3 text-xs font-semibold text-slate-300 cursor-pointer select-none hover:text-white">
            Root Cause Analysis
          </summary>
          <div className="px-4 pb-4 text-xs text-slate-400 leading-relaxed">
            {analysis.root_cause_analysis}
          </div>
        </details>
      )}

      {/* Evasion + sophistication */}
      {(analysis.attack_sophistication || (analysis.evasion_techniques || []).length > 0) && (
        <div className="flex flex-wrap gap-2">
          {analysis.attack_sophistication && (
            <span className="text-[10px] px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-400">
              Sophistication: {analysis.attack_sophistication}
            </span>
          )}
          {(analysis.evasion_techniques || []).map((t) => (
            <span key={t} className="text-[10px] px-2.5 py-1 rounded-full bg-yellow-900/20 border border-yellow-800/40 text-yellow-400">
              {t}
            </span>
          ))}
        </div>
      )}

      {/* Immediate actions */}
      {(rec.immediate_actions || []).length > 0 && (
        <div className="bg-red-900/10 border border-red-800/30 rounded-xl p-4">
          <div className="text-[10px] text-red-400 uppercase tracking-widest font-semibold mb-2">
            Immediate Actions
          </div>
          <ul className="space-y-1.5">
            {rec.immediate_actions.map((a, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                <span className="text-red-500 mt-0.5 flex-shrink-0">→</span>
                {a}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Mitigations expandable */}
      {(rec.waf_rules?.length > 0 || rec.regex_patterns?.length > 0 || rec.long_term_fixes?.length > 0) && (
        <details className="bg-slate-800/60 rounded-xl border border-slate-700/40 overflow-hidden">
          <summary className="px-4 py-3 text-xs font-semibold text-slate-300 cursor-pointer select-none hover:text-white">
            Mitigations (WAF rules, Regex, Long-term)
          </summary>
          <div className="px-4 pb-4 space-y-3">
            {rec.waf_rules?.length > 0 && (
              <div>
                <div className="text-[10px] text-slate-500 uppercase mb-1.5">WAF Rules (ModSecurity)</div>
                {rec.waf_rules.map((r, i) => (
                  <code key={i} className="block bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-[10px] text-green-300 font-mono break-all mb-1">
                    {r}
                  </code>
                ))}
              </div>
            )}
            {rec.regex_patterns?.length > 0 && (
              <div>
                <div className="text-[10px] text-slate-500 uppercase mb-1.5">Regex Patterns (Layer 1)</div>
                {rec.regex_patterns.map((p, i) => (
                  <code key={i} className="block bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-[10px] text-cyan-300 font-mono break-all mb-1">
                    {p}
                  </code>
                ))}
              </div>
            )}
            {rec.long_term_fixes?.length > 0 && (
              <div>
                <div className="text-[10px] text-slate-500 uppercase mb-1.5">Long-term Fixes</div>
                <ul className="space-y-1">
                  {rec.long_term_fixes.map((f, i) => (
                    <li key={i} className="text-xs text-slate-400 flex gap-2">
                      <span className="text-indigo-500 flex-shrink-0">•</span>{f}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </details>
      )}

      {/* IOCs */}
      {(analysis.iocs || []).length > 0 && (
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">Indicators of Compromise</div>
          <div className="flex flex-wrap gap-1.5">
            {analysis.iocs.map((ioc, i) => (
              <span key={i} className="text-[10px] font-mono px-2 py-1 bg-slate-800 border border-slate-700 rounded text-slate-300 break-all">
                {ioc}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="text-[10px] text-slate-600 text-right">
        Analysed by Claude · {new Date(analysis.analysis_timestamp * 1000).toLocaleTimeString()}
      </div>
    </div>
  );
}

/* ── Main Modal ──────────────────────────────────────────────── */
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

          {/* Per-Layer Scores */}
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

          {/* AI Deep Analysis */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 inline-block animate-pulse" />
              AI Deep Analysis
              <span className="text-[10px] text-slate-500 font-normal ml-1">powered by Claude</span>
            </h3>
            <AIAnalysisCard event={event} />
          </div>

          {/* GPT Quick Explain (existing feature) */}
          <div>
            <button
              onClick={explainAttack}
              disabled={explainLoading}
              className="w-full py-2.5 bg-gradient-to-r from-purple-700 to-indigo-700 hover:from-purple-600 hover:to-indigo-600 disabled:opacity-50 text-white rounded-xl font-medium text-sm"
            >
              {explainLoading ? "⏳ Asking GPT-4o-mini…" : "✨ Quick Explain (GPT-4o-mini)"}
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
