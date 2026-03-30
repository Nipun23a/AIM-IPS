import { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import {
  Shield,
  Eye,
  Cpu,
  Zap,
  Globe,
  Sparkles,
  Radio,
  Network,
  Activity,
  Brain,
  AlertTriangle,
  FileSearch,
  CheckCircle,
} from "lucide-react";

// ── Threat Correlation Animation ─────────────────────────────────────────────

function CorrelationDemo() {
  const [step, setStep] = useState(0);
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setStep(1), 800),
      setTimeout(() => setStep(2), 2200),
      setTimeout(() => setStep(3), 3800),
      setTimeout(() => { setStep(0); setAnimKey(k => k + 1); }, 6500),
    ];
    return () => timers.forEach(clearTimeout);
  }, [animKey]);

  const networkVisible  = step >= 1;
  const appVisible      = step >= 2;
  const corrVisible     = step >= 3;

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 md:p-8 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-xs font-mono text-slate-400">correlation_engine.py · live simulation</span>
        </div>
        <div className="text-xs text-slate-600 font-mono">auto-replay</div>
      </div>

      {/* Two-lane timeline */}
      <div className="space-y-3 mb-6">
        {/* Network lane */}
        <div className="flex items-center gap-3">
          <div className="w-24 text-right flex-shrink-0">
            <span className="text-xs font-mono text-cyan-400">network</span>
          </div>
          <div className="flex-1 h-10 bg-slate-800/60 rounded-lg border border-slate-700/40 relative overflow-hidden">
            <div
              className={`absolute inset-y-0 left-0 flex items-center px-3 gap-2 transition-all duration-500 ${networkVisible ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"}`}
            >
              <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse flex-shrink-0" />
              <span className="text-xs text-cyan-300 font-mono">portscan</span>
              <span className="text-xs text-slate-500">score=0.45</span>
              <span className="ml-auto text-xs font-mono px-1.5 py-0.5 rounded bg-yellow-500/10 border border-yellow-500/30 text-yellow-400">DELAY</span>
            </div>
          </div>
          <div className="w-12 text-xs font-mono text-slate-600 flex-shrink-0">t=0s</div>
        </div>

        {/* App lane */}
        <div className="flex items-center gap-3">
          <div className="w-24 text-right flex-shrink-0">
            <span className="text-xs font-mono text-purple-400">application</span>
          </div>
          <div className="flex-1 h-10 bg-slate-800/60 rounded-lg border border-slate-700/40 relative overflow-hidden">
            <div
              className={`absolute inset-y-0 left-0 flex items-center px-3 gap-2 transition-all duration-500 ${appVisible ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-4"}`}
            >
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse flex-shrink-0" />
              <span className="text-xs text-purple-300 font-mono">sqli</span>
              <span className="text-xs text-slate-500">score=0.65</span>
              <span className="ml-auto text-xs font-mono px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/30 text-orange-400">CAPTCHA</span>
            </div>
          </div>
          <div className="w-12 text-xs font-mono text-slate-600 flex-shrink-0">t=4s</div>
        </div>
      </div>

      {/* Correlation result */}
      <div className={`transition-all duration-700 ${corrVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
        <div className="border border-red-500/40 bg-red-500/5 rounded-xl p-4">
          <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
            {/* Left: correlation badge */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <div className="w-8 h-8 rounded-lg bg-red-500/20 border border-red-500/40 flex items-center justify-center">
                <span className="text-sm">🔗</span>
              </div>
              <div>
                <div className="text-xs text-red-400 font-semibold uppercase tracking-wider">Correlation Detected</div>
                <div className="text-xs text-slate-500 font-mono">layers=[network, application] · types=2</div>
              </div>
            </div>

            {/* Score amplification */}
            <div className="flex items-center gap-2 flex-wrap md:ml-auto">
              <div className="text-xs font-mono text-slate-400 bg-slate-800 px-2 py-1 rounded border border-slate-700">
                0.65
              </div>
              <span className="text-slate-600 text-xs">×</span>
              <div className="text-xs font-mono text-yellow-400 bg-yellow-500/10 px-2 py-1 rounded border border-yellow-500/30">
                1.3×
              </div>
              <span className="text-slate-600 text-xs">=</span>
              <div className="text-xs font-mono text-red-300 bg-red-500/10 px-2 py-1 rounded border border-red-500/30 font-bold">
                0.845
              </div>
              <div className="text-xs font-bold px-3 py-1 rounded-lg bg-red-600 text-white ml-1 animate-pulse">
                BLOCK
              </div>
            </div>
          </div>

          {/* Explanation */}
          <div className="mt-3 pt-3 border-t border-red-500/20 grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-xs text-slate-500">Window</div>
              <div className="text-xs font-mono text-slate-300">10 seconds</div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Distinct types</div>
              <div className="text-xs font-mono text-slate-300">2 → ×1.3</div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Stored in</div>
              <div className="text-xs font-mono text-cyan-400">corr:hist:&#123;ip&#125;</div>
            </div>
          </div>
        </div>
      </div>

      {/* Step indicators */}
      <div className="flex items-center gap-2 mt-5 justify-center">
        {[1,2,3].map((s) => (
          <div
            key={s}
            className={`h-1 rounded-full transition-all duration-300 ${step >= s ? "w-8 bg-indigo-500" : "w-3 bg-slate-700"}`}
          />
        ))}
      </div>
    </div>
  );
}



// ── AI Analysis Animation ─────────────────────────────────────────────────────────
function AIAnalysisDemo() {
  const [step, setStep] = useState(0);
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => {
    const timers = [
      setTimeout(() => setStep(1), 600),
      setTimeout(() => setStep(2), 1800),
      setTimeout(() => setStep(3), 3200),
      setTimeout(() => setStep(4), 4800),
      setTimeout(() => { setStep(0); setAnimKey(k => k + 1); }, 8000),
    ];
    return () => timers.forEach(clearTimeout);
  }, [animKey]);

  const stages = [
    { icon: AlertTriangle, color: "text-red-400",    bg: "bg-red-500/10 border-red-500/30",     label: "Threat detected",        sub: "score=1.0 · hard-block",       active: step >= 1 },
    { icon: Brain,         color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/30", label: "Enqueued for Claude",   sub: "ai:threat:queue · Redis",      active: step >= 2 },
    { icon: FileSearch,    color: "text-cyan-400",   bg: "bg-cyan-500/10 border-cyan-500/30",    label: "Threat Intel gathered",  sub: "HoneyDB · AbuseIPDB · MITRE",  active: step >= 3 },
    { icon: CheckCircle,   color: "text-green-400",  bg: "bg-green-500/10 border-green-500/30",  label: "Claude analysis ready",  sub: "severity=CRITICAL · T1190",    active: step >= 4 },
  ];

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 md:p-8">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
          <span className="text-xs font-mono text-slate-400">ai_analysis_worker.py · live simulation</span>
        </div>
        <div className="text-xs text-slate-600 font-mono">auto-replay</div>
      </div>

      {/* Stage pipeline */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {stages.map(({ icon: Icon, color, bg, label, sub, active }, i) => (
          <div key={i} className={`rounded-xl border p-3 text-center transition-all duration-500 ${active ? bg : "bg-slate-800/40 border-slate-700/40 opacity-40"}`}>
            <Icon className={`w-5 h-5 mx-auto mb-2 transition-colors duration-500 ${active ? color : "text-slate-600"}`} />
            <div className={`text-xs font-semibold transition-colors duration-500 ${active ? "text-slate-200" : "text-slate-600"}`}>{label}</div>
            <div className="text-xs text-slate-500 mt-0.5 font-mono">{sub}</div>
          </div>
        ))}
      </div>

      {/* Result card */}
      <div className={`transition-all duration-700 ${step >= 4 ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
        <div className="border border-green-500/30 bg-green-500/5 rounded-xl p-4 font-mono text-xs space-y-1.5">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-green-400 font-semibold">Claude Analysis Complete</span>
            <span className="text-slate-600 ml-auto">~50s elapsed</span>
          </div>
          <div><span className="text-slate-500">attack_type   </span><span className="text-orange-300">SQL Injection — UNION-based</span></div>
          <div><span className="text-slate-500">severity      </span><span className="text-red-400 font-bold">CRITICAL</span></div>
          <div><span className="text-slate-500">mitre_id      </span><span className="text-cyan-300">T1190 · Exploit Public-Facing App</span></div>
          <div><span className="text-slate-500">kill_chain    </span><span className="text-yellow-300">Exploitation</span></div>
          <div><span className="text-slate-500">honeydb_hits  </span><span className="text-slate-300">638 reports</span></div>
          <div><span className="text-slate-500">waf_rules     </span><span className="text-slate-300">6 ModSecurity rules generated</span></div>
          <div><span className="text-slate-500">cached in     </span><span className="text-indigo-300">Redis ai:analysis:&#123;id&#125; · TTL=24h</span></div>
        </div>
      </div>

      <div className="flex items-center gap-2 mt-5 justify-center">
        {[1,2,3,4].map((s) => (
          <div key={s} className={`h-1 rounded-full transition-all duration-300 ${step >= s ? "w-8 bg-purple-500" : "w-3 bg-slate-700"}`} />
        ))}
      </div>
    </div>
  );
}

// ── Network layer pipeline steps ─────────────────────────────────────────────────
const NET_STEPS = [
  { icon: Radio,    color: "text-cyan-400",   bg: "bg-cyan-500/10 border-cyan-500/30",    label: "Scapy captures",      sub: "raw packets on eth0" },
  { icon: Activity, color: "text-blue-400",   bg: "bg-blue-500/10 border-blue-500/30",    label: "FlowAccumulator",     sub: "builds IP flows" },
  { icon: Cpu,      color: "text-orange-400", bg: "bg-orange-500/10 border-orange-500/30", label: "LightGBM + Ensemble", sub: "classifies flows" },
  { icon: Network,  color: "text-red-400",    bg: "bg-red-500/10 border-red-500/30",      label: "Redis SETEX",         sub: "threat:ip:{ip} TTL=60s" },
  { icon: Shield,   color: "text-indigo-400", bg: "bg-indigo-500/10 border-indigo-500/30", label: "Middleware reads",    sub: "<0.1ms per request" },
];

// ── Feature cards ────────────────────────────────────────────────────────────────
const FEATURES = [
  {
    icon: Shield,
    color: "from-blue-500/20 to-blue-600/10 border-blue-500/30",
    iconColor: "text-blue-400",
    title: "Static Firewall (Layer 0)",
    desc: "Instant IP blacklist lookup, rate-limit enforcement, and bad User-Agent detection. Zero latency first line of defense.",
  },
  {
    icon: Eye,
    color: "from-yellow-500/20 to-yellow-600/10 border-yellow-500/30",
    iconColor: "text-yellow-400",
    title: "Regex Attack Filter (Layer 1)",
    desc: "Pattern-matched detection for SQLi, XSS, command injection, and path traversal with confidence scoring.",
  },
  {
    icon: Cpu,
    color: "from-orange-500/20 to-orange-600/10 border-orange-500/30",
    iconColor: "text-orange-400",
    title: "LightGBM Classifier (Layer 2a)",
    desc: "Gradient-boosted tree model trained on network traffic features. Classifies 12+ attack categories with sub-ms inference.",
  },
  {
    icon: Zap,
    color: "from-purple-500/20 to-purple-600/10 border-purple-500/30",
    iconColor: "text-purple-400",
    title: "CNN Autoencoder (Layer 2b)",
    desc: "Deep learning anomaly detection using reconstruction error and Mahalanobis distance to catch zero-day threats.",
  },
  {
    icon: Globe,
    color: "from-cyan-500/20 to-cyan-600/10 border-cyan-500/30",
    iconColor: "text-cyan-400",
    title: "Network Layer IPS",
    desc: "Runs as a separate background process using Scapy to capture raw packets, build IP flows, and classify DDoS, port-scans, and botnet C2 traffic — writing threat scores to Redis so the middleware reads them in <0.1ms per request.",
  },
  {
    icon: Sparkles,
    color: "from-indigo-500/20 to-indigo-600/10 border-indigo-500/30",
    iconColor: "text-indigo-400",
    title: "Autonomous AI Agent (Claude)",
    desc: "Every high-severity threat is queued for deep analysis by Claude Sonnet — generating MITRE ATT&CK mappings, root cause analysis, ModSecurity WAF rules, and IOCs. Zero impact on request latency.",
  },
];


const PIPELINE = [
  { layer: "Layer 0", label: "Static Firewall",  desc: "IP + UA checks",       color: "bg-blue-900/30 border-blue-700/40" },
  { layer: "Layer 1", label: "Regex Filter",      desc: "Pattern matching",     color: "bg-yellow-900/30 border-yellow-700/40" },
  { layer: "Network", label: "Redis Score",       desc: "Async threat intel",   color: "bg-cyan-900/30 border-cyan-700/40" },
  { layer: "Layer 2b", label: "CNN Gate",         desc: "Anomaly detection",    color: "bg-purple-900/30 border-purple-700/40" },
  { layer: "Layer 2a", label: "LightGBM",         desc: "Attack classifier",    color: "bg-orange-900/30 border-orange-700/40" },
  { layer: "Fusion",   label: "Final Decision",   desc: "Weighted ensemble",    color: "bg-red-900/30 border-red-700/40" },
];

// ── Tech badges ───────────────────────────────────────────────────────────────────
const TECH = [
  { name: "Python 3.11", color: "bg-blue-900/40 text-blue-300 border-blue-700/40" },
  { name: "FastAPI",     color: "bg-green-900/40 text-green-300 border-green-700/40" },
  { name: "React 19",   color: "bg-cyan-900/40 text-cyan-300 border-cyan-700/40" },
  { name: "Redis",       color: "bg-red-900/40 text-red-300 border-red-700/40" },
  { name: "LightGBM",   color: "bg-orange-900/40 text-orange-300 border-orange-700/40" },
  { name: "PyTorch CNN", color: "bg-purple-900/40 text-purple-300 border-purple-700/40" },
  { name: "Claude Sonnet", color: "bg-purple-900/40 text-purple-300 border-purple-700/40" },
  { name: "Anthropic API", color: "bg-emerald-900/40 text-emerald-300 border-emerald-700/40" },
  { name: "Tailwind CSS",color: "bg-sky-900/40 text-sky-300 border-sky-700/40" },
];

// ── Stats ────────────────────────────────────────────────────────────────────────
const STATS = [
  { value: "5",     label: "Detection Layers",   color: "text-indigo-400" },
  { value: "12+",   label: "Attack Categories",  color: "text-red-400"    },
  { value: "<5ms",  label: "Avg Decision Time",  color: "text-green-400"  },
  { value: "99.8%", label: "Detection Accuracy", color: "text-yellow-400" },
];

// ── Home Page ────────────────────────────────────────────────────────────────────
export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">

      {/* ── Hero ─────────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden pt-20 pb-28 px-4">
        {/* Gradient background blobs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -left-40 w-[600px] h-[600px] bg-indigo-600/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -right-40 w-[600px] h-[600px] bg-purple-600/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-blue-600/5 rounded-full blur-3xl" />
        </div>

        {/* Grid pattern overlay */}
        <div
          className="absolute inset-0 opacity-[0.03] pointer-events-none"
          style={{
            backgroundImage: "linear-gradient(rgba(99,102,241,1) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,1) 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="relative max-w-5xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-medium mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse inline-block" />
            AI-Powered Intrusion Prevention System
          </div>

          {/* Headline */}
          <h1 className="text-5xl md:text-7xl font-black mb-6 leading-none tracking-tight">
            <span className="text-white">AIM-IPS</span>
            <br />
            <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Inspector
            </span>
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            A multi-layer, AI-driven web security system that analyses HTTP requests
            in real time — combining static rules, regex patterns, machine learning
            and deep learning to stop cyber threats before they reach your application.
          </p>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              to="/aim-ips-inspector"
              className="group flex items-center gap-2 px-7 py-3.5 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all duration-200 shadow-xl shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:-translate-y-0.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
              </svg>
              Try the Inspector
            </Link>
            <Link
              to="/login"
              className="flex items-center gap-2 px-7 py-3.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-xl border border-slate-700 hover:border-slate-600 transition-all duration-200 hover:-translate-y-0.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5m.75-9l3-3 2.148 2.148A12.061 12.061 0 0116.5 7.605" />
              </svg>
              Admin Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* ── Stats ────────────────────────────────────────────────────────────── */}
      <section className="px-4 py-12 border-y border-slate-800/60">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6">
          {STATS.map(({ value, label, color }) => (
            <div key={label} className="text-center">
              <div className={`text-4xl font-black ${color} mb-1`}>{value}</div>
              <div className="text-sm text-slate-500">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── What is AIM-IPS ──────────────────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-5xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">About the Project</div>
              <h2 className="text-3xl md:text-4xl font-black text-white mb-5 leading-tight">
                What is<br />AIM-IPS?
              </h2>
              <p className="text-slate-400 leading-relaxed mb-5">
                <strong className="text-slate-200">AIM-IPS</strong> (AI-based Intrusion Prevention System) is a research-grade
                web security middleware that sits in front of your application and analyses every HTTP request
                through a multi-stage AI pipeline.
              </p>
              <p className="text-slate-400 leading-relaxed mb-5">
                Unlike traditional WAFs that rely solely on static rules, AIM-IPS combines
                <strong className="text-slate-300"> static firewall rules</strong>,
                <strong className="text-slate-300"> regex pattern matching</strong>,
                <strong className="text-slate-300"> network threat intelligence</strong>, a
                <strong className="text-slate-300"> CNN autoencoder</strong> for zero-day anomaly detection, and a
                <strong className="text-slate-300"> LightGBM classifier</strong> for known attack categorisation
                — all fused into a single weighted decision.
              </p>
              <p className="text-slate-400 leading-relaxed">
                The Inspector tool lets you fire custom HTTP requests at the pipeline and observe
                every layer's decision, score, and reasoning in real time.
              </p>
            </div>
            {/* Visual card */}
            <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
              {[
                { label: "Request arrives", color: "bg-slate-600", text: "text-slate-300", icon: "→" },
                { label: "Layer 0: Firewall check",   color: "bg-blue-600",   text: "text-blue-300",   icon: "🔒" },
                { label: "Layer 1: Regex patterns",   color: "bg-yellow-600", text: "text-yellow-300", icon: "🔍" },
                { label: "Layer 2b: CNN anomaly",     color: "bg-purple-600", text: "text-purple-300", icon: "🧠" },
                { label: "Layer 2a: LightGBM class.", color: "bg-orange-600", text: "text-orange-300", icon: "⚡" },
                { label: "Fusion → Final decision",   color: "bg-red-600",    text: "text-red-300",    icon: "✓" },
              ].map(({ label, color, text, icon }, i) => (
                <div key={i} className="flex items-center gap-3">
                  {i > 0 && <div className="absolute ml-4 -mt-5 w-0.5 h-3 bg-slate-700" />}
                  <div className={`w-8 h-8 rounded-lg ${color}/20 border border-${color}/30 flex items-center justify-center text-sm flex-shrink-0`}>
                    {icon}
                  </div>
                  <span className={`text-sm font-medium ${text}`}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Research Gaps ───────────────────────────────────────────────────── */}
      <section className="px-4 py-20 bg-slate-900/40">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <div className="text-xs text-red-400 font-semibold uppercase tracking-widest mb-3">Motivation</div>
            <h2 className="text-3xl md:text-4xl font-black text-white mb-4">5 Critical Research Gaps</h2>
            <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed">
              A systematic literature review identified five critical gaps in current intrusion prevention systems.
              AIM-IPS was designed to address every one of them.
            </p>
          </div>

          <div className="space-y-6">

            {/* GAP 1 */}
            <div className="rounded-2xl border border-orange-500/30 bg-orange-500/5 p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-shrink-0">
                  <div className="w-14 h-14 rounded-2xl bg-orange-500/15 border border-orange-500/30 flex items-center justify-center">
                    <span className="text-2xl font-black text-orange-400">1</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-bold text-white mb-2">Limited Integration of Response Mechanisms</h3>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Most IDS research focuses solely on detection accuracy while neglecting autonomous mitigation.
                    Ali &amp; Jack's 2022 reinforcement learning approach remained experimental. Ismail's AI-driven
                    system achieved 1.5s reaction time but relied on static datasets and offline inference.
                    Critically, existing approaches provide only binary <span className="text-slate-300 font-medium">block or allow</span> decisions with no proportional response.
                  </p>
                  <div className="bg-orange-500/10 border border-orange-500/20 rounded-xl p-4">
                    <div className="text-xs text-orange-400 font-semibold uppercase tracking-wider mb-2">AIM-IPS Solution</div>
                    <p className="text-slate-300 text-sm leading-relaxed">
                      Five graduated response actions — <span className="font-mono text-green-400">ALLOW</span>, <span className="font-mono text-yellow-400">DELAY</span>, <span className="font-mono text-orange-400">THROTTLE</span>, <span className="font-mono text-red-400">CAPTCHA</span>, <span className="font-mono text-red-600">BLOCK</span> — calibrated to threat confidence.
                      A scanner hitting THROTTLE experiences a 3-second artificial delay per request, reducing scan rate by <span className="text-white font-bold">99%</span> without impacting legitimate users.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GAP 2 */}
            <div className="rounded-2xl border border-blue-500/30 bg-blue-500/5 p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-shrink-0">
                  <div className="w-14 h-14 rounded-2xl bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
                    <span className="text-2xl font-black text-blue-400">2</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-bold text-white mb-2">Lack of Middleware / API-Based Deployment for Web Applications</h3>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Most academic works are evaluated on controlled datasets like NSL-KDD with no consideration
                    for production environments. Diaz-Verdejo's 2022 evaluation of Snort and ModSecurity showed
                    these tools require continuous tuning and <span className="text-slate-300 font-medium">fail against zero-day attacks</span>.
                    No existing IPS framework is deployable as a drop-in web middleware.
                  </p>
                  <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                    <div className="text-xs text-blue-400 font-semibold uppercase tracking-wider mb-2">AIM-IPS Solution</div>
                    <p className="text-slate-300 text-sm leading-relaxed">
                      <span className="font-mono text-blue-300">IPSMiddleware</span> implemented as a FastAPI ASGI component that intercepts every HTTP request in-path.
                      Production-deployed on Ubuntu 22.04 behind Nginx with SSL termination, processing real traffic with end-to-end latency under <span className="text-white font-bold">5ms</span>.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GAP 3 */}
            <div className="rounded-2xl border border-yellow-500/30 bg-yellow-500/5 p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-shrink-0">
                  <div className="w-14 h-14 rounded-2xl bg-yellow-500/15 border border-yellow-500/30 flex items-center justify-center">
                    <span className="text-2xl font-black text-yellow-400">3</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-bold text-white mb-2">Scalability and Adaptability to Evolving Attacks</h3>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Models like AE-IDS and AE-LSTM show strong benchmark results but degrade in live environments
                    as new attack variants emerge. Mirsky's Kitsune showed promise in IoT scenarios but didn't extend
                    to enterprise-scale web environments. <span className="text-slate-300 font-medium">No existing system adapts its rules without retraining.</span>
                  </p>
                  <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-4">
                    <div className="text-xs text-yellow-400 font-semibold uppercase tracking-wider mb-2">AIM-IPS Solution</div>
                    <p className="text-slate-300 text-sm leading-relaxed">
                      TFLite optimisation reduces CNN inference from 52ms to under <span className="text-white font-bold">5ms (10× faster)</span>.
                      Autonomous Adaptive Rule Injection: Claude-generated regex patterns are hot-reloaded into Layer 1 via Redis every 30 seconds —
                      no model retraining, no system restart. Novel attack variants progressively migrate from expensive ML inference to cheap regex matching.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GAP 4 */}
            <div className="rounded-2xl border border-red-500/30 bg-red-500/5 p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-shrink-0">
                  <div className="w-14 h-14 rounded-2xl bg-red-500/15 border border-red-500/30 flex items-center justify-center">
                    <span className="text-2xl font-black text-red-400">4</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="text-lg font-bold text-white">Absence of Cross-Layer Threat Correlation</h3>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-red-500/20 border border-red-500/40 text-red-400">Key Contribution</span>
                  </div>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    Network-layer tools like Snort monitor packets for DDoS and port scans.
                    Application-layer WAFs monitor HTTP for injection attacks.
                    <span className="text-slate-300 font-medium"> No existing framework correlates intelligence across both layers</span> to identify
                    multi-stage coordinated attacks. A port scan below threshold followed by SQL injection below threshold
                    will individually evade any single-layer system.
                  </p>
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                    <div className="text-xs text-red-400 font-semibold uppercase tracking-wider mb-2">AIM-IPS Solution</div>
                    <p className="text-slate-300 text-sm leading-relaxed mb-3">
                      Dedicated cross-pipeline correlation engine tracks per-IP events across both pipelines in Redis within a <span className="text-white font-bold">10-second window</span>.
                      Threat score is amplified proportional to attack diversity.
                    </p>
                    <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                      <div className="rounded-lg bg-slate-900 border border-slate-700 p-2">
                        <div className="text-slate-400 font-bold">×1.0</div>
                        <div className="text-slate-500 mt-0.5">1 type</div>
                      </div>
                      <div className="rounded-lg bg-yellow-900/30 border border-yellow-700/40 p-2">
                        <div className="text-yellow-400 font-bold">×1.3</div>
                        <div className="text-slate-500 mt-0.5">2 types</div>
                      </div>
                      <div className="rounded-lg bg-red-900/30 border border-red-700/40 p-2">
                        <div className="text-red-400 font-bold">×1.5–1.7</div>
                        <div className="text-slate-500 mt-0.5">3–4 types</div>
                      </div>
                    </div>
                    <p className="text-slate-400 text-xs mt-3 font-mono">
                      Example: SQLi at CAPTCHA (0.65) + prior port scan → <span className="text-red-300">0.65 × 1.3 = 0.845 → BLOCK</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* GAP 5 */}
            <div className="rounded-2xl border border-purple-500/30 bg-purple-500/5 p-6 md:p-8">
              <div className="flex flex-col md:flex-row gap-6">
                <div className="flex-shrink-0">
                  <div className="w-14 h-14 rounded-2xl bg-purple-500/15 border border-purple-500/30 flex items-center justify-center">
                    <span className="text-2xl font-black text-purple-400">5</span>
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg font-bold text-white mb-2">Lack of Autonomous Explainable Threat Intelligence</h3>
                  <p className="text-slate-400 text-sm leading-relaxed mb-4">
                    While SHAP and LIME have been applied to IDS outputs in offline settings, <span className="text-slate-300 font-medium">no existing IPS
                    autonomously generates structured threat explanations</span>, maps to MITRE ATT&amp;CK, or produces
                    ready-to-deploy defensive artifacts like WAF rules — all without impacting latency.
                    Security analysts are left interpreting raw anomaly scores with no contextual guidance.
                  </p>
                  <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
                    <div className="text-xs text-purple-400 font-semibold uppercase tracking-wider mb-3">AIM-IPS Solution</div>
                    <p className="text-slate-300 text-sm leading-relaxed mb-3">
                      Asynchronous AI reasoning agent powered by Claude Sonnet. For any request scoring above 0.65,
                      the system enqueues it for deep analysis — decoupled via Redis queue with <span className="text-white font-bold">zero impact</span> on sub-5ms request latency.
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                      {[
                        { label: "Attack Classification", sub: "OWASP category · severity", color: "text-red-400" },
                        { label: "MITRE ATT&CK Mapping", sub: "Technique · Kill Chain phase", color: "text-orange-400" },
                        { label: "WAF Rules + IOCs",      sub: "Ready-to-deploy ModSecurity", color: "text-cyan-400" },
                        { label: "Threat Intel Fusion",   sub: "HoneyDB · AbuseIPDB",        color: "text-green-400" },
                      ].map(({ label, sub, color }) => (
                        <div key={label} className="rounded-lg bg-slate-900 border border-slate-700 p-2.5 text-center">
                          <div className={`font-semibold ${color} mb-0.5`}>{label}</div>
                          <div className="text-slate-500 font-mono">{sub}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

          </div>

          {/* Summary row */}
          <div className="mt-10 bg-slate-900 rounded-2xl border border-slate-800 p-6">
            <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider text-center mb-5">AIM-IPS addresses all five gaps</div>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs text-center">
              {[
                { num: "G1", label: "Graduated Response Engine",          color: "border-orange-500/30 bg-orange-500/5 text-orange-400" },
                { num: "G2", label: "FastAPI Middleware Deployment",       color: "border-blue-500/30 bg-blue-500/5 text-blue-400" },
                { num: "G3", label: "TFLite + Adaptive Rule Injection",   color: "border-yellow-500/30 bg-yellow-500/5 text-yellow-400" },
                { num: "G4", label: "Cross-Pipeline Correlation Engine",  color: "border-red-500/30 bg-red-500/5 text-red-400" },
                { num: "G5", label: "Autonomous AI Reasoning Agent",      color: "border-purple-500/30 bg-purple-500/5 text-purple-400" },
              ].map(({ num, label, color }) => (
                <div key={num} className={`rounded-xl border p-3 ${color}`}>
                  <div className="font-black text-base mb-1">{num}</div>
                  <div className="text-slate-300 leading-snug">{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Pipeline Architecture ────────────────────────────────────────────── */}
      <section className="px-4 py-16 bg-slate-900/40">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">Architecture</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Detection Pipeline</h2>
            <p className="text-slate-400 mt-3 max-w-xl mx-auto">
              Each request traverses up to 5 detection layers before a final fused decision is made.
              Short-circuit logic skips deep ML when early layers already confirm a block.
            </p>
          </div>

          {/* Pipeline flow */}
          <div className="flex items-stretch gap-2 md:gap-3 overflow-x-auto pb-4">
            {PIPELINE.map((step, i) => (
              <div key={i} className="flex items-center gap-2 md:gap-3 flex-shrink-0 md:flex-1">
                <div className={`rounded-xl border p-3 md:p-4 text-center flex-shrink-0 w-28 md:w-auto md:flex-1 ${step.color}`}>
                  <div className="text-xs text-slate-500 font-mono mb-1">{step.layer}</div>
                  <div className="text-sm font-semibold text-slate-200 leading-tight">{step.label}</div>
                  <div className="text-xs text-slate-500 mt-1 hidden md:block">{step.desc}</div>
                </div>
                {i < PIPELINE.length - 1 && (
                  <svg className="w-4 h-4 text-slate-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── System Architecture Diagram ─────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-10">
            <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">System Design</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Full System Architecture</h2>
            <p className="text-slate-400 mt-3 max-w-2xl mx-auto">
              End-to-end view of how Internet traffic flows through the Application Layer IPS,
              Network Layer IPS, Cross-Pipeline Correlation Engine, and AI Analysis Worker
              before reaching the protected application.
            </p>
          </div>
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-4 md:p-6 overflow-hidden">
            <img
              src="/architecture_diagram.png"
              alt="AIM-IPS Full System Architecture Diagram"
              className="w-full h-auto rounded-xl object-contain"
            />
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 text-center text-xs">
            {[
              { label: "Application Layer IPS", sub: "HTTP Middleware · FastAPI ASGI", color: "border-blue-500/30 bg-blue-500/5 text-blue-400" },
              { label: "Network Layer IPS",      sub: "Scapy · Async Background Process", color: "border-cyan-500/30 bg-cyan-500/5 text-cyan-400" },
              { label: "Cross-Pipeline Engine",  sub: "Redis Sorted Set · 10s Window",   color: "border-red-500/30 bg-red-500/5 text-red-400" },
              { label: "AI Analysis Worker",     sub: "Claude Sonnet · Zero Latency",     color: "border-purple-500/30 bg-purple-500/5 text-purple-400" },
            ].map(({ label, sub, color }) => (
              <div key={label} className={`rounded-xl border p-3 ${color}`}>
                <div className="font-semibold text-slate-200 mb-1">{label}</div>
                <div className="text-slate-500 font-mono">{sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Network Layer ────────────────────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <div className="text-xs text-cyan-400 font-semibold uppercase tracking-widest mb-3">Network Layer</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Packet-Level Threat Detection</h2>
            <p className="text-slate-400 mt-3 max-w-2xl mx-auto">
              A completely independent background process captures raw network packets, builds IP flows,
              and classifies threats — feeding scores into the middleware via Redis with zero latency overhead.
            </p>
          </div>

          {/* Architecture diagram */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 mb-10">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm">
              {NET_STEPS.map((step, i) => (
                <div key={i} className="flex items-center gap-3 md:flex-1">
                  <div className={`flex-1 rounded-xl border p-3 text-center ${step.bg}`}>
                    <step.icon className={`w-5 h-5 ${step.color} mx-auto mb-1.5`} />
                    <div className="font-semibold text-slate-200 text-xs">{step.label}</div>
                    <div className="text-slate-500 text-xs mt-0.5">{step.sub}</div>
                  </div>
                  {i < NET_STEPS.length - 1 && (
                    <svg className="w-4 h-4 text-slate-600 flex-shrink-0 hidden md:block" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* What it detects + How to run */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Detects */}
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-6">
              <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" /> What It Detects
              </h3>
              <div className="space-y-3">
                {[
                  { label: "DDoS / SYN Flood",  method: "LightGBM",          color: "text-red-400",    dot: "bg-red-500",    desc: "High packet rate, SYN count spike" },
                  { label: "Port Scan",           method: "LightGBM",          color: "text-orange-400", dot: "bg-orange-500", desc: "Many small flows, low bytes/packet" },
                  { label: "Botnet C2",           method: "LightGBM",          color: "text-yellow-400", dot: "bg-yellow-500", desc: "Periodic beaconing pattern" },
                  { label: "Zero-Day / Unknown",  method: "Ensemble AE+VAE+IF", color: "text-purple-400", dot: "bg-purple-500", desc: "Anomaly score from unsupervised models" },
                ].map(({ label, method, color, dot, desc }) => (
                  <div key={label} className="flex items-start gap-3">
                    <div className={`w-2 h-2 rounded-full ${dot} mt-1.5 flex-shrink-0`} />
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-slate-200 text-sm font-medium">{label}</span>
                        <span className={`text-xs font-mono ${color}`}>{method}</span>
                      </div>
                      <div className="text-slate-500 text-xs">{desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* How to run */}
            <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-6">
              <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                <Radio className="w-4 h-4 text-cyan-400" /> How to Start It
              </h3>
              <p className="text-slate-400 text-xs mb-4 leading-relaxed">
                The network layer runs as a <span className="text-slate-200 font-medium">completely separate process</span> alongside
                FastAPI. It requires <code className="bg-slate-800 px-1 rounded text-cyan-400">sudo</code> for raw socket access (Scapy).
              </p>
              <div className="space-y-3">
                {[
                  { label: "Live capture on VPS", cmd: "sudo python -m pipeline.network_level.network_ips --interface eth0" },
                  { label: "Test without sudo",   cmd: "python -m pipeline.network_level.network_ips --simulate" },
                  { label: "Debug mode",          cmd: "sudo python -m pipeline.network_level.network_ips --interface eth0 --debug" },
                ].map(({ label, cmd }) => (
                  <div key={label}>
                    <div className="text-xs text-slate-500 mb-1">{label}</div>
                    <code className="block bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-cyan-300 font-mono break-all">{cmd}</code>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-600 mt-4">
                If not running, network score defaults to 0.0 — the other 4 layers still protect normally.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Threat Correlation Pipeline ──────────────────────────────────────── */}
      <section className="px-4 py-20 bg-slate-900/40">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-10">
            <div className="text-xs text-red-400 font-semibold uppercase tracking-widest mb-3">Cross-Pipeline Intelligence</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Threat Correlation Engine</h2>
            <p className="text-slate-400 mt-3 max-w-2xl mx-auto">
              When an IP is seen attacking at both the network layer (Scapy) and the application layer (FastAPI)
              within a short time window, AIM-IPS correlates the events and amplifies the final score —
              turning a soft block into a hard one.
            </p>
          </div>
          <CorrelationDemo />
          <div className="grid md:grid-cols-3 gap-4 mt-6 text-center">
            {[
              { label: "1 attack type",  mult: "×1.0", color: "text-slate-400", bg: "bg-slate-800 border-slate-700" },
              { label: "2 attack types", mult: "×1.3", color: "text-yellow-400", bg: "bg-yellow-900/20 border-yellow-700/40" },
              { label: "3+ attack types",mult: "×1.5–1.7", color: "text-red-400", bg: "bg-red-900/20 border-red-700/40" },
            ].map(({ label, mult, color, bg }) => (
              <div key={label} className={`rounded-xl border p-4 ${bg}`}>
                <div className={`text-xl font-black font-mono ${color}`}>{mult}</div>
                <div className="text-xs text-slate-500 mt-1">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Autonomous AI Agent ──────────────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-10">
            <div className="text-xs text-purple-400 font-semibold uppercase tracking-widest mb-3">Autonomous AI Agent</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Claude-Powered Threat Intelligence</h2>
            <p className="text-slate-400 mt-3 max-w-2xl mx-auto">
              Every blocked or high-severity request is automatically queued for deep analysis.
              A background worker calls Claude Sonnet with full threat context — layer scores, SHAP weights,
              HoneyDB reputation, and AbuseIPDB data — and returns a structured forensic report in seconds,
              with zero impact on request latency.
            </p>
          </div>

          <AIAnalysisDemo />

          <div className="grid md:grid-cols-3 gap-4 mt-8">
            {[
              {
                title: "MITRE ATT&CK Mapping",
                desc: "Every attack is automatically mapped to a MITRE technique ID, tactic, and Cyber Kill Chain phase.",
                color: "border-purple-500/30 bg-purple-500/5",
                tag: "T1190 · T1059 · T1046",
                tagColor: "text-purple-400",
              },
              {
                title: "Threat Intelligence Fusion",
                desc: "HoneyDB honeypot reputation and AbuseIPDB crowd-sourced abuse scores are merged into the analysis context.",
                color: "border-cyan-500/30 bg-cyan-500/5",
                tag: "HoneyDB · AbuseIPDB",
                tagColor: "text-cyan-400",
              },
              {
                title: "Actionable Mitigations",
                desc: "Claude generates ready-to-deploy ModSecurity WAF rules, regex patterns, and ML threshold adjustment recommendations.",
                color: "border-green-500/30 bg-green-500/5",
                tag: "WAF rules · IOCs · fixes",
                tagColor: "text-green-400",
              },
            ].map(({ title, desc, color, tag, tagColor }) => (
              <div key={title} className={`rounded-xl border p-5 ${color}`}>
                <div className={`text-xs font-mono mb-2 ${tagColor}`}>{tag}</div>
                <h3 className="font-bold text-white text-sm mb-2">{title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>

          {/* Architecture callout */}
          <div className="mt-6 bg-slate-900 rounded-2xl border border-slate-800 p-5">
            <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-3">How it stays non-blocking</div>
            <div className="flex flex-col md:flex-row items-stretch gap-3 text-xs font-mono">
              {[
                { label: "Request blocked",   detail: "<3ms · middleware",       color: "text-red-400",    bg: "bg-red-500/10 border-red-500/20" },
                { label: "Event enqueued",    detail: "Redis RPUSH · fire & forget", color: "text-yellow-400", bg: "bg-yellow-500/10 border-yellow-500/20" },
                { label: "Worker dequeues",   detail: "ai:threat:queue · 0.5s poll", color: "text-cyan-400",   bg: "bg-cyan-500/10 border-cyan-500/20" },
                { label: "Claude called",     detail: "thread executor · ~50s",  color: "text-purple-400", bg: "bg-purple-500/10 border-purple-500/20" },
                { label: "Result cached",     detail: "Redis · 24h TTL · dashboard", color: "text-green-400",  bg: "bg-green-500/10 border-green-500/20" },
              ].map(({ label, detail, color, bg }, i) => (
                <div key={i} className="flex items-center gap-2 md:flex-1">
                  <div className={`flex-1 rounded-lg border p-3 ${bg}`}>
                    <div className={`font-semibold ${color}`}>{label}</div>
                    <div className="text-slate-500 mt-0.5">{detail}</div>
                  </div>
                  {i < 4 && <span className="text-slate-600 flex-shrink-0 hidden md:block">→</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">Capabilities</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">Everything You Need</h2>
            <p className="text-slate-400 mt-3 max-w-xl mx-auto">
              AIM-IPS brings enterprise-grade threat detection to developers with full pipeline transparency.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((feature) => {
              const Icon = feature.icon;

              return (
                <div
                  key={feature.title}
                  className={`bg-gradient-to-br ${feature.color} rounded-xl border p-5 hover:-translate-y-1 transition-transform duration-200`}
                >
                  <div className="w-10 h-10 rounded-lg bg-slate-900/60 flex items-center justify-center mb-3">
                    <Icon className={`w-5 h-5 ${feature.iconColor}`} />
                  </div>

                  <h3 className="font-semibold text-slate-200 mb-2 text-sm">
                    {feature.title}
                  </h3>

                  <p className="text-slate-400 text-xs leading-relaxed">
                    {feature.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── How to Use ───────────────────────────────────────────────────────── */}
      <section className="px-4 py-16 bg-slate-900/40">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">Quick Start</div>
            <h2 className="text-3xl md:text-4xl font-black text-white">How to Use</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Open the Inspector",
                desc: "Navigate to /aim-ips-inspector and select an attack preset or craft a custom HTTP request.",
                link: "/aim-ips-inspector",
                cta: "Open Inspector",
                color: "border-indigo-500/30 bg-indigo-500/5",
                stepColor: "text-indigo-400",
              },
              {
                step: "02",
                title: "Configure Layers",
                desc: "Toggle individual detection layers on or off to see how each one contributes to the final decision.",
                link: "/aim-ips-inspector",
                cta: "Try It",
                color: "border-purple-500/30 bg-purple-500/5",
                stepColor: "text-purple-400",
              },
              {
                step: "03",
                title: "Monitor Dashboard",
                desc: "Log in to the Admin Dashboard to see real-time stats, charts, threat maps and AI-powered SOC analysis.",
                link: "/login",
                cta: "Go to Dashboard",
                color: "border-cyan-500/30 bg-cyan-500/5",
                stepColor: "text-cyan-400",
              },
            ].map(({ step, title, desc, link, cta, color, stepColor }) => (
              <div key={step} className={`rounded-2xl border p-6 ${color}`}>
                <div className={`text-4xl font-black ${stepColor} opacity-40 mb-3`}>{step}</div>
                <h3 className="font-bold text-white text-lg mb-2">{title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed mb-5">{desc}</p>
                <Link
                  to={link}
                  className="inline-flex items-center gap-1.5 text-sm font-medium text-slate-300 hover:text-white transition-colors"
                >
                  {cta}
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                  </svg>
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Tech Stack ───────────────────────────────────────────────────────── */}
      <section className="px-4 py-16">
        <div className="max-w-5xl mx-auto text-center">
          <div className="text-xs text-indigo-400 font-semibold uppercase tracking-widest mb-3">Built With</div>
          <h2 className="text-2xl font-black text-white mb-8">Technology Stack</h2>
          <div className="flex flex-wrap gap-3 justify-center">
            {TECH.map(({ name, color }) => (
              <span
                key={name}
                className={`px-4 py-2 rounded-lg border text-sm font-medium ${color}`}
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ───────────────────────────────────────────────────────── */}
      <section className="px-4 py-20">
        <div className="max-w-3xl mx-auto text-center">
          <div className="bg-gradient-to-br from-indigo-900/40 to-purple-900/40 rounded-3xl border border-indigo-500/20 p-12">
            <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl shadow-indigo-500/30">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <h2 className="text-3xl md:text-4xl font-black text-white mb-4">
              Ready to Inspect?
            </h2>
            <p className="text-slate-400 mb-8 max-w-lg mx-auto">
              Fire your first request through the multi-layer AI pipeline and see every detection decision explained in detail.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/aim-ips-inspector"
                className="px-8 py-3.5 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 hover:-translate-y-0.5"
              >
                Launch Inspector
              </Link>
              <Link
                to="/login"
                className="px-8 py-3.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-xl border border-slate-700 hover:border-slate-600 transition-all duration-200 hover:-translate-y-0.5"
              >
                Admin Login
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800 py-8 px-4">
        <div className="max-w-5xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-indigo-500 to-purple-600 rounded flex items-center justify-center">
              <Shield className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold text-sm text-slate-300">AIM-IPS Inspector</span>
          </div>
          <div className="flex gap-6 text-xs text-slate-500">
            <Link to="/aim-ips-inspector" className="hover:text-slate-300 transition-colors">Inspector</Link>
            <Link to="/login"            className="hover:text-slate-300 transition-colors">Admin</Link>
            <Link to="/admin-dashboard"  className="hover:text-slate-300 transition-colors">Dashboard</Link>
          </div>
          <p className="text-xs text-slate-600">AI-powered multi-layer IPS · Research Project</p>
        </div>
      </footer>
    </div>
  );
}
