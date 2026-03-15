import { Link } from "react-router-dom";

// ── Icon helpers ────────────────────────────────────────────────────────────────
function ShieldIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
    </svg>
  );
}
function BoltIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
    </svg>
  );
}
function ChipIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
    </svg>
  );
}
function EyeIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.964-7.178z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}
function GlobeIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418" />
    </svg>
  );
}
function SparklesIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
    </svg>
  );
}

// ── Feature cards ────────────────────────────────────────────────────────────────
const FEATURES = [
  {
    icon: ShieldIcon,
    color: "from-blue-500/20 to-blue-600/10 border-blue-500/30",
    iconColor: "text-blue-400",
    title: "Static Firewall (Layer 0)",
    desc: "Instant IP blacklist lookup, rate-limit enforcement, and bad User-Agent detection. Zero latency first line of defense.",
  },
  {
    icon: EyeIcon,
    color: "from-yellow-500/20 to-yellow-600/10 border-yellow-500/30",
    iconColor: "text-yellow-400",
    title: "Regex Attack Filter (Layer 1)",
    desc: "Pattern-matched detection for SQLi, XSS, command injection, and path traversal with confidence scoring.",
  },
  {
    icon: ChipIcon,
    color: "from-orange-500/20 to-orange-600/10 border-orange-500/30",
    iconColor: "text-orange-400",
    title: "LightGBM Classifier (Layer 2a)",
    desc: "Gradient-boosted tree model trained on network traffic features. Classifies 12+ attack categories with sub-ms inference.",
  },
  {
    icon: BoltIcon,
    color: "from-purple-500/20 to-purple-600/10 border-purple-500/30",
    iconColor: "text-purple-400",
    title: "CNN Autoencoder (Layer 2b)",
    desc: "Deep learning anomaly detection using reconstruction error and Mahalanobis distance to catch zero-day threats.",
  },
  {
    icon: GlobeIcon,
    color: "from-cyan-500/20 to-cyan-600/10 border-cyan-500/30",
    iconColor: "text-cyan-400",
    title: "Network Intelligence",
    desc: "Redis-backed async threat scoring enriched with historical IP behavior and network-level ensemble models.",
  },
  {
    icon: SparklesIcon,
    color: "from-indigo-500/20 to-indigo-600/10 border-indigo-500/30",
    iconColor: "text-indigo-400",
    title: "AI-Powered Analysis",
    desc: "GPT-4o-mini integration explains every detected attack in plain language and recommends SOC response actions.",
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
  { name: "GPT-4o",     color: "bg-emerald-900/40 text-emerald-300 border-emerald-700/40" },
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
            {FEATURES.map(({ icon: Icon, color, iconColor, title, desc }) => (
              <div
                key={title}
                className={`bg-gradient-to-br ${color} rounded-xl border p-5 hover:-translate-y-1 transition-transform duration-200`}
              >
                <div className={`w-10 h-10 rounded-lg bg-slate-900/60 flex items-center justify-center mb-3`}>
                  <Icon className={`w-5 h-5 ${iconColor}`} />
                </div>
                <h3 className="font-semibold text-slate-200 mb-2 text-sm">{title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{desc}</p>
              </div>
            ))}
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
              <ShieldIcon className="w-7 h-7 text-white" />
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
              <ShieldIcon className="w-3.5 h-3.5 text-white" />
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
