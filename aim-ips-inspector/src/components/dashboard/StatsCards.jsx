export default function StatsCards({ stats }) {
  const cards = [
    { id: "total_requests",    label: "Total Requests", sub: "All traffic",   color: "text-blue-400",   suffix: "" },
    { id: "blocked_attacks",   label: "Blocked",        sub: "Hard blocks",   color: "text-red-400",    suffix: "" },
    { id: "allowed_requests",  label: "Allowed",        sub: "Clean traffic", color: "text-green-400",  suffix: "" },
    { id: "throttled_captcha", label: "Throttled/Cap",  sub: "Rate limited",  color: "text-yellow-400", suffix: "" },
    { id: "block_rate_pct",    label: "Block Rate",     sub: "% of traffic",  color: "text-orange-400", suffix: "%" },
    { id: "avg_latency_ms",    label: "Avg Latency",    sub: "ms overhead",   color: "text-purple-400", suffix: "ms" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {cards.map(({ id, label, sub, color, suffix }) => (
        <div
          key={id}
          className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 hover:-translate-y-0.5 transition-transform duration-200"
        >
          <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">{label}</div>
          <div className={`text-2xl font-bold ${color}`}>
            {stats ? (stats[id] != null ? `${stats[id]}${suffix}` : "—") : "—"}
          </div>
          <div className={`text-xs mt-1 ${color} opacity-70`}>{sub}</div>
        </div>
      ))}
    </div>
  );
}
