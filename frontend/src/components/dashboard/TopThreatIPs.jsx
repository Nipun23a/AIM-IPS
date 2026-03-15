import ActionPill from "../common/ActionPill";

export default function TopThreatIPs({ stats, onLoadBlocked }) {
  const blockIP = async (ip) => {
    if (!window.confirm(`Block IP ${ip}?`)) return;
    try {
      const r = await fetch("/api/admin/block-ip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ip, reason: "admin_dashboard" }),
      });
      const d = await r.json();
      alert(d.message || `${ip} blocked`);
    } catch {
      alert("Failed to block IP");
    }
  };

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-white text-sm">Top Threat IPs</h2>
        <button
          onClick={onLoadBlocked}
          className="text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg"
        >
          View Blocked IPs
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              {["IP Address", "Requests", "Blocked", "Last Action", ""].map((h) => (
                <th key={h} className="text-left text-xs text-slate-500 uppercase tracking-wide py-2 px-3">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {!(stats?.top_ips?.length) ? (
              <tr><td colSpan={5} className="text-slate-500 text-center py-6 text-sm">No data yet</td></tr>
            ) : (
              (stats.top_ips || []).map((ip, i) => (
                <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                  <td className="py-2 px-3 font-mono text-blue-300 text-sm">{ip.ip}</td>
                  <td className="py-2 px-3 text-slate-300">{ip.total || 0}</td>
                  <td className="py-2 px-3 text-red-400">{ip.blocked || 0}</td>
                  <td className="py-2 px-3"><ActionPill action={ip.last_action || "ALLOW"} /></td>
                  <td className="py-2 px-3">
                    <button
                      onClick={() => blockIP(ip.ip)}
                      className="text-xs px-3 py-1 bg-red-900/60 hover:bg-red-800/80 text-red-300 rounded-lg border border-red-800"
                    >
                      🚫 Block
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
