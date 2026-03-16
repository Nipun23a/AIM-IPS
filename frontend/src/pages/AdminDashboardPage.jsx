import { useState, useEffect, useCallback, memo } from "react";
import { useNavigate }       from "react-router-dom";
import StatsCards            from "../components/dashboard/StatsCards";
import ChartsRow             from "../components/dashboard/ChartsRow";
import GlobePanel            from "../components/dashboard/GlobePanel";
import TopThreatIPs          from "../components/dashboard/TopThreatIPs";
import AiThreatAnalysis      from "../components/dashboard/AiThreatAnalysis";
import EventsTable           from "../components/dashboard/EventsTable";
import EventDetailModal      from "../components/dashboard/EventDetailModal";
import BlockedIPsModal       from "../components/dashboard/BlockedIPsModal";

const LiveClock = memo(function LiveClock() {
  const [clock, setClock] = useState(() =>
    new Date().toLocaleTimeString("en-US", { hour12: false })
  );
  useEffect(() => {
    const t = setInterval(
      () => setClock(new Date().toLocaleTimeString("en-US", { hour12: false })),
      1000,
    );
    return () => clearInterval(t);
  }, []);
  return <span className="text-slate-400 text-sm font-mono">{clock}</span>;
});

export default function AdminDashboardPage() {
  const navigate = useNavigate();

  // Auth guard
  useEffect(() => {
    if (sessionStorage.getItem("ips_auth") !== "1") {
      navigate("/login", { replace: true });
    }
  }, [navigate]);

  const logout = () => {
    sessionStorage.removeItem("ips_auth");
    navigate("/login", { replace: true });
  };

  const [stats,         setStats]         = useState(null);
  const [allEvents,     setAllEvents]     = useState([]);
  const [paused,        setPaused]        = useState(false);
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [blockedIPs,    setBlockedIPs]    = useState([]);
  const [showBlocked,   setShowBlocked]   = useState(false);
  const [filterIp,      setFilterIp]      = useState("");
  const [filterAction,  setFilterAction]  = useState("");
  const [filterLabel,   setFilterLabel]   = useState("");

  const fetchStats = useCallback(async () => {
    try { const r = await fetch("/api/stats"); setStats(await r.json()); } catch {}
  }, []);

  const fetchEvents = useCallback(async () => {
    if (paused) return;
    try {
      const r = await fetch("/api/events?limit=100");
      const d = await r.json();
      setAllEvents(d.events || []);
    } catch {}
  }, [paused]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchStats();
    fetchEvents();
    const si = setInterval(fetchStats,  15000);
    const ei = setInterval(fetchEvents, 5000);
    return () => { clearInterval(si); clearInterval(ei); };
  }, [fetchStats, fetchEvents]);

  const loadBlockedIPs = async () => {
    try {
      const r = await fetch("/api/admin/blocked-ips");
      const d = await r.json();
      setBlockedIPs(d.blocked_ips || []);
      setShowBlocked(true);
    } catch {
      alert("Failed to load blocked IPs");
    }
  };

  const filtered = allEvents
    .filter((e) => {
      if (filterIp     && !e.ip.includes(filterIp)) return false;
      if (filterAction && e.action !== filterAction) return false;
      if (filterLabel  && !(e.best_label || e.block_reason || "").includes(filterLabel)) return false;
      return true;
    })
    .slice(0, 100);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Sub-header */}
      <div className="flex items-center justify-between px-6 py-3 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-bold text-white">Security Operations Center</h1>
          <div className="flex items-center gap-1.5 text-xs text-green-400">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse inline-block" />
            System Active
          </div>
        </div>
        <div className="flex items-center gap-4">
          <LiveClock />
          {/* Filters */}
          <div className="hidden lg:flex items-center gap-2">
            <input
              type="text"
              value={filterIp}
              onChange={(e) => setFilterIp(e.target.value)}
              placeholder="Filter IP…"
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 w-24"
            />
            <select
              value={filterAction}
              onChange={(e) => setFilterAction(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Actions</option>
              {["BLOCK", "ALLOW", "THROTTLE", "DELAY", "CAPTCHA"].map((a) => <option key={a}>{a}</option>)}
            </select>
            <select
              value={filterLabel}
              onChange={(e) => setFilterLabel(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Labels</option>
              {["xss", "sqli", "path_traversal", "rce", "anomaly", "clean"].map((l) => <option key={l}>{l}</option>)}
            </select>
          </div>
          <button
            onClick={logout}
            className="text-xs border border-slate-700 hover:border-red-600 hover:text-red-400 text-slate-400 px-3 py-1.5 rounded-lg transition-colors"
          >
            Logout
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="p-5 space-y-5">
        <StatsCards stats={stats} />

        <div className="space-y-4">
          <ChartsRow stats={stats} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <GlobePanel events={allEvents} />
        </div>

        <TopThreatIPs stats={stats} onLoadBlocked={loadBlockedIPs} />

        <AiThreatAnalysis stats={stats} events={allEvents} />

        <EventsTable
          events={filtered}
          paused={paused}
          onPause={() => setPaused((p) => !p)}
          onSelectEvent={setSelectedEvent}
        />
      </div>

      {selectedEvent && (
        <EventDetailModal event={selectedEvent} onClose={() => setSelectedEvent(null)} />
      )}
      {showBlocked && (
        <BlockedIPsModal blockedIPs={blockedIPs} onClose={() => setShowBlocked(false)} />
      )}
    </div>
  );
}
