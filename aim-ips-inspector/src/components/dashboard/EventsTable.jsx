import ActionPill from "../common/ActionPill";
import { fmtTime, scoreTextColor } from "../../utils/helpers";

export default function EventsTable({ events, paused, onPause, onSelectEvent }) {
  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <h2 className="font-semibold text-white text-sm">Recent Events</h2>
        <div className="ml-auto flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full inline-block ${paused ? "bg-red-500" : "bg-green-500 animate-pulse"}`} />
          <span className="text-xs text-slate-500">{paused ? "Paused" : "Auto-refresh 5s"}</span>
          <button
            onClick={onPause}
            className={`text-xs border px-3 py-1 rounded-lg ${
              paused
                ? "border-green-700 text-green-400 hover:border-green-500"
                : "border-slate-700 text-slate-400 hover:border-yellow-600 hover:text-yellow-400"
            }`}
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>
      <div className="overflow-auto" style={{ maxHeight: 420 }}>
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10">
            <tr className="border-b border-slate-700 bg-slate-800">
              {["Time", "IP", "Method", "Path", "Score", "Action", "Label", "Latency"].map((h) => (
                <th key={h} className="text-left text-xs text-slate-500 uppercase tracking-wide py-2 px-3 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {events.length === 0 ? (
              <tr><td colSpan={8} className="text-slate-500 text-center py-8 text-sm">No events</td></tr>
            ) : (
              events.map((e, i) => (
                <tr
                  key={i}
                  onClick={() => onSelectEvent(e)}
                  className="border-b border-slate-700/40 hover:bg-slate-700/40 cursor-pointer"
                >
                  <td className="py-2 px-3 font-mono text-slate-400 text-xs whitespace-nowrap">{fmtTime(e.timestamp)}</td>
                  <td className="py-2 px-3 font-mono text-blue-300 text-xs">{e.ip}</td>
                  <td className="py-2 px-3 text-slate-300 text-xs">{e.method}</td>
                  <td className="py-2 px-3 font-mono text-slate-300 text-xs max-w-xs truncate">{e.path}</td>
                  <td className={`py-2 px-3 font-mono font-semibold text-xs ${scoreTextColor(e.final_score)}`}>
                    {(e.final_score || 0).toFixed(3)}
                  </td>
                  <td className="py-2 px-3"><ActionPill action={e.action} /></td>
                  <td className="py-2 px-3 text-slate-400 text-xs">{e.best_label || e.block_reason || ""}</td>
                  <td className="py-2 px-3 text-slate-500 text-xs">{e.latency_ms != null ? `${e.latency_ms}ms` : ""}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
