import KVEditor from "../common/KVEditor";
import { PRESETS } from "../../constants";

export default function RequestBuilder({
  method, setMethod,
  path, setPath,
  ip, setIp,
  body, setBody,
  headers, setHeaders,
  params, setParams,
  preset, setPreset,
  loading, error,
  onSend,
}) {
  const loadPreset = (idx) => {
    if (idx === "") return;
    const p = PRESETS[parseInt(idx)];
    if (!p) return;
    setPreset(idx);
    setMethod(p.method);
    setPath(p.path);
    setBody(p.body);
    setHeaders(Object.entries(p.headers).map(([k, v]) => ({ k, v })));
    setParams(Object.entries(p.params).map(([k, v]) => ({ k, v })));
  };

  return (
    <div className="bg-slate-800/80 rounded-xl border border-slate-700/50 p-4">
      <h2 className="text-xs text-slate-400 uppercase tracking-wider mb-3">Request Builder</h2>

      <div className="mb-3">
        <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Preset</label>
        <select
          value={preset}
          onChange={(e) => loadPreset(e.target.value)}
          className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
        >
          <option value="">— select preset —</option>
          {PRESETS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
        </select>
      </div>

      <div className="flex gap-2 mb-3">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          className="w-24 bg-slate-900 border border-slate-600 rounded px-2 py-1.5 text-sm text-indigo-400 font-bold focus:outline-none focus:border-indigo-500"
        >
          {["GET", "POST", "PUT", "DELETE"].map((m) => <option key={m}>{m}</option>)}
        </select>
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          className="flex-1 bg-slate-900 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
          placeholder="/api/probe"
        />
      </div>

      <div className="mb-3">
        <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">IP Override</label>
        <input
          value={ip}
          onChange={(e) => setIp(e.target.value)}
          className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
          placeholder="127.0.0.1"
        />
      </div>

      <div className="mb-3">
        <KVEditor title="Headers" entries={headers} onChange={setHeaders} />
      </div>
      <div className="mb-3">
        <KVEditor title="Query Params" entries={params} onChange={setParams} />
      </div>

      <div className="mb-4">
        <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1">Body</label>
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={4}
          className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 font-mono focus:outline-none focus:border-indigo-500 resize-none"
          placeholder="Request body..."
        />
      </div>

      <button
        onClick={onSend}
        disabled={loading}
        className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-bold py-2.5 rounded-lg transition-colors duration-150 text-sm flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block" />
            Sending…
          </>
        ) : "▶  SEND"}
      </button>

      {error && (
        <div className="mt-2 text-xs text-red-400 bg-red-900/20 rounded p-2 border border-red-800">
          {error}
        </div>
      )}
    </div>
  );
}
