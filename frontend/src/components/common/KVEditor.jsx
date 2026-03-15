export default function KVEditor({ title, entries, onChange }) {
  const add    = () => onChange([...entries, { k: "", v: "" }]);
  const remove = (i) => onChange(entries.filter((_, j) => j !== i));
  const update = (i, field, val) => {
    const next = [...entries];
    next[i] = { ...next[i], [field]: val };
    onChange(next);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400 uppercase tracking-wider">{title}</span>
        <button onClick={add} className="text-xs text-indigo-400 hover:text-indigo-300">+ Add</button>
      </div>
      {entries.map((e, i) => (
        <div key={i} className="flex gap-1 mb-1">
          <input
            className="flex-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            placeholder="key"
            value={e.k}
            onChange={(ev) => update(i, "k", ev.target.value)}
          />
          <input
            className="flex-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            placeholder="value"
            value={e.v}
            onChange={(ev) => update(i, "v", ev.target.value)}
          />
          <button onClick={() => remove(i)} className="text-slate-500 hover:text-red-400 text-xs px-1">✕</button>
        </div>
      ))}
    </div>
  );
}
