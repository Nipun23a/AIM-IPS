export default function Badge({ text, action }) {
  const styles = {
    BLOCK:    "bg-red-900 text-red-300 border-red-700",
    CAPTCHA:  "bg-yellow-900 text-yellow-300 border-yellow-700",
    THROTTLE: "bg-purple-900 text-purple-300 border-purple-700",
    DELAY:    "bg-blue-900 text-blue-300 border-blue-700",
    ALLOW:    "bg-green-900 text-green-300 border-green-700",
    SKIPPED:  "bg-slate-800 text-slate-400 border-slate-600",
    DISABLED: "bg-slate-800 text-slate-500 border-slate-700",
  };
  const cls = styles[action || text] || "bg-slate-800 text-slate-400 border-slate-600";
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded border uppercase tracking-wider ${cls}`}>
      {text}
    </span>
  );
}
