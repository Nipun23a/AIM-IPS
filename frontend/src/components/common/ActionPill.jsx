export default function ActionPill({ action }) {
  const cls = {
    BLOCK:    "bg-red-900/80 text-red-300 border border-red-700",
    ALLOW:    "bg-green-900/80 text-green-300 border border-green-700",
    THROTTLE: "bg-purple-900/80 text-purple-300 border border-purple-700",
    DELAY:    "bg-blue-900/80 text-blue-300 border border-blue-700",
    CAPTCHA:  "bg-yellow-900/80 text-yellow-300 border border-yellow-700",
  }[action] || "bg-slate-800 text-slate-400 border border-slate-600";

  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded-full uppercase ${cls}`}>
      {action}
    </span>
  );
}
