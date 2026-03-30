export default function ScoreBar({ score, mini = false }) {
  const pct = Math.round((score || 0) * 100);
  const color =
    pct >= 70 ? "bg-red-500" :
    pct >= 50 ? "bg-yellow-400" :
    pct >= 35 ? "bg-purple-500" :
    pct >= 25 ? "bg-blue-500" :
    "bg-green-500";

  return (
    <div className={`bg-slate-900 rounded-full overflow-hidden ${mini ? "h-1.5 w-20 inline-block align-middle" : "h-2"}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
