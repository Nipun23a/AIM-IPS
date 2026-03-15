import { useEffect, useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend, Filler,
} from "chart.js";
import { Line, Doughnut, Bar } from "react-chartjs-2";
import { ATTACK_COLORS } from "../../constants";

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, ArcElement, Title, Tooltip, Legend, Filler,
);

const GRID  = "#1e2938";
const TICK  = "#6b7280";
const LABEL = "#8b949e";

const baseScales = {
  x: { ticks: { color: TICK, maxTicksLimit: 6 }, grid: { color: GRID } },
  y: { ticks: { color: TICK }, grid: { color: GRID }, beginAtZero: true },
};

export default function ChartsRow({ stats }) {
  const lineRef   = useRef(null);
  const donutRef  = useRef(null);
  const barRef    = useRef(null);

  // Imperatively push new data when stats change (avoids full re-render flicker)
  useEffect(() => {
    if (!stats) return;

    if (lineRef.current) {
      const c = lineRef.current;
      const labels = stats.rpm_total.map((_, i) => `-${stats.rpm_total.length - 1 - i}m`);
      c.data.labels                  = labels;
      c.data.datasets[0].data        = stats.rpm_total;
      c.data.datasets[1].data        = stats.rpm_blocked;
      c.update("none");
    }

    if (donutRef.current) {
      const c  = donutRef.current;
      const at = stats.attack_types || {};
      c.data.labels               = Object.keys(at);
      c.data.datasets[0].data     = Object.values(at);
      c.data.datasets[0].backgroundColor = ATTACK_COLORS.slice(0, Object.keys(at).length);
      c.update("none");
    }

    if (barRef.current) {
      const c  = barRef.current;
      const lc = stats.layer_counts || {};
      c.data.labels           = Object.keys(lc).map((k) => k.replace("layer", "L").replace("_", " "));
      c.data.datasets[0].data = Object.values(lc);
      c.update("none");
    }
  }, [stats]);

  const lineData = {
    labels: [],
    datasets: [
      { label: "Total",   data: [], borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.1)",  tension: 0.4, fill: true, pointRadius: 2 },
      { label: "Blocked", data: [], borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,.08)", tension: 0.4, fill: true, pointRadius: 2 },
    ],
  };
  const lineOpts = {
    responsive: true, animation: false,
    plugins: { legend: { labels: { color: LABEL, boxWidth: 10, font: { size: 11 } } } },
    scales: baseScales,
  };

  const donutData = {
    labels: [],
    datasets: [{ data: [], backgroundColor: ATTACK_COLORS, borderWidth: 0 }],
  };
  const donutOpts = {
    responsive: true, animation: false, cutout: "65%",
    plugins: { legend: { position: "right", labels: { color: LABEL, boxWidth: 10, font: { size: 11 } } } },
  };

  const barData = {
    labels: [],
    datasets: [{ label: "Detections", data: [], backgroundColor: "rgba(139,92,246,.7)", borderColor: "#7c3aed", borderWidth: 1 }],
  };
  const barOpts = {
    responsive: true, animation: false, indexAxis: "y",
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: TICK }, grid: { color: GRID }, beginAtZero: true },
      y: { ticks: { color: LABEL }, grid: { display: false } },
    },
  };

  return (
    <>
      {/* RPM + Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-white text-sm">
              Requests / Minute <span className="text-xs text-slate-500">(last 30 min)</span>
            </h2>
            <div className="flex gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 rounded inline-block" />Total</span>
              <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 rounded inline-block" />Blocked</span>
            </div>
          </div>
          <Line ref={lineRef} data={lineData} options={lineOpts} height={120} />
        </div>
        <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
          <h2 className="font-semibold text-white text-sm mb-3">Attack Types</h2>
          <Doughnut ref={donutRef} data={donutData} options={donutOpts} height={200} />
        </div>
      </div>

      {/* Layer bar + Globe slot */}
      <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 lg:col-span-1">
        <h2 className="font-semibold text-white text-sm mb-3">Detection Layers</h2>
        <Bar ref={barRef} data={barData} options={barOpts} height={220} />
      </div>
    </>
  );
}
