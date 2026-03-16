import {
  Chart as ChartJS,
  CategoryScale, LinearScale,
  PointElement, LineElement, BarElement, ArcElement,
  Title, Tooltip, Legend, Filler,
} from "chart.js";
import { Line, Doughnut, Bar } from "react-chartjs-2";
import { useMemo } from "react";
import { ATTACK_COLORS } from "../../constants";

ChartJS.register(
  CategoryScale, LinearScale,
  PointElement, LineElement, BarElement, ArcElement,
  Title, Tooltip, Legend, Filler,
);

const GRID  = "#1e2938";
const TICK  = "#6b7280";
const LABEL = "#8b949e";

const LAYER_COLORS = [
  "rgba(99,102,241,0.85)",
  "rgba(59,130,246,0.85)",
  "rgba(16,185,129,0.85)",
  "rgba(139,92,246,0.85)",
  "rgba(6,182,212,0.85)",
];

const lineOpts = {
  responsive: true, animation: false,
  plugins: { legend: { labels: { color: LABEL, boxWidth: 10, font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: TICK, maxTicksLimit: 8 }, grid: { color: GRID } },
    y: { ticks: { color: TICK }, grid: { color: GRID }, beginAtZero: true },
  },
};

const donutOpts = {
  responsive: true, animation: false, cutout: "65%",
  plugins: { legend: { position: "right", labels: { color: LABEL, boxWidth: 8, font: { size: 10 }, padding: 8 } } },
};

const hBarOpts = {
  responsive: true, animation: false,
  indexAxis: "y",
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: TICK, font: { size: 11 } }, grid: { color: GRID }, beginAtZero: true },
    y: { ticks: { color: LABEL, font: { size: 11 } }, grid: { display: false } },
  },
};

export default function ChartsRow({ stats }) {
  const lineData = useMemo(() => {
    const rpm   = stats?.rpm_total   || [];
    const block = stats?.rpm_blocked || [];
    return {
      labels: rpm.map((_, i) => `-${rpm.length - 1 - i}m`),
      datasets: [
        { label: "Total",   data: rpm,   borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,.12)", tension: 0.4, fill: true, pointRadius: 2 },
        { label: "Blocked", data: block, borderColor: "#ef4444", backgroundColor: "rgba(239,68,68,.08)",  tension: 0.4, fill: true, pointRadius: 2 },
      ],
    };
  }, [stats]);

  const donutData = useMemo(() => {
    const at   = stats?.attack_types || {};
    const keys = Object.keys(at);
    return {
      labels: keys,
      datasets: [{ data: Object.values(at), backgroundColor: ATTACK_COLORS.slice(0, keys.length), borderWidth: 0 }],
    };
  }, [stats]);

  const layerData = useMemo(() => {
    const lc   = stats?.layer_counts || {};
    const keys = Object.keys(lc);
    return {
      labels: keys.map((k) => k.replace("layer", "Layer ").replace(/_/g, " ")),
      datasets: [{
        label: "Detections",
        data: Object.values(lc),
        backgroundColor: LAYER_COLORS.slice(0, keys.length),
        borderColor:     LAYER_COLORS.slice(0, keys.length).map((c) => c.replace("0.85", "1")),
        borderWidth: 1,
        borderRadius: 4,
        barThickness: 20,
      }],
    };
  }, [stats]);

  return (
    <div className="flex flex-col gap-4">
      {/* Row 1: RPM line + Attack Types donut */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-white text-sm">
              Requests / Minute <span className="text-xs text-slate-500">(last 30 min)</span>
            </h2>
            <div className="flex gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-blue-500 rounded inline-block" />Total
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-red-500 rounded inline-block" />Blocked
              </span>
            </div>
          </div>
          <Line data={lineData} options={lineOpts} height={100} />
        </div>

        <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
          <h2 className="font-semibold text-white text-sm mb-3">Attack Types</h2>
          <Doughnut data={donutData} options={donutOpts} height={160} />
        </div>
      </div>

      {/* Row 2: Detection Layers — full width horizontal bar */}
      <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4">
        <h2 className="font-semibold text-white text-sm mb-3">Detection Layers</h2>
        <Bar data={layerData} options={hBarOpts} height={90} />
      </div>
    </div>
  );
}
