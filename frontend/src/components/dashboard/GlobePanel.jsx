import { useEffect, useRef, useState } from "react";
import {
  ComposableMap, Geographies, Geography,
  Line as MapLine, Marker,
} from "react-simple-maps";
import Globe from "globe.gl";
import { SERVER_LOC } from "../../constants";

const GEO_URL   = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";
const FLAG_BASE = "https://flagcdn.com/20x15";

const LABEL_COLOR = {
  sqli:           "#ef4444",
  xss:            "#f97316",
  path_traversal: "#eab308",
  rce:            "#a855f7",
  anomaly:        "#06b6d4",
  default:        "#ef4444",
};
function colorFor(label) { return LABEL_COLOR[label] || LABEL_COLOR.default; }

/* ── 2D flat map ─────────────────────────────────────────────────────── */
function FlatMap({ feed, arcs, pulse }) {
  return (
    <div className="flex-1 rounded-lg overflow-hidden bg-slate-900/60" style={{ minHeight: 360 }}>
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 110, center: [20, 15] }}
        style={{ width: "100%", height: 360 }}
      >
        <Geographies geography={GEO_URL}>
          {({ geographies }) =>
            geographies.map((geo) => (
              <Geography
                key={geo.rsmKey} geography={geo}
                fill="#1e2938" stroke="#2d3f55" strokeWidth={0.4}
                style={{
                  default: { outline: "none" },
                  hover:   { fill: "#243447", outline: "none" },
                  pressed: { outline: "none" },
                }}
              />
            ))
          }
        </Geographies>

        {arcs.map((arc, i) => (
          <MapLine
            key={i} from={arc.from} to={arc.to}
            stroke={arc.color}
            strokeWidth={pulse ? 1.4 : 0.7}
            strokeOpacity={pulse ? 0.9 : 0.45}
            strokeLinecap="round"
            style={{ transition: "stroke-opacity 1s ease, stroke-width 1s ease" }}
          />
        ))}

        {feed.map((f, i) => (
          <Marker key={i} coordinates={[f.lon, f.lat]}>
            <circle r={3} fill={f.color} fillOpacity={0.9} stroke="#0f172a" strokeWidth={0.5} />
          </Marker>
        ))}

        <Marker coordinates={[SERVER_LOC.lng, SERVER_LOC.lat]}>
          <circle r={7} fill="none" stroke="#3fb950" strokeWidth={1.5} />
          <circle r={3.5} fill="#3fb950" />
          <text textAnchor="middle" y={-12}
            style={{ fontSize: 8, fill: "#3fb950", fontFamily: "monospace", fontWeight: 600 }}>
            AIM-IPS
          </text>
        </Marker>
      </ComposableMap>
    </div>
  );
}

/* ── 3D globe ────────────────────────────────────────────────────────── */
function Globe3D({ feed, arcs }) {
  const containerRef = useRef(null);
  const globeRef     = useRef(null);

  useEffect(() => {
    if (!containerRef.current || globeRef.current) return;
    try {
      globeRef.current = Globe()(containerRef.current)
        .globeImageUrl("https://unpkg.com/three-globe/example/img/earth-night.jpg")
        .backgroundImageUrl("https://unpkg.com/three-globe/example/img/night-sky.png")
        .showAtmosphere(true)
        .atmosphereColor("#1a3a5c")
        .atmosphereAltitude(0.15)
        .arcColor((d) => [d.color.replace(")", ",0.95)").replace("rgb", "rgba"), d.color.replace(")", ",0.05)").replace("rgb", "rgba")])
        .arcDashLength(0.4)
        .arcDashGap(0.2)
        .arcDashAnimateTime(1600)
        .arcStroke(0.6)
        .pointColor((d) => d.color || "#ef4444")
        .pointAltitude(0.01)
        .pointRadius((d) => (d.isServer ? 0.7 : 0.35))
        .pointLabel((d) =>
          `<div style="color:#fff;font-size:11px;background:rgba(0,0,0,.85);padding:4px 10px;border-radius:6px">${d.label}</div>`
        );

      const ctrl = globeRef.current.controls();
      ctrl.autoRotate      = true;
      ctrl.autoRotateSpeed = 0.35;
      globeRef.current.pointOfView({ lat: 20, lng: 50, altitude: 2.0 }, 800);

      const el = containerRef.current;
      const pause  = () => { ctrl.autoRotate = false; };
      const resume = () => { ctrl.autoRotate = true; };
      el.addEventListener("mouseenter", pause);
      el.addEventListener("mouseleave", resume);

      return () => {
        el.removeEventListener("mouseenter", pause);
        el.removeEventListener("mouseleave", resume);
        globeRef.current = null;
      };
    } catch (e) { console.error("Globe init failed", e); }
  }, []);

  // Sync data
  useEffect(() => {
    if (!globeRef.current) return;
    const points = [
      { lat: SERVER_LOC.lat, lng: SERVER_LOC.lng, isServer: true,
        color: "#3fb950", label: "🛡️ AIM-IPS Server (Colombo)" },
      ...feed.map((f) => ({
        lat: f.lat, lng: f.lon, isServer: false, color: f.color,
        label: `${f.ip} · ${f.country} · ${f.label}`,
      })),
    ];
    const globeArcs = arcs.map((a) => ({
      startLat: a.from[1], startLng: a.from[0],
      endLat:   a.to[1],   endLng:   a.to[0],
      color: a.color,
    }));
    globeRef.current.pointsData(points).arcsData(globeArcs);
  }, [feed, arcs]);

  return (
    <div
      ref={containerRef}
      className="flex-1 rounded-lg overflow-hidden cursor-grab active:cursor-grabbing"
      style={{ minHeight: 360 }}
    />
  );
}

/* ── Main panel ──────────────────────────────────────────────────────── */
export default function GlobePanel({ events }) {
  const lastUpdate    = useRef(0);
  const [view,        setView]        = useState("2d");   // "2d" | "3d"
  const [feed,        setFeed]        = useState([]);
  const [arcs,        setArcs]        = useState([]);
  const [loading,     setLoading]     = useState(false);
  const [attackCount, setAttackCount] = useState(0);
  const [pulse,       setPulse]       = useState(false);

  useEffect(() => {
    const t = setInterval(() => setPulse((p) => !p), 2000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!events.length) return;
    const now = Date.now();
    if (now - lastUpdate.current < 10000) return;
    lastUpdate.current = now;

    const blocked = events.filter((e) => e.action === "BLOCK" || e.final_score > 0.4);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAttackCount(blocked.length);

    const seen = new Map();
    blocked.forEach((e) => { if (!seen.has(e.ip)) seen.set(e.ip, e); });
    const uniqueIps = [...seen.entries()].slice(0, 25);
    if (!uniqueIps.length) return;

    setLoading(true);
    Promise.allSettled(
      uniqueIps.map(([ip]) =>
        fetch(`https://ipwho.is/${ip}`)
          .then((r) => r.json())
      )
    ).then((results) => {
      const newFeed = [];
      const newArcs = [];
      results.forEach(({ status, value: g }) => {
        if (status !== "fulfilled" || !g?.latitude || !g?.longitude) return;
        const ev    = seen.get(g.ip) || {};
        const label = ev.best_label || ev.block_reason || "threat";
        const color = colorFor(label);
        newFeed.push({
          ip: g.ip, country: g.country || "Unknown",
          countryCode: (g.country_code || "").toLowerCase(),
          lat: g.latitude, lon: g.longitude, label, color,
        });
        newArcs.push({
          from: [g.longitude, g.latitude],
          to:   [SERVER_LOC.lng, SERVER_LOC.lat],
          label, color,
        });
      });
      setFeed(newFeed);
      setArcs(newArcs);
    }).finally(() => setLoading(false));
  }, [events]);

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 lg:col-span-2">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h2 className="font-semibold text-white text-sm">Live Threat Map</h2>
          <span className="text-xs text-slate-500">→ Colombo, LK</span>
          {attackCount > 0 && (
            <span className="text-xs bg-red-500/20 text-red-400 border border-red-500/30 px-2 py-0.5 rounded-full">
              {attackCount} attacks
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {loading && (
            <span className="flex items-center gap-1 text-xs text-blue-400">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse inline-block" />
              Locating…
            </span>
          )}
          {/* View toggle */}
          <div className="flex items-center bg-slate-900 border border-slate-700 rounded-lg p-0.5 gap-0.5">
            <button
              onClick={() => setView("2d")}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                view === "2d"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              2D Map
            </button>
            <button
              onClick={() => setView("3d")}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                view === "3d"
                  ? "bg-indigo-600 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              3D Globe
            </button>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="flex gap-3">
        {view === "2d"
          ? <FlatMap feed={feed} arcs={arcs} pulse={pulse} />
          : <Globe3D feed={feed} arcs={arcs} />
        }

        {/* Attack feed */}
        <div className="w-48 flex flex-col gap-1 overflow-y-auto flex-shrink-0" style={{ maxHeight: 360 }}>
          {feed.length === 0 ? (
            <div className="text-slate-500 text-xs text-center mt-10">
              {loading ? "Fetching…" : "No blocked IPs yet"}
            </div>
          ) : (
            feed.map((f, i) => (
              <div
                key={i}
                className="flex items-start gap-2 bg-slate-900/70 border border-slate-700/40 rounded-lg px-2.5 py-2 flex-shrink-0"
              >
                {f.countryCode && (
                  <img
                    src={`${FLAG_BASE}/${f.countryCode}.png`}
                    alt={f.country}
                    className="mt-0.5 rounded-sm flex-shrink-0"
                    width={20} height={15}
                    onError={(e) => { e.target.style.display = "none"; }}
                  />
                )}
                <div className="min-w-0">
                  <div className="text-white text-xs font-mono truncate">{f.ip}</div>
                  <div className="text-slate-400 text-[10px] truncate">{f.country}</div>
                  <span
                    className="inline-block text-[10px] px-1.5 py-0.5 rounded mt-0.5 font-medium"
                    style={{ background: f.color + "22", color: f.color, border: `1px solid ${f.color}44` }}
                  >
                    {f.label}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-slate-700/40">
        {Object.entries(LABEL_COLOR).filter(([k]) => k !== "default").map(([k, c]) => (
          <span key={k} className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: c }} />
            {k.replace(/_/g, " ")}
          </span>
        ))}
        <span className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" />
          server
        </span>
        {view === "3d" && (
          <span className="ml-auto text-[11px] text-slate-500">Hover to pause rotation</span>
        )}
      </div>
    </div>
  );
}
