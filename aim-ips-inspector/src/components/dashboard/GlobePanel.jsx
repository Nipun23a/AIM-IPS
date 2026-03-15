import { useEffect, useRef } from "react";
import Globe from "globe.gl";
import { SERVER_LOC } from "../../constants";

export default function GlobePanel({ events }) {
  const containerRef = useRef(null);
  const globeRef     = useRef(null);
  const lastUpdate   = useRef(0);

  // Init globe once
  useEffect(() => {
    if (!containerRef.current || globeRef.current) return;
    try {
      globeRef.current = Globe()(containerRef.current)
        .globeImageUrl("https://unpkg.com/three-globe/example/img/earth-night.jpg")
        .backgroundImageUrl("https://unpkg.com/three-globe/example/img/night-sky.png")
        .showAtmosphere(true)
        .atmosphereColor("#1a3a5c")
        .atmosphereAltitude(0.15)
        .arcColor(() => "#ef4444")
        .arcOpacity(0.7)
        .arcDashLength(0.4)
        .arcDashGap(0.2)
        .arcDashAnimateTime(1800)
        .arcStroke(0.5)
        .pointColor((d) => (d.isServer ? "#3fb950" : "#ef4444"))
        .pointAltitude(0.01)
        .pointRadius((d) => (d.isServer ? 0.6 : 0.3))
        .pointLabel(
          (d) =>
            `<div style="color:#fff;font-size:11px;background:rgba(0,0,0,.75);padding:3px 8px;border-radius:5px">${d.label}</div>`,
        );
      globeRef.current.controls().autoRotate      = true;
      globeRef.current.controls().autoRotateSpeed = 0.4;
      globeRef.current.pointOfView({ lat: 20, lng: 50, altitude: 2.2 }, 1000);
    } catch (e) {
      console.error("Globe init failed", e);
    }
    return () => {
      globeRef.current = null;
    };
  }, []);

  // Update points/arcs when events change (rate-limited to 30s)
  useEffect(() => {
    if (!globeRef.current || !events.length) return;
    const now = Date.now();
    if (now - lastUpdate.current < 30000) return;
    lastUpdate.current = now;

    const ips = [
      ...new Set(
        events.filter((e) => e.action === "BLOCK" || e.final_score > 0.4).map((e) => e.ip),
      ),
    ].slice(0, 30);
    if (!ips.length) return;

    fetch("https://ip-api.com/batch?fields=query,lat,lon", {
      method: "POST",
      body: JSON.stringify(ips.map((ip) => ({ query: ip }))),
      headers: { "Content-Type": "application/json" },
    })
      .then((r) => r.json())
      .then((geos) => {
        const points = [
          { lat: SERVER_LOC.lat, lng: SERVER_LOC.lng, isServer: true, label: "🛡️ AIM-IPS Server (Colombo)" },
        ];
        const arcs = [];
        geos.forEach((g) => {
          if (!g.lat || !g.lon) return;
          points.push({ lat: g.lat, lng: g.lon, isServer: false, label: g.query });
          arcs.push({ startLat: g.lat, startLng: g.lon, endLat: SERVER_LOC.lat, endLng: SERVER_LOC.lng });
        });
        globeRef.current?.pointsData(points).arcsData(arcs);
      })
      .catch(() => {});
  }, [events]);

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700/60 p-4 lg:col-span-2">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-white text-sm">
          Live Threat Map <span className="text-xs text-slate-500">→ Colombo, LK</span>
        </h2>
        <span className="text-xs text-slate-500">Showing attack arcs (last 30 blocked IPs)</span>
      </div>
      <div ref={containerRef} style={{ height: 320, borderRadius: 8, overflow: "hidden" }} />
    </div>
  );
}
