import { useState } from "react";
import RequestBuilder   from "../components/inspector/RequestBuilder";
import LayerToggle      from "../components/inspector/LayerToggle";
import WeightPreview    from "../components/inspector/WeightPreview";
import Layer0Card       from "../components/inspector/Layer0Card";
import Layer1Card       from "../components/inspector/Layer1Card";
import NetworkCard      from "../components/inspector/NetworkCard";
import Layer2CNNCard    from "../components/inspector/Layer2CNNCard";
import Layer2LGBMCard   from "../components/inspector/Layer2LGBMCard";
import FinalCard        from "../components/inspector/FinalCard";
import { DEFAULT_IP, DEFAULT_PATH, DEFAULT_METHOD, LAYER_META } from "../constants";

export default function InspectorPage() {
  const [method,  setMethod]  = useState(DEFAULT_METHOD);
  const [path,    setPath]    = useState(DEFAULT_PATH);
  const [ip,      setIp]      = useState(DEFAULT_IP);
  const [body,    setBody]    = useState("");
  const [headers, setHeaders] = useState([]);
  const [params,  setParams]  = useState([]);
  const [preset,  setPreset]  = useState("");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);
  const [layers,  setLayers]  = useState({
    layer0: true, layer1: true, network: true, layer2_lgbm: true, layer2_cnn: true,
  });

  const toggleLayer = (key, val) => setLayers((prev) => ({ ...prev, [key]: val }));
  const kvToObj = (arr) => Object.fromEntries(arr.filter((e) => e.k).map((e) => [e.k, e.v]));

  const send = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch("/api/inspect", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ip, method, path,
          headers: kvToObj(headers),
          query_params: kvToObj(params),
          body,
          layers_enabled: layers,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Page header */}
      <div className="border-b border-slate-800 bg-slate-900/50 px-6 py-4">
        <div className="max-w-[1400px] mx-auto">
          <h1 className="text-lg font-bold text-white">Pipeline Inspector</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Toggle layers on/off · see per-layer scores · inspect every decision
          </p>
        </div>
      </div>

      <div className="p-4 md:p-6">
        <div className="max-w-[1400px] mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-[320px_260px_1fr] gap-4">

            {/* LEFT: Request Builder */}
            <div className="space-y-3">
              <RequestBuilder
                method={method}    setMethod={setMethod}
                path={path}        setPath={setPath}
                ip={ip}            setIp={setIp}
                body={body}        setBody={setBody}
                headers={headers}  setHeaders={setHeaders}
                params={params}    setParams={setParams}
                preset={preset}    setPreset={setPreset}
                loading={loading}  error={error}
                onSend={send}
              />
            </div>

            {/* MIDDLE: Layer Toggles */}
            <div>
              <div className="bg-slate-800/80 rounded-xl border border-slate-700/50 p-4 md:sticky md:top-[73px]">
                <h2 className="text-xs text-slate-400 uppercase tracking-wider mb-1">Layer Toggles</h2>
                <p className="text-xs text-slate-600 mb-3">Disabled layers still run — scores excluded from fusion</p>
                <div>
                  {LAYER_META.map((m) => (
                    <LayerToggle
                      key={m.key}
                      layerKey={m.key}
                      meta={m}
                      enabled={layers[m.key]}
                      onChange={toggleLayer}
                      liveResult={result}
                    />
                  ))}
                </div>
                <WeightPreview layers={layers} />
              </div>
            </div>

            {/* RIGHT: Results */}
            <div className="space-y-3">
              {!result && !loading && (
                <div className="flex items-center justify-center h-64 bg-slate-800/40 rounded-xl border border-slate-700/50 border-dashed">
                  <div className="text-center text-slate-600">
                    <div className="text-4xl mb-2">⟳</div>
                    <div className="text-sm">Send a request to see layer results</div>
                  </div>
                </div>
              )}
              {loading && (
                <div className="flex items-center justify-center h-64 bg-slate-800/40 rounded-xl border border-slate-700/50">
                  <div className="text-center">
                    <div className="w-8 h-8 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin mx-auto mb-2" />
                    <div className="text-sm text-slate-500">Running pipeline…</div>
                  </div>
                </div>
              )}
              {result && !loading && (
                <>
                  {result.short_circuited && (
                    <div className="bg-red-900/30 border border-red-800 rounded-lg px-3 py-2 text-xs text-red-300">
                      ⚡ Short-circuited at <span className="font-bold">{result.short_circuit_at}</span> — Layer 2 was not run in normal pipeline flow
                    </div>
                  )}
                  <Layer0Card     data={result.layers?.layer0} />
                  <Layer1Card     data={result.layers?.layer1} shortCircuitAt={result.short_circuit_at} />
                  <NetworkCard    data={result.layers?.network} />
                  <Layer2CNNCard  data={result.layers?.layer2_cnn} />
                  <Layer2LGBMCard data={result.layers?.layer2_lgbm} />
                  <FinalCard      result={result} />
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
