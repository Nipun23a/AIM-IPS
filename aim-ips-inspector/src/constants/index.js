export const PRESETS = [
  { label: "Clean GET request",            method: "GET",  path: "/api/probe", headers: {}, params: {},                                            body: "" },
  { label: "Clean POST JSON",              method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: '{"username":"alice","page":1}' },
  { label: "SQLi — OR bypass",             method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: "username=admin'--&password=x" },
  { label: "SQLi — UNION SELECT",          method: "GET",  path: "/api/probe", headers: {}, params: { q: "' UNION SELECT NULL,username,password FROM users--" }, body: "" },
  { label: "SQLi — time based blind",      method: "GET",  path: "/api/probe", headers: {}, params: { id: "1; SELECT SLEEP(5)--" },                body: "" },
  { label: "XSS — script tag",             method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: "<script>alert(document.cookie)</script>" },
  { label: "XSS — event handler",          method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: "<img src=x onerror=alert(1)>" },
  { label: "Path traversal — /etc/passwd", method: "GET",  path: "/api/probe", headers: {}, params: { file: "../../../etc/passwd" },               body: "" },
  { label: "CMDi — semicolon cat",         method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: "host=127.0.0.1; cat /etc/passwd" },
  { label: "Bad UA — sqlmap",              method: "GET",  path: "/api/probe", headers: { "User-Agent": "sqlmap/1.7" }, params: {},                body: "" },
  { label: "Bad UA — nikto",               method: "GET",  path: "/api/probe", headers: { "User-Agent": "Nikto/2.1.6" }, params: {},               body: "" },
  { label: "Zero-day anomaly payload",     method: "POST", path: "/api/probe", headers: {}, params: {},                                            body: btoa("AAAA".repeat(200)) },
];

export const DEFAULT_IP     = "127.0.0.1";
export const DEFAULT_PATH   = "/api/probe";
export const DEFAULT_METHOD = "GET";

export const LAYER_WEIGHTS = {
  layer1:      0.20,
  layer2_lgbm: 0.35,
  layer2_cnn:  0.25,
  network:     0.20,
};

export const SERVER_LOC = {
  lat: parseFloat(process.env.REACT_APP_SERVER_LAT) || 6.9271,
  lng: parseFloat(process.env.REACT_APP_SERVER_LNG) || 79.8612,
};

export const ATTACK_COLORS = [
  "#ef4444", "#f97316", "#eab308", "#22c55e",
  "#06b6d4", "#8b5cf6", "#ec4899", "#14b8a6",
];

export const LAYER_META = [
  { key: "layer0",      label: "Layer 0",  name: "Static Firewall", desc: "IP blacklist, bad UA, rate limits" },
  { key: "layer1",      label: "Layer 1",  name: "Regex Filter",    desc: "SQLi, XSS, CMDi, traversal patterns" },
  { key: "network",     label: "Network",  name: "Redis Score",     desc: "Async network threat intelligence" },
  { key: "layer2_cnn",  label: "Layer 2b", name: "CNN Gate",        desc: "Anomaly gate — flags anomalous traffic" },
  { key: "layer2_lgbm", label: "Layer 2a", name: "LightGBM App",    desc: "Threat classifier — runs after CNN gate" },
];

export const ADMIN_USERNAME = process.env.REACT_APP_ADMIN_USERNAME || "admin";
export const ADMIN_PASSWORD = process.env.REACT_APP_ADMIN_PASSWORD || "aimips2024";
