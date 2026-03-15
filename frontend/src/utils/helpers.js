export function actionColor(action) {
  return {
    BLOCK:    "text-red-400",
    CAPTCHA:  "text-yellow-400",
    THROTTLE: "text-purple-400",
    DELAY:    "text-blue-400",
    ALLOW:    "text-green-400",
  }[action] || "text-slate-400";
}

export function actionBg(action) {
  return {
    BLOCK:    "bg-red-900/60 border-red-700",
    CAPTCHA:  "bg-yellow-900/60 border-yellow-700",
    THROTTLE: "bg-purple-900/60 border-purple-700",
    DELAY:    "bg-blue-900/60 border-blue-700",
    ALLOW:    "bg-green-900/60 border-green-700",
  }[action] || "bg-slate-800 border-slate-600";
}

export function decisionBorderColor(decision, enabled) {
  if (!enabled) return "border-slate-600";
  return {
    BLOCK:      "border-red-500",
    SUSPICIOUS: "border-yellow-500",
    CLEAN:      "border-green-600",
  }[decision] || "border-slate-600";
}

export function decisionDot(decision, enabled) {
  if (!enabled) return "bg-slate-500";
  return {
    BLOCK:      "bg-red-500",
    SUSPICIOUS: "bg-yellow-400",
    CLEAN:      "bg-green-500",
  }[decision] || "bg-slate-500";
}

export function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString("en-US", {
    hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

export function scoreBarFillColor(s) {
  if (s >= 0.7)  return "bg-red-500";
  if (s >= 0.5)  return "bg-orange-500";
  if (s >= 0.35) return "bg-yellow-400";
  if (s >= 0.25) return "bg-blue-500";
  return "bg-green-500";
}

export function scoreTextColor(s) {
  if (s >= 0.7)  return "text-red-400";
  if (s >= 0.5)  return "text-orange-400";
  if (s >= 0.35) return "text-yellow-400";
  if (s >= 0.25) return "text-blue-400";
  return "text-green-400";
}
