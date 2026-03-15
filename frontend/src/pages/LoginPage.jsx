import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { ADMIN_USERNAME, ADMIN_PASSWORD } from "../constants";

export default function LoginPage() {
  const [user,    setUser]    = useState("");
  const [pass,    setPass]    = useState("");
  const [err,     setErr]     = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Already logged in
  useEffect(() => {
    if (sessionStorage.getItem("ips_auth") === "1") {
      navigate("/admin-dashboard", { replace: true });
    }
  }, [navigate]);

  const login = () => {
    setLoading(true);
    setTimeout(() => {
      if (user === ADMIN_USERNAME && pass === ADMIN_PASSWORD) {
        sessionStorage.setItem("ips_auth", "1");
        navigate("/admin-dashboard", { replace: true });
      } else {
        setErr(true);
        setLoading(false);
      }
    }, 400); // brief delay for UX
  };

  const handleKey = (e) => { if (e.key === "Enter") login(); };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center px-4">
      {/* Background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-indigo-600/10 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-sm">
        {/* Card */}
        <div className="bg-slate-900 rounded-2xl border border-slate-800 p-8 shadow-2xl shadow-black/50">
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-xl shadow-indigo-500/30">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
            </div>
            <h1 className="text-2xl font-black text-white">Admin Access</h1>
            <p className="text-sm text-slate-500 mt-1">AIM-IPS Security Operations</p>
          </div>

          {/* Form */}
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1.5">Username</label>
              <input
                type="text"
                value={user}
                onChange={(e) => { setUser(e.target.value); setErr(false); }}
                onKeyDown={handleKey}
                placeholder="admin"
                className={`w-full bg-slate-800 border rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none transition-colors ${
                  err ? "border-red-600 focus:border-red-500" : "border-slate-700 focus:border-indigo-500"
                }`}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 uppercase tracking-wider block mb-1.5">Password</label>
              <input
                type="password"
                value={pass}
                onChange={(e) => { setPass(e.target.value); setErr(false); }}
                onKeyDown={handleKey}
                placeholder="••••••••"
                className={`w-full bg-slate-800 border rounded-xl px-4 py-2.5 text-sm text-slate-200 focus:outline-none transition-colors ${
                  err ? "border-red-600 focus:border-red-500" : "border-slate-700 focus:border-indigo-500"
                }`}
              />
            </div>

            {err && (
              <div className="flex items-center gap-2 text-red-400 text-xs bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
                Invalid credentials. Please try again.
              </div>
            )}

            <button
              onClick={login}
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-semibold py-2.5 rounded-xl transition-colors duration-150 text-sm flex items-center justify-center gap-2 mt-2"
            >
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Signing in…
                </>
              ) : "Sign In"}
            </button>
          </div>
        </div>

        {/* Back link */}
        <div className="text-center mt-5">
          <Link to="/" className="text-sm text-slate-500 hover:text-slate-300 transition-colors inline-flex items-center gap-1.5">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5L3 12m0 0l7.5-7.5M3 12h18" />
            </svg>
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
}
