export default function BlockedIPsModal({ blockedIPs, onClose }) {
  return (
    <div
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between p-5 border-b border-slate-800">
          <h2 className="font-bold text-white">
            Blocked IPs <span className="text-slate-500 font-normal text-sm">({blockedIPs.length})</span>
          </h2>
          <button onClick={onClose} className="text-slate-500 hover:text-white text-2xl leading-none">×</button>
        </div>
        <div className="p-4 space-y-2">
          {blockedIPs.length === 0 ? (
            <p className="text-slate-500 text-sm text-center py-4">No blocked IPs</p>
          ) : (
            blockedIPs.map((b, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-slate-800 rounded-xl">
                <div>
                  <div className="font-mono text-red-300 text-sm">{b.ip}</div>
                  {b.reason && <div className="text-slate-500 text-xs mt-0.5">{b.reason}</div>}
                </div>
                <span className="text-xs text-slate-500">{b.permanent ? "permanent" : "temp"}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
