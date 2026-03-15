import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import Navbar              from "./components/layout/Navbar";
import HomePage            from "./pages/HomePage";
import InspectorPage       from "./pages/InspectorPage";
import LoginPage           from "./pages/LoginPage";
import AdminDashboardPage  from "./pages/AdminDashboardPage";

// Layout wraps all routes that should show the Navbar.
// LoginPage and AdminDashboardPage get their own full-screen layouts.
function AppLayout() {
  return (
    <div className="min-h-screen bg-slate-950">
      <Navbar />
      <Outlet />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Routes with Navbar */}
        <Route element={<AppLayout />}>
          <Route path="/"                   element={<HomePage />} />
          <Route path="/aim-ips-inspector"  element={<InspectorPage />} />
        </Route>

        {/* Standalone pages (own layout) */}
        <Route path="/login"           element={<LoginPage />} />
        <Route path="/admin-dashboard" element={<AdminDashboardPage />} />

        {/* Catch-all redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
