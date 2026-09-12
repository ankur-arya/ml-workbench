import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { get } from "./api";
import type { Bootstrap } from "./types";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Wizard } from "./pages/Wizard";
import { Experiment } from "./pages/Experiment";
import { RunDetail } from "./pages/Run";
import { Registry } from "./pages/Registry";
import { Datasets } from "./pages/Datasets";

export default function App() {
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [error, setError] = useState<string | null>(null);
  const location = useLocation();

  async function refresh() {
    try {
      setBoot(await get<Bootstrap>("/api/bootstrap"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the API");
    }
  }

  useEffect(() => {
    void refresh();
  }, [location.pathname]);

  return (
    <Layout boot={boot} onRefresh={refresh}>
      {error && <div className="toast">{error}</div>}
      <Routes>
        <Route path="/" element={<Home boot={boot} onRefresh={refresh} />} />
        <Route path="/new" element={<Wizard boot={boot} />} />
        <Route path="/experiments/:id" element={<Experiment />} />
        <Route path="/runs/:id" element={<RunDetail />} />
        <Route path="/registry" element={<Registry />} />
        <Route path="/datasets" element={<Datasets boot={boot} onRefresh={refresh} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
