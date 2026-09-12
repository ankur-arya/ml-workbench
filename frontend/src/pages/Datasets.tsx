import { useState } from "react";
import { post } from "../api";
import type { Bootstrap, Dataset } from "../types";

export function Datasets({ boot, onRefresh }: { boot: Bootstrap | null; onRefresh: () => void }) {
  const items = boot?.catalog.datasets ?? [];
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [target, setTarget] = useState("target");
  const [task, setTask] = useState("classification");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function register() {
    setBusy(true);
    setError(null);
    try {
      await post<Dataset>("/api/datasets", { path, name, target, task });
      setPath("");
      setName("");
      onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not register dataset");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Datasets</h1>
          <p className="lede">
            Built-in sklearn tables are ready offline. Register a local CSV when you want your own data
            in an experiment.
          </p>
        </div>
      </div>
      <div className="grid-cards">
        {items.map((ds) => (
          <div className="card pad" key={ds.id}>
            <div className="tiny">
              {ds.task} · {ds.source}
            </div>
            <h3 style={{ margin: "6px 0" }}>{ds.name}</h3>
            <p className="muted">{ds.description}</p>
            <p className="tiny" style={{ marginTop: 10 }}>
              {ds.n_rows} rows · {ds.n_features} features · used in {ds.used_in_runs ?? 0} runs
            </p>
          </div>
        ))}
      </div>
      <div className="card pad" style={{ marginTop: 18 }}>
        <h2>Register a local CSV</h2>
        <div className="grid-2" style={{ marginTop: 12 }}>
          <div className="field">
            <label>File path</label>
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/data/customers.csv" />
          </div>
          <div className="field">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="customers" />
          </div>
          <div className="field">
            <label>Target column</label>
            <input value={target} onChange={(e) => setTarget(e.target.value)} />
          </div>
          <div className="field">
            <label>Task</label>
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="classification">classification</option>
              <option value="regression">regression</option>
            </select>
          </div>
        </div>
        {error && <p className="tiny" style={{ color: "var(--bad)", marginTop: 8 }}>{error}</p>}
        <button className="btn primary" style={{ marginTop: 12 }} disabled={busy || !path || !name} onClick={() => void register()}>
          Register dataset
        </button>
      </div>
    </>
  );
}
