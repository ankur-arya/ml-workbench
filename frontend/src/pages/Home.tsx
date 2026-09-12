import { Link, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { get, post } from "../api";
import { fmtMetric, prettyModel, relativeTime } from "../format";
import type { Bootstrap, Experiment } from "../types";

export function Home({ boot, onRefresh }: { boot: Bootstrap | null; onRefresh: () => void }) {
  const [items, setItems] = useState<Experiment[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function load() {
    setItems(await get<Experiment[]>("/api/experiments"));
  }

  useEffect(() => {
    void load().catch((err) => setError(err.message));
  }, []);

  async function demo() {
    setBusy(true);
    setError(null);
    try {
      const created = await post<Experiment>("/api/demo");
      onRefresh();
      navigate(`/experiments/${created.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Demo failed");
    } finally {
      setBusy(false);
    }
  }

  if (!items.length) {
    return (
      <div className="card hero-empty">
        <p className="tiny">Local-first classical ML</p>
        <h1>Evaluate. Compare. Promote.</h1>
        <p className="lede">
          A workbench for sklearn models — not a raw run dump. Train a few candidates, read a
          scorecard, and ship a winner with one intentional action. Everything stays on this machine.
        </p>
        <div className="actions">
          <button className="btn gold" disabled={busy} onClick={() => void demo()}>
            {busy ? "Starting demo…" : "Run demo experiment"}
          </button>
          <Link className="btn primary" to="/new">
            New experiment
          </Link>
        </div>
        {error && <p className="tiny" style={{ color: "var(--bad)", marginTop: 12 }}>{error}</p>}
      </div>
    );
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Experiments</h1>
          <p className="lede">Scorecards, not a task list. Open any experiment to see the leaderboard first.</p>
        </div>
        <div className="actions">
          <button className="btn" disabled={busy} onClick={() => void demo()}>
            {busy ? "Starting…" : "Run demo"}
          </button>
          <Link className="btn primary" to="/new">
            New experiment
          </Link>
        </div>
      </div>
      {error && <div className="toast">{error}</div>}
      <div className="grid-cards">
        {items.map((item) => (
          <Link key={item.id} to={`/experiments/${item.id}`} className="card exp-card">
            <div className="top">
              <h3>{item.name}</h3>
              <span className={`status ${item.status}`}>{item.status}</span>
            </div>
            <div>
              <div className="score">
                {fmtMetric(item.best_value)}
                <small>{item.best_metric?.replace(/_/g, " ") || "primary metric"}</small>
              </div>
            </div>
            <div className="meta">
              <span>{item.candidate_count ?? 0} candidates</span>
              <span>{(item.datasets || []).join(", ") || "no datasets yet"}</span>
              {item.winner_name && <span>winner {prettyModel(item.winner_name)}</span>}
              <span>{relativeTime(item.updated_at)}</span>
            </div>
          </Link>
        ))}
      </div>
      {boot && (
        <p className="tiny" style={{ marginTop: 18 }}>
          Store {boot.home} · {boot.stats.runs} runs · {boot.stats.production_models} in production
        </p>
      )}
    </>
  );
}
