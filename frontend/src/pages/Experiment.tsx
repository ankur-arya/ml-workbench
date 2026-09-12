import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { get } from "../api";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { MetricBar } from "../components/MetricBar";
import { PromoteModal } from "../components/PromoteModal";
import { ResidualPlot } from "../components/ResidualPlot";
import { fmtDelta, fmtMetric, prettyMetric, prettyModel, relativeTime } from "../format";
import type { ComparePayload, ConfusionData, Experiment as ExperimentT, ResidualData, Run } from "../types";

export function Experiment() {
  const { id } = useParams();
  const [data, setData] = useState<ExperimentT | null>(null);
  const [facet, setFacet] = useState<string>("all");
  const [selected, setSelected] = useState<string[]>([]);
  const [promote, setPromote] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    if (!id) return;
    const payload = await get<ExperimentT>(`/api/experiments/${id}`);
    setData(payload);
  }

  useEffect(() => {
    void load().catch((err) => setError(err.message));
  }, [id]);

  useEffect(() => {
    if (!data || data.status !== "running") return;
    const timer = window.setInterval(() => {
      void load();
    }, 900);
    return () => window.clearInterval(timer);
  }, [data?.status, id]);

  const compare: ComparePayload | undefined = data?.compare;
  const board = useMemo(() => {
    if (!compare) return [];
    if (facet === "all") return compare.leaderboard;
    return compare.by_dataset[facet] || [];
  }, [compare, facet]);

  const testMetrics = (compare?.metric_keys || []).filter((key) => key.startsWith("test_")).slice(0, 4);
  const maxByMetric: Record<string, number> = {};
  for (const key of testMetrics) {
    maxByMetric[key] = Math.max(...board.map((run) => Math.abs(run.metrics?.[key] ?? 0)), 1e-6);
  }

  function toggle(runId: string) {
    setSelected((cur) => {
      if (cur.includes(runId)) return cur.filter((x) => x !== runId);
      if (cur.length >= 2) return [cur[1], runId];
      return [...cur, runId];
    });
  }

  if (error) return <div className="card pad">{error}</div>;
  if (!data) return <div className="card pad">Loading scorecard…</div>;

  const winner = facet === "all" ? compare?.recommended : board.find((row) => row.is_recommended);
  const progress = data.progress || { done: 0, total: 0, current: null };

  return (
    <>
      <div className="page-head">
        <div>
          <p className="tiny">
            <Link to="/">Experiments</Link> / {data.id}
          </p>
          <h1>{data.name}</h1>
          <p className="lede">{data.description || "Leaderboard for this experiment."}</p>
          <div className="meta" style={{ marginTop: 8 }}>
            <span className={`status ${data.status}`}>{data.status}</span>
            <span>{(data.datasets || []).join(" · ")}</span>
            <span>{prettyMetric(compare?.primary_metric || data.primary_metric || "")}</span>
            <span>{relativeTime(data.updated_at)}</span>
          </div>
        </div>
        <div className="actions">
          <button className="btn gold" disabled={!winner} onClick={() => setPromote(true)}>
            Promote winner
          </button>
        </div>
      </div>

      {data.status === "running" && (
        <div className="card pad">
          <strong>Training locally</strong>
          <p className="tiny">
            {progress.done}/{progress.total} · {progress.current || "starting…"}
          </p>
          <div className="progress">
            <span style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 8}%` }} />
          </div>
        </div>
      )}

      {winner && (
        <div className="card winner">
          <div>
            <div className="kicker">Recommended winner</div>
            <h2>{prettyModel(winner.model_name)}</h2>
            <p className="muted">
              {winner.name} · {(winner.dataset && winner.dataset.slug) || "dataset"} · {compare?.reason}
            </p>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="score">{fmtMetric(winner.primary_value)}</div>
            <div className="tiny">{prettyMetric(compare?.primary_metric || "")}</div>
            <button className="btn gold" style={{ marginTop: 10 }} onClick={() => setPromote(true)}>
              Promote to production
            </button>
          </div>
        </div>
      )}

      {(compare?.datasets.length || 0) > 1 && (
        <div className="tabs">
          <button className={`tab ${facet === "all" ? "on" : ""}`} onClick={() => setFacet("all")}>
            All datasets
          </button>
          {compare?.datasets.map((slug) => (
            <button key={slug} className={`tab ${facet === slug ? "on" : ""}`} onClick={() => setFacet(slug)}>
              {slug}
            </button>
          ))}
        </div>
      )}

      <div className="card">
        <div className="table-wrap">
          <table className="board">
            <thead>
              <tr>
                <th></th>
                <th>#</th>
                <th>Candidate</th>
                <th>Dataset</th>
                {testMetrics.map((key) => (
                  <th key={key}>{prettyMetric(key)}</th>
                ))}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {board.map((run) => (
                <tr
                  key={run.id}
                  className={`${run.is_recommended ? "winner-row" : ""} ${selected.includes(run.id) ? "selected" : ""}`}
                >
                  <td>
                    <input type="checkbox" checked={selected.includes(run.id)} onChange={() => toggle(run.id)} />
                  </td>
                  <td>
                    <span className={`rank ${run.rank === 1 ? "gold" : ""}`}>{run.rank ?? "–"}</span>
                  </td>
                  <td>
                    <Link to={`/runs/${run.id}`}>
                      <strong>{prettyModel(run.model_name)}</strong>
                    </Link>
                    <div className="tiny">
                      {run.status}
                      {run.delta_vs_best != null && run.rank !== 1 ? ` · ${fmtDelta(run.delta_vs_best)} vs best` : ""}
                      {run.constraint_failed ? " · failed constraint" : ""}
                    </div>
                  </td>
                  <td>{run.dataset?.slug}</td>
                  {testMetrics.map((key) => (
                    <td key={key}>
                      <MetricBar
                        value={run.metrics?.[key]}
                        max={maxByMetric[key]}
                        gold={key === compare?.primary_metric && run.is_recommended}
                      />
                    </td>
                  ))}
                  <td>
                    <Link className="tiny" to={`/runs/${run.id}`}>
                      Detail
                    </Link>
                  </td>
                </tr>
              ))}
              {!board.length && (
                <tr>
                  <td colSpan={6} className="muted">
                    Waiting for the first succeeded run…
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selected.length === 2 && <SideBySide leftId={selected[0]} rightId={selected[1]} metrics={testMetrics} />}

      {promote && winner && (
        <PromoteModal
          runId={winner.id}
          onClose={() => setPromote(false)}
          onDone={() => {
            void load();
          }}
        />
      )}
    </>
  );
}

function SideBySide({ leftId, rightId, metrics }: { leftId: string; rightId: string; metrics: string[] }) {
  const [left, setLeft] = useState<Run | null>(null);
  const [right, setRight] = useState<Run | null>(null);
  const [leftPlot, setLeftPlot] = useState<ConfusionData | ResidualData | null>(null);
  const [rightPlot, setRightPlot] = useState<ConfusionData | ResidualData | null>(null);

  useEffect(() => {
    void (async () => {
      const [a, b] = await Promise.all([get<Run>(`/api/runs/${leftId}`), get<Run>(`/api/runs/${rightId}`)]);
      setLeft(a);
      setRight(b);
      setLeftPlot(await loadPlot(a));
      setRightPlot(await loadPlot(b));
    })();
  }, [leftId, rightId]);

  if (!left || !right) return <div className="card pad">Loading comparison…</div>;

  const INTERESTING = new Set([
    "n_estimators",
    "max_depth",
    "learning_rate",
    "C",
    "alpha",
    "max_iter",
    "kernel",
    "solver",
    "min_samples_leaf",
    "min_samples_split",
  ]);
  const paramKeys = Array.from(
    new Set(
      [...Object.keys(left.params || {}), ...Object.keys(right.params || {})]
        .filter((key) => key.startsWith("model__"))
        .filter((key) => {
          const short = key.replace("model__", "");
          const a = String(left.params?.[key] ?? "—");
          const b = String(right.params?.[key] ?? "—");
          return INTERESTING.has(short) || a !== b;
        }),
    ),
  ).slice(0, 12);

  return (
    <div style={{ marginTop: 16 }}>
      <h2 style={{ marginBottom: 10 }}>Side-by-side</h2>
      <div className="compare-grid">
        <ComparePane run={left} plot={leftPlot} other={right} metrics={metrics} paramKeys={paramKeys} />
        <ComparePane run={right} plot={rightPlot} other={left} metrics={metrics} paramKeys={paramKeys} />
      </div>
    </div>
  );
}

function ComparePane({
  run,
  plot,
  other,
  metrics,
  paramKeys,
}: {
  run: Run;
  plot: ConfusionData | ResidualData | null;
  other: Run;
  metrics: string[];
  paramKeys: string[];
}) {
  return (
    <div className="card pad">
      <div className="tiny">{run.dataset?.slug}</div>
      <h3>
        <Link to={`/runs/${run.id}`}>{prettyModel(run.model_name)}</Link>
      </h3>
      <table className="board">
        <tbody>
          {metrics.map((key) => {
            const value = run.metrics?.[key];
            const vs = other.metrics?.[key];
            const cls = value == null || vs == null ? "diff-same" : value > vs ? "diff-better" : value < vs ? "diff-worse" : "diff-same";
            return (
              <tr key={key}>
                <td>{prettyMetric(key)}</td>
                <td className={`mono ${cls}`}>{fmtMetric(value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <h3 style={{ margin: "16px 0 8px", fontSize: 18 }}>Params</h3>
      <dl className="kv">
        {paramKeys.map((key) => {
          const a = String(run.params?.[key] ?? "—");
          const b = String(other.params?.[key] ?? "—");
          return (
            <div key={key} style={{ display: "contents" }}>
              <dt>{key.replace("model__", "")}</dt>
              <dd className={a === b ? "diff-same" : "diff-better"}>{a}</dd>
            </div>
          );
        })}
      </dl>
      <div style={{ marginTop: 16 }}>
        {plot && "matrix" in plot && <ConfusionMatrix data={plot} />}
        {plot && "residuals" in plot && <ResidualPlot data={plot} />}
      </div>
    </div>
  );
}

async function loadPlot(run: Run): Promise<ConfusionData | ResidualData | null> {
  const artifact =
    run.artifacts.find((item) => item.name === "confusion_matrix.json") ||
    run.artifacts.find((item) => item.name === "residuals.json");
  if (!artifact) return null;
  return get(artifactUrlSafe(artifact.id));
}

function artifactUrlSafe(id: string) {
  return `/api/artifacts/${id}`;
}
