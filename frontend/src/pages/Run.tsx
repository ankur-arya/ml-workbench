import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { artifactUrl, get } from "../api";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { PromoteModal } from "../components/PromoteModal";
import { ResidualPlot } from "../components/ResidualPlot";
import { fmtMetric, prettyMetric, prettyModel, relativeTime } from "../format";
import type { ConfusionData, ResidualData, Run } from "../types";

export function RunDetail() {
  const { id } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [plot, setPlot] = useState<ConfusionData | ResidualData | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [promote, setPromote] = useState(false);

  async function load() {
    if (!id) return;
    const payload = await get<Run>(`/api/runs/${id}`);
    setRun(payload);
    const matrix = payload.artifacts.find((a) => a.name === "confusion_matrix.json");
    const residuals = payload.artifacts.find((a) => a.name === "residuals.json");
    const text = payload.artifacts.find((a) => a.name === "classification_report.txt");
    if (matrix) setPlot(await get(artifactUrl(matrix.id)));
    else if (residuals) setPlot(await get(artifactUrl(residuals.id)));
    if (text) {
      const body = await get<{ text: string }>(artifactUrl(text.id));
      setReport(body.text);
    }
  }

  useEffect(() => {
    void load();
  }, [id]);

  if (!run) return <div className="card pad">Loading run…</div>;

  const testMetrics = Object.entries(run.metrics || {}).filter(([key]) => key.startsWith("test_"));
  const trainMetrics = Object.entries(run.metrics || {}).filter(([key]) => key.startsWith("train_"));
  const plots = run.artifacts.filter((a) => a.kind === "plot");

  return (
    <>
      <div className="page-head">
        <div>
          <p className="tiny">
            <Link to={`/experiments/${run.experiment_id}`}>← {run.experiment?.name || "Experiment"}</Link>
          </p>
          <h1>{prettyModel(run.model_name)}</h1>
          <p className="lede">{run.name}</p>
          <div className="meta" style={{ marginTop: 8 }}>
            <span className={`status ${run.status}`}>{run.status}</span>
            <span>{run.dataset?.name}</span>
            <span>{run.task}</span>
            <span>{relativeTime(run.finished_at || run.created_at)}</span>
          </div>
        </div>
        <div className="actions">
          <button className="btn gold" onClick={() => setPromote(true)}>
            Promote
          </button>
        </div>
      </div>

      <div className="grid-2">
        <div className="card pad">
          <h2>Metrics</h2>
          <table className="board">
            <tbody>
              {testMetrics.map(([key, value]) => (
                <tr key={key}>
                  <td>{prettyMetric(key)}</td>
                  <td className="mono">{fmtMetric(value)}</td>
                </tr>
              ))}
              {trainMetrics.map(([key, value]) => (
                <tr key={key}>
                  <td className="muted">{prettyMetric(key)}</td>
                  <td className="mono muted">{fmtMetric(value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card pad">
          <h2>Lineage</h2>
          <dl className="kv" style={{ marginTop: 10 }}>
            <dt>Dataset</dt>
            <dd>
              <Link to="/datasets">{run.dataset?.slug}</Link> · {run.dataset?.n_rows} × {run.dataset?.n_features}
            </dd>
            <dt>Target</dt>
            <dd>{run.dataset?.target}</dd>
            <dt>Estimator</dt>
            <dd>{String(run.params.estimator || run.model_name)}</dd>
            <dt>Split</dt>
            <dd>
              {String(run.params.n_train)} / {String(run.params.n_test)} · test_size={String(run.params.test_size)}
            </dd>
            <dt>Seed</dt>
            <dd>{String(run.params.random_state)}</dd>
            <dt>Registry</dt>
            <dd>
              {run.registry?.length
                ? run.registry.map((v) => `${v.model_name || "model"} v${v.version} (${v.stage})`).join(", ")
                : "not registered"}
            </dd>
          </dl>
        </div>
      </div>

      <div className="card pad" style={{ marginTop: 14 }}>
        <h2>Hyperparameters</h2>
        <dl className="kv" style={{ marginTop: 10 }}>
          {Object.entries(run.params)
            .filter(([key]) => key.startsWith("model__"))
            .map(([key, value]) => (
              <div key={key} style={{ display: "contents" }}>
                <dt>{key.replace("model__", "")}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
        </dl>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card pad">
          <h2>Diagnostic</h2>
          <div style={{ marginTop: 12 }}>
            {plot && "matrix" in plot && <ConfusionMatrix data={plot} />}
            {plot && "residuals" in plot && <ResidualPlot data={plot} />}
            {!plot && <p className="muted">No structured plot for this run.</p>}
          </div>
        </div>
        <div className="card pad">
          <h2>Artifacts</h2>
          <ul style={{ paddingLeft: 18, marginTop: 10 }}>
            {run.artifacts.map((art) => (
              <li key={art.id}>
                <a href={artifactUrl(art.id)} target="_blank" rel="noreferrer">
                  {art.label || art.name}
                </a>
                <span className="tiny"> · {art.kind}</span>
              </li>
            ))}
          </ul>
          {plots.map((art) => (
            <img
              key={art.id}
              src={artifactUrl(art.id)}
              alt={art.label || art.name}
              style={{ width: "100%", borderRadius: 12, marginTop: 12 }}
            />
          ))}
          {report && (
            <pre className="tiny" style={{ whiteSpace: "pre-wrap", marginTop: 12 }}>
              {report}
            </pre>
          )}
        </div>
      </div>

      {promote && (
        <PromoteModal
          runId={run.id}
          onClose={() => setPromote(false)}
          onDone={() => {
            void load();
          }}
        />
      )}
    </>
  );
}
