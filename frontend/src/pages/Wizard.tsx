import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { post } from "../api";
import type { Bootstrap, Experiment } from "../types";

const STEPS = ["Setup", "Datasets", "Models", "Review"];

export function Wizard({ boot }: { boot: Bootstrap | null }) {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [name, setName] = useState("Iris bakeoff");
  const [description, setDescription] = useState("");
  const [primary, setPrimary] = useState("test_accuracy");
  const [maximize, setMaximize] = useState(true);
  const [testSize, setTestSize] = useState(0.2);
  const [datasets, setDatasets] = useState<string[]>(["iris"]);
  const [models, setModels] = useState<Record<string, Record<string, number | null>>>({
    logistic_regression: { max_iter: 400 },
    random_forest: { n_estimators: 80, max_depth: 4 },
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const catalogDatasets = boot?.catalog.datasets ?? [];
  const catalogModels = boot?.catalog.models ?? [];
  const selectedTasks = useMemo(
    () => new Set(catalogDatasets.filter((d) => datasets.includes(d.slug)).map((d) => d.task)),
    [catalogDatasets, datasets],
  );

  function toggleDataset(slug: string) {
    setDatasets((cur) => (cur.includes(slug) ? cur.filter((x) => x !== slug) : [...cur, slug]));
  }

  function toggleModel(key: string) {
    setModels((cur) => {
      const next = { ...cur };
      if (next[key]) delete next[key];
      else {
        const meta = catalogModels.find((m) => m.key === key);
        const params: Record<string, number | null> = {};
        for (const p of meta?.params ?? []) params[p.key] = p.default;
        next[key] = params;
      }
      return next;
    });
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const payload = {
        name,
        description,
        datasets,
        models: Object.entries(models).map(([modelName, params]) => ({ name: modelName, params })),
        primary_metric: primary,
        maximize,
        test_size: testSize,
        random_state: 42,
        register_model: true,
        start: true,
        background: true,
      };
      const created = await post<Experiment>("/api/experiments", payload);
      navigate(`/experiments/${created.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start experiment");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wizard">
      <div className="page-head">
        <div>
          <h1>New experiment</h1>
          <p className="lede">Pick datasets and candidates. After the run, you land on the leaderboard — not a task dump.</p>
        </div>
      </div>
      <div className="steps">
        {STEPS.map((label, i) => (
          <button key={label} className={`step ${i === step ? "on" : ""}`} onClick={() => setStep(i)}>
            <b>Step {i + 1}</b>
            {label}
          </button>
        ))}
      </div>

      {step === 0 && (
        <div className="card pad">
          <div className="grid-2">
            <div className="field">
              <label>Name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="field">
              <label>Primary metric</label>
              <select value={primary} onChange={(e) => setPrimary(e.target.value)}>
                <option value="test_accuracy">test_accuracy</option>
                <option value="test_f1_macro">test_f1_macro</option>
                <option value="test_r2">test_r2</option>
                <option value="test_rmse">test_rmse</option>
                <option value="test_mae">test_mae</option>
              </select>
            </div>
            <div className="field">
              <label>Direction</label>
              <select value={maximize ? "max" : "min"} onChange={(e) => setMaximize(e.target.value === "max")}>
                <option value="max">Higher is better</option>
                <option value="min">Lower is better</option>
              </select>
            </div>
            <div className="field">
              <label>Holdout</label>
              <input type="number" step="0.05" min="0.1" max="0.5" value={testSize} onChange={(e) => setTestSize(Number(e.target.value))} />
            </div>
          </div>
          <div className="field" style={{ marginTop: 12 }}>
            <label>Why this experiment</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Find a production iris classifier." />
          </div>
        </div>
      )}

      {step === 1 && (
        <div className="grid-cards">
          {catalogDatasets.map((ds) => (
            <button key={ds.id} className={`choice ${datasets.includes(ds.slug) ? "on" : ""}`} onClick={() => toggleDataset(ds.slug)}>
              <div className="tiny">{ds.task} · {ds.source}</div>
              <h3>{ds.name}</h3>
              <p>{ds.description}</p>
              <p className="tiny" style={{ marginTop: 8 }}>{ds.n_rows} rows · {ds.n_features} features</p>
            </button>
          ))}
        </div>
      )}

      {step === 2 && (
        <div className="grid-cards">
          {catalogModels
            .filter((model) => !selectedTasks.size || model.tasks.some((t) => selectedTasks.has(t)))
            .map((model) => {
              const on = Boolean(models[model.key]);
              return (
                <div key={model.key} className={`choice ${on ? "on" : ""}`} onClick={() => toggleModel(model.key)} role="button" tabIndex={0}>
                  <div className="tiny">{model.tasks.join(" · ")}</div>
                  <h3>{model.label}</h3>
                  <p>{model.blurb}</p>
                  {on && model.params.length > 0 && (
                    <div className="grid-2" style={{ marginTop: 12 }} onClick={(e) => e.stopPropagation()}>
                      {model.params.map((param) => (
                        <div className="field" key={param.key}>
                          <label>{param.label}</label>
                          <input
                            type="number"
                            value={models[model.key][param.key] ?? ""}
                            onChange={(e) =>
                              setModels((cur) => ({
                                ...cur,
                                [model.key]: {
                                  ...cur[model.key],
                                  [param.key]: e.target.value === "" ? null : Number(e.target.value),
                                },
                              }))
                            }
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      )}

      {step === 3 && (
        <div className="card pad">
          <h2>{name}</h2>
          <p className="lede">{description || "No description"}</p>
          <dl className="kv" style={{ marginTop: 16 }}>
            <dt>Datasets</dt>
            <dd>{datasets.join(", ") || "—"}</dd>
            <dt>Models</dt>
            <dd>{Object.keys(models).join(", ") || "—"}</dd>
            <dt>Primary</dt>
            <dd>
              {primary} · {maximize ? "maximize" : "minimize"}
            </dd>
            <dt>Holdout</dt>
            <dd>{testSize}</dd>
            <dt>Candidates</dt>
            <dd>{datasets.length * Object.keys(models).length} runs</dd>
          </dl>
        </div>
      )}

      {error && <p className="tiny" style={{ color: "var(--bad)", marginTop: 12 }}>{error}</p>}
      <div className="actions" style={{ marginTop: 18 }}>
        <button className="btn" disabled={step === 0} onClick={() => setStep((s) => s - 1)}>
          Back
        </button>
        {step < 3 ? (
          <button className="btn primary" onClick={() => setStep((s) => s + 1)}>
            Continue
          </button>
        ) : (
          <button className="btn gold" disabled={busy || !datasets.length || !Object.keys(models).length} onClick={() => void run()}>
            {busy ? "Starting…" : "Run experiment"}
          </button>
        )}
      </div>
    </div>
  );
}
