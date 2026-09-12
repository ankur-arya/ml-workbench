import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { get } from "../api";
import { PromoteModal } from "../components/PromoteModal";
import { relativeTime } from "../format";
import type { RegisteredModel } from "../types";

export function Registry() {
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [promote, setPromote] = useState<{ versionId: string; stage: string } | null>(null);

  async function load() {
    setModels(await get<RegisteredModel[]>("/api/registry"));
  }

  useEffect(() => {
    void load();
  }, []);

  if (!models.length) {
    return (
      <div className="card hero-empty">
        <h1>Nothing in production yet.</h1>
        <p className="lede">
          Train an experiment, pick the winner from the leaderboard, and promote it with a note. The
          champion lives here — linked back to the run that earned it.
        </p>
        <Link className="btn gold" to="/">
          Go to experiments
        </Link>
      </div>
    );
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model registry</h1>
          <p className="lede">Stages: candidate → staging → production. Production is exclusive per model name.</p>
        </div>
      </div>
      {models.map((model) => (
        <div className="card pad" key={model.id} style={{ marginBottom: 14 }}>
          <div className="top" style={{ display: "flex", justifyContent: "space-between" }}>
            <div>
              <h2>{model.name}</h2>
              <p className="tiny">
                {model.alias ? `${model.alias} · ` : ""}
                updated {relativeTime(model.updated_at)}
              </p>
            </div>
            {model.production && <span className="status production">production</span>}
          </div>
          <table className="board" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Ver</th>
                <th>Stage</th>
                <th>Source run</th>
                <th>Why</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {model.versions.map((version) => (
                <tr key={version.id}>
                  <td className="mono">v{version.version}</td>
                  <td>
                    <span className={`status ${version.stage}`}>{version.stage}</span>
                  </td>
                  <td>
                    <Link to={`/runs/${version.run_id}`}>{version.run_name || version.run_id}</Link>
                  </td>
                  <td className="tiny">
                    {version.promotions?.[0]
                      ? `${version.promotions[0].actor}: ${version.promotions[0].note}`
                      : "registered as candidate"}
                  </td>
                  <td>
                    <button className="btn ghost" onClick={() => setPromote({ versionId: version.id, stage: "production" })}>
                      Promote
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      {promote && (
        <PromoteModal
          versionId={promote.versionId}
          defaultStage={promote.stage}
          onClose={() => setPromote(null)}
          onDone={() => {
            setPromote(null);
            void load();
          }}
        />
      )}
    </>
  );
}
