import { useState } from "react";
import { post } from "../api";

export function PromoteModal({
  runId,
  experimentId,
  versionId,
  defaultStage = "production",
  onClose,
  onDone,
}: {
  runId?: string;
  experimentId?: string;
  versionId?: string;
  defaultStage?: string;
  onClose: () => void;
  onDone: () => void;
}) {
  const [note, setNote] = useState("");
  const [stage, setStage] = useState(defaultStage);
  const [actor, setActor] = useState("local");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      await post("/api/promote", {
        run_id: runId,
        experiment_id: experimentId,
        version_id: versionId,
        stage,
        note,
        actor,
      });
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Promote failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="card pad modal" onClick={(event) => event.stopPropagation()}>
        <h2>Promote to {stage}</h2>
        <p className="lede">This is an intentional production action. Write why this model wins.</p>
        <div className="field" style={{ marginTop: 16 }}>
          <label>Stage</label>
          <select value={stage} onChange={(e) => setStage(e.target.value)}>
            <option value="staging">staging (challenger)</option>
            <option value="production">production (champion)</option>
            <option value="archived">archived</option>
            <option value="candidate">candidate</option>
          </select>
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <label>Who</label>
          <input value={actor} onChange={(e) => setActor(e.target.value)} />
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <label>Note (required)</label>
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Best holdout accuracy on iris; confusion matrix is clean."
          />
        </div>
        {error && <p className="tiny" style={{ color: "var(--bad)", marginTop: 8 }}>{error}</p>}
        <div className="actions" style={{ marginTop: 16 }}>
          <button className="btn gold" disabled={busy || note.trim().length < 4} onClick={() => void submit()}>
            {busy ? "Promoting…" : "Confirm promote"}
          </button>
          <button className="btn ghost" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
