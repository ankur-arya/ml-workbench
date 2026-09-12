import { useState } from "react";
import { post } from "../api";

const MIN_NOTE = 4;
const SUCCESS_PAUSE_MS = 1100;

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
  const [success, setSuccess] = useState(false);

  const trimmedLen = note.trim().length;
  const noteReady = trimmedLen >= MIN_NOTE;

  function dismiss() {
    if (success) onDone();
    else onClose();
  }

  async function submit() {
    if (!noteReady || busy || success) return;
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
      setSuccess(true);
      window.setTimeout(() => onDone(), SUCCESS_PAUSE_MS);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Promote failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-back" onClick={dismiss}>
      <div className="card pad modal" onClick={(event) => event.stopPropagation()}>
        <h2>Promote to {stage}</h2>
        <p className="lede">This is an intentional production action. Write why this model wins.</p>
        <div className="field" style={{ marginTop: 16 }}>
          <label>Stage</label>
          <select value={stage} onChange={(e) => setStage(e.target.value)} disabled={busy || success}>
            <option value="staging">staging (challenger)</option>
            <option value="production">production (champion)</option>
            <option value="archived">archived</option>
            <option value="candidate">candidate</option>
          </select>
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <label>Who</label>
          <input value={actor} onChange={(e) => setActor(e.target.value)} disabled={busy || success} />
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <label htmlFor="promote-note">Note (required, at least {MIN_NOTE} characters)</label>
          <textarea
            id="promote-note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Best holdout accuracy on iris; confusion matrix is clean."
            disabled={busy || success}
            aria-describedby="promote-note-hint"
          />
          <p id="promote-note-hint" className="tiny" style={{ color: noteReady ? "var(--ok)" : "var(--warn)" }}>
            {noteReady
              ? "Note looks good — Confirm promote is enabled."
              : `Confirm stays disabled until the note has at least ${MIN_NOTE} characters (${trimmedLen}/${MIN_NOTE}).`}
          </p>
        </div>
        {error && (
          <p className="tiny" style={{ color: "var(--bad)", marginTop: 8 }} role="alert">
            {error}
          </p>
        )}
        {success && (
          <p className="tiny" style={{ color: "var(--ok)", marginTop: 8 }} role="status">
            Promoted to {stage}. Refreshing the registry…
          </p>
        )}
        {success && <div className="toast ok">Promoted to {stage}.</div>}
        <div className="actions" style={{ marginTop: 16 }}>
          <button className="btn gold" disabled={busy || success || !noteReady} onClick={() => void submit()}>
            {success ? "Promoted" : busy ? "Promoting…" : "Confirm promote"}
          </button>
          <button className="btn ghost" onClick={dismiss} disabled={busy}>
            {success ? "Close" : "Cancel"}
          </button>
        </div>
      </div>
    </div>
  );
}
