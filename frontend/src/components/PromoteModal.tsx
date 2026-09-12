import { useState } from "react";
import { post } from "../api";

const MIN_NOTE = 4;

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
        {success ? (
          <>
            <h2>Promoted to {stage}</h2>
            <p className="lede" style={{ color: "var(--ok)" }}>
              This model is now in {stage}. The registry and experiment views have been refreshed.
            </p>
            <p className="tiny" style={{ marginTop: 10 }}>
              {note.trim()}
            </p>
            <div className="toast ok" role="status">
              Promoted to {stage}.
            </div>
            <div className="actions" style={{ marginTop: 16 }}>
              <button className="btn gold" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : (
          <>
            <h2>Promote to {stage}</h2>
            <p className="lede">This is an intentional production action. Write why this model wins.</p>
            <div className="field" style={{ marginTop: 16 }}>
              <label>Stage</label>
              <select value={stage} onChange={(e) => setStage(e.target.value)} disabled={busy}>
                <option value="staging">staging (challenger)</option>
                <option value="production">production (champion)</option>
                <option value="archived">archived</option>
                <option value="candidate">candidate</option>
              </select>
            </div>
            <div className="field" style={{ marginTop: 12 }}>
              <label>Who</label>
              <input value={actor} onChange={(e) => setActor(e.target.value)} disabled={busy} />
            </div>
            <div className="field" style={{ marginTop: 12 }}>
              <label htmlFor="promote-note">Note (required, at least {MIN_NOTE} characters)</label>
              <textarea
                id="promote-note"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Best holdout accuracy on iris; confusion matrix is clean."
                disabled={busy}
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
            <div className="actions" style={{ marginTop: 16 }}>
              <button className="btn gold" disabled={busy || !noteReady} onClick={() => void submit()}>
                {busy ? "Promoting…" : "Confirm promote"}
              </button>
              <button className="btn ghost" onClick={onClose} disabled={busy}>
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
