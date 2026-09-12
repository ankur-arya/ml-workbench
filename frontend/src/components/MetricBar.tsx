import { fmtMetric } from "../format";

export function MetricBar({
  value,
  max,
  gold,
}: {
  value: number | null | undefined;
  max: number;
  gold?: boolean;
}) {
  if (value === null || value === undefined) return <span className="faint">—</span>;
  const width = Math.max(4, Math.min(100, (Math.abs(value) / (max || 1)) * 100));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div className={`bar ${gold ? "gold" : ""}`} style={{ flex: 1 }}>
        <span style={{ width: `${width}%` }} />
      </div>
      <span className="mono">{fmtMetric(value)}</span>
    </div>
  );
}
