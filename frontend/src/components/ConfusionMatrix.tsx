import type { ConfusionData } from "../types";

export function ConfusionMatrix({ data }: { data: ConfusionData }) {
  const max = Math.max(1, ...data.matrix.flat());
  return (
    <div>
      <div className="tiny" style={{ marginBottom: 8 }}>
        Predicted →
      </div>
      <div
        className="matrix"
        style={{ gridTemplateColumns: `44px repeat(${data.labels.length}, 44px)` }}
      >
        <div />
        {data.labels.map((label) => (
          <div className="lab" key={`h-${label}`}>
            {label}
          </div>
        ))}
        {data.matrix.map((row, i) => (
          <Row
            key={`row-${data.labels[i] ?? i}`}
            label={data.labels[i]}
            values={row}
            max={max}
            rowIndex={i}
          />
        ))}
      </div>
    </div>
  );
}

function Row({
  label,
  values,
  max,
  rowIndex,
}: {
  label: string;
  values: number[];
  max: number;
  rowIndex: number;
}) {
  return (
    <>
      <div className="lab" style={{ textAlign: "right", paddingRight: 4 }}>
        {label}
      </div>
      {values.map((value, j) => {
        const intensity = value / max;
        return (
          <div
            className="cell"
            key={`${rowIndex}-${j}`}
            style={{
              background: `color-mix(in srgb, var(--gold) ${Math.round(intensity * 80)}%, var(--bg-3))`,
              color: intensity > 0.55 ? "#1a1406" : "var(--text)",
            }}
          >
            {value}
          </div>
        );
      })}
    </>
  );
}
