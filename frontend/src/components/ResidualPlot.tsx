import type { ResidualData } from "../types";

export function ResidualPlot({ data }: { data: ResidualData }) {
  const xs = data.y_pred;
  const ys = data.residuals;
  if (!xs.length) return <p className="muted">No residual points.</p>;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const maxAbs = Math.max(...ys.map(Math.abs), 1e-6);
  const w = 420;
  const h = 220;
  const pad = 24;
  const x = (v: number) => pad + ((v - minX) / (maxX - minX || 1)) * (w - pad * 2);
  const y = (v: number) => h / 2 - (v / maxAbs) * (h / 2 - pad);
  return (
    <svg className="scatter" viewBox={`0 0 ${w} ${h}`}>
      <line x1={pad} x2={w - pad} y1={h / 2} y2={h / 2} stroke="currentColor" opacity="0.25" />
      {xs.map((px, i) => (
        <circle key={i} cx={x(px)} cy={y(ys[i])} r="3.2" fill="#e4c36b" opacity="0.85" />
      ))}
    </svg>
  );
}
