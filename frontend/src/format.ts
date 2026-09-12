export function fmtMetric(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  return value.toFixed(digits);
}

export function fmtDelta(value: number | null | undefined): string {
  if (value === null || value === undefined) return "";
  if (Math.abs(value) < 1e-9) return "tied";
  const sign = value > 0 ? "+" : "";
  return `${sign}${fmtMetric(value)}`;
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.round((Date.now() - then) / 1000);
  const abs = Math.abs(seconds);
  const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (abs < 60) return rtf.format(-seconds, "second");
  if (abs < 3600) return rtf.format(-Math.round(seconds / 60), "minute");
  if (abs < 86400) return rtf.format(-Math.round(seconds / 3600), "hour");
  return rtf.format(-Math.round(seconds / 86400), "day");
}

export function prettyModel(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

export function prettyMetric(key: string): string {
  return key.replace(/^test_/, "").replace(/^train_/, "train ").replace(/_/g, " ");
}

export function statusLabel(status: string): string {
  return status;
}
