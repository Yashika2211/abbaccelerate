/** Formatting helpers. Indian numbering, because a plant manager reads lakhs. */

const LAKH = 100_000;
const CRORE = 10_000_000;

/** Money in the unit a reader actually parses at a glance. */
export function money(amount: number | null | undefined, currency = "INR"): string {
  if (amount === null || amount === undefined || Number.isNaN(amount)) return "—";
  const symbol = currency === "INR" ? "₹" : "";
  const sign = amount < 0 ? "-" : "";
  const value = Math.abs(amount);
  if (currency !== "INR") return `${sign}${symbol}${value.toLocaleString()}`;
  if (value >= CRORE) return `${sign}${symbol}${(value / CRORE).toFixed(2)} Cr`;
  if (value >= LAKH) return `${sign}${symbol}${(value / LAKH).toFixed(2)} L`;
  return `${sign}${symbol}${Math.round(value).toLocaleString("en-IN")}`;
}

/** Full precision, for tooltips and anywhere a reader is checking the arithmetic. */
export function moneyExact(amount: number | null | undefined, currency = "INR"): string {
  if (amount === null || amount === undefined || Number.isNaN(amount)) return "—";
  const symbol = currency === "INR" ? "₹" : "";
  return `${symbol}${Math.round(amount).toLocaleString("en-IN")}`;
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

export function titleCase(text: string): string {
  return text.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
