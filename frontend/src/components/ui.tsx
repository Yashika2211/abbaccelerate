/** Small shared primitives. Deliberately plain — the UI gets polish after hour 15. */

import type { ReactNode } from "react";

export function Card({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`border-edge bg-panel rounded-lg border ${className}`}>
      {(title || right) && (
        <header className="border-edge flex items-start justify-between gap-4 border-b px-5 py-3">
          <div>
            {title && <h2 className="text-sm font-medium tracking-wide">{title}</h2>}
            {subtitle && <p className="text-muted mt-0.5 text-xs">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="px-5 py-4">{children}</div>
    </section>
  );
}

/**
 * Shown wherever there is no data. PROJECT_BRIEF.md §11 forbids placeholder
 * numbers, so an absent value renders as an explanation of why it is absent.
 */
export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="border-edge rounded border border-dashed px-4 py-8 text-center">
      <p className="text-sm">{title}</p>
      {detail && <p className="text-muted mx-auto mt-1 max-w-md text-xs">{detail}</p>}
    </div>
  );
}

export function Dot({ up, label }: { up: boolean; label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={`inline-block h-2 w-2 shrink-0 rounded-full ${up ? "bg-signal" : "bg-alarm"}`}
        aria-label={up ? "up" : "down"}
      />
      {label && <span className="text-xs">{label}</span>}
    </span>
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
  children: ReactNode;
}) {
  const tones = {
    neutral: "bg-edge text-muted",
    good: "bg-signal/15 text-signal",
    warn: "bg-amber-400/15 text-amber-300",
    bad: "bg-alarm/15 text-alarm",
    accent: "bg-accent/15 text-accent",
  };
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  const styles = {
    primary: "bg-accent text-ink hover:bg-accent/85",
    ghost: "border-edge border text-muted hover:text-white hover:border-muted",
    danger: "border-alarm/40 border text-alarm hover:bg-alarm/10",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]}`}
    >
      {children}
    </button>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p className="border-alarm/30 bg-alarm/10 text-alarm mt-3 rounded border px-3 py-2 text-xs">
      {error}
    </p>
  );
}

/** A labelled statistic. `hint` carries the assumption behind an annualised figure. */
export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "good" | "bad";
}) {
  const colour = tone === "good" ? "text-signal" : tone === "bad" ? "text-alarm" : "";
  return (
    <div>
      <dt className="text-muted text-[11px] tracking-wide uppercase">{label}</dt>
      <dd className={`mt-0.5 text-lg font-semibold tabular-nums ${colour}`}>{value}</dd>
      {hint && <p className="text-muted mt-0.5 text-[11px]">{hint}</p>}
    </div>
  );
}
