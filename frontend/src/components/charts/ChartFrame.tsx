import type { ReactNode } from "react";
import { INK } from "../../lib/viz";

/**
 * Shared chart chrome: title, one-line reading instruction, and a table view.
 *
 * The table is not decoration — it is how a chart stays readable under CVD,
 * print and forced-colors, and it is the honest answer when someone asks what a
 * mark's exact value is.
 */
export function ChartFrame({
  title,
  subtitle,
  legend,
  footnote,
  table,
  children,
  height = 260,
}: {
  title: string;
  subtitle?: string;
  legend?: ReactNode;
  footnote?: ReactNode;
  table?: ReactNode;
  children: ReactNode;
  height?: number;
}) {
  return (
    <figure className="m-0">
      <figcaption className="mb-3">
        <h3 className="text-sm font-medium">{title}</h3>
        {subtitle && <p className="text-muted mt-0.5 text-xs leading-relaxed">{subtitle}</p>}
      </figcaption>
      {legend && <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">{legend}</div>}
      <div style={{ height }}>{children}</div>
      {footnote && <p className="text-muted mt-2 text-[11px] leading-relaxed">{footnote}</p>}
      {table && (
        <details className="mt-2">
          <summary className="text-muted cursor-pointer text-[11px] hover:text-white">
            Show the numbers
          </summary>
          <div className="mt-2 max-h-56 overflow-auto">{table}</div>
        </details>
      )}
    </figure>
  );
}

/** A legend entry. The swatch carries identity; the text stays in ink tokens. */
export function LegendKey({
  colour,
  label,
  dashed = false,
}: {
  colour: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <span className="text-muted inline-flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-block h-0.5 w-4 shrink-0"
        style={
          dashed
            ? { backgroundImage: `repeating-linear-gradient(90deg, ${colour} 0 4px, transparent 4px 8px)` }
            : { background: colour }
        }
      />
      {label}
    </span>
  );
}

/** Tooltip shell. Rows are supplied by the caller so units stay correct per chart. */
export function TooltipCard({ rows, heading }: { heading?: string; rows: [string, string][] }) {
  return (
    <div
      className="rounded border px-2.5 py-2 text-[11px] shadow-lg"
      style={{ background: INK.surface, borderColor: INK.axis, color: INK.text }}
    >
      {heading && <p className="mb-1 font-medium">{heading}</p>}
      <dl className="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt style={{ color: INK.muted }}>{k}</dt>
            <dd className="text-right tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
