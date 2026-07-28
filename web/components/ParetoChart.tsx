"use client";

import { motion, useReducedMotion } from "motion/react";
import { useMemo, useState } from "react";
import { MAX_CONFIGS, type DesignPoint } from "@/lib/dse";
import { compact, cyclesToTime, pct } from "@/lib/format";

const W = 640;
const H = 300;
const PAD = { top: 18, right: 18, bottom: 42, left: 54 };

export function ParetoChart({
  front,
  currentLatency,
  onPick,
  truncatedFrom,
}: {
  front: DesignPoint[];
  currentLatency: number;
  onPick: (pt: DesignPoint) => void;
  /** Total size of the space when the search was capped; null when exhaustive. */
  truncatedFrom: number | null;
}) {
  const reduced = useReducedMotion();
  const [hover, setHover] = useState<number | null>(null);

  const { pts, xTicks, yTicks } = useMemo(() => {
    if (!front.length) return { pts: [], xTicks: [], yTicks: [] };
    const lats = front.map((p) => Math.log10(Math.max(1, p.latency)));
    const utils = front.map((p) => p.utilisation);
    const xMin = Math.min(...lats);
    const xMax = Math.max(...lats, xMin + 0.3);
    const yMax = Math.max(...utils, 0.05) * 1.15;

    const sx = (v: number) =>
      PAD.left +
      ((Math.log10(Math.max(1, v)) - xMin) / (xMax - xMin || 1)) *
        (W - PAD.left - PAD.right);
    const sy = (v: number) => H - PAD.bottom - (v / yMax) * (H - PAD.top - PAD.bottom);

    return {
      pts: front.map((p, i) => ({ p, i, x: sx(p.latency), y: sy(p.utilisation) })),
      xTicks: [xMin, (xMin + xMax) / 2, xMax].map((l) => ({
        x: sx(10 ** l),
        label: compact(Math.round(10 ** l)),
      })),
      yTicks: [0, yMax / 2, yMax].map((v) => ({ y: sy(v), label: pct(v) })),
    };
  }, [front]);

  if (!front.length) {
    return (
      <p className="font-mono text-sm text-over">
        No configuration of this network fits the selected device.
      </p>
    );
  }

  const active = hover !== null ? front[hover] : null;

  return (
    <div>
      <p className="mb-4 max-w-2xl text-[13px] leading-relaxed text-muted">
        Every point is a configuration that fits the device. Nothing to the lower-left
        of a point beats it on both axes — that is what makes this a Pareto front.
        Click one to load it.
      </p>
      {truncatedFrom !== null && (
        <p className="mb-4 rounded border border-over/30 bg-over/[0.06] px-3 py-2 font-mono text-[11px] text-bone">
          Search capped: {truncatedFrom.toLocaleString("en-US")} configurations exist,
          only the first {MAX_CONFIGS.toLocaleString("en-US")} were evaluated. This
          front is not guaranteed complete.
        </p>
      )}
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full min-w-[520px]"
          role="img"
          aria-label={`Pareto front of ${front.length} configurations, latency versus peak resource utilisation`}
        >
          {yTicks.map((t, i) => (
            <g key={`y${i}`}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={t.y}
                y2={t.y}
                stroke="var(--edge)"
                strokeWidth="1"
              />
              <text
                x={PAD.left - 10}
                y={t.y + 3.5}
                textAnchor="end"
                className="fill-[var(--muted-dim)] font-mono text-[10px]"
              >
                {t.label}
              </text>
            </g>
          ))}
          {xTicks.map((t, i) => (
            <text
              key={`x${i}`}
              x={t.x}
              y={H - PAD.bottom + 16}
              textAnchor="middle"
              className="fill-[var(--muted-dim)] font-mono text-[10px]"
            >
              {t.label}
            </text>
          ))}

          <text
            x={W / 2}
            y={H - 6}
            textAnchor="middle"
            className="fill-[var(--muted-dim)] font-mono text-[10px]"
          >
            latency (cycles, log scale) →
          </text>
          <text
            x={-H / 2}
            y={13}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-[var(--muted-dim)] font-mono text-[10px]"
          >
            ← peak utilisation
          </text>

          <motion.polyline
            points={pts.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke="var(--trace)"
            strokeWidth="1"
            strokeDasharray="3 3"
            opacity={0.5}
            initial={reduced ? false : { pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          />

          {pts.map(({ p, i, x, y }) => {
            const isCurrent = p.latency === currentLatency;
            return (
              <g key={i}>
                <motion.circle
                  cx={x}
                  cy={y}
                  r={hover === i ? 6.5 : isCurrent ? 5.5 : 4}
                  fill={isCurrent ? "var(--signal)" : "var(--substrate)"}
                  stroke={isCurrent ? "var(--signal)" : "var(--trace)"}
                  strokeWidth="1.5"
                  initial={reduced ? false : { opacity: 0, scale: 0.4 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: reduced ? 0 : i * 0.03, duration: 0.25 }}
                />
                <circle
                  cx={x}
                  cy={y}
                  r={14}
                  fill="transparent"
                  className="cursor-pointer"
                  tabIndex={0}
                  role="button"
                  aria-label={`Configuration: ${p.latency} cycles, ${pct(p.utilisation)} utilisation`}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                  onClick={() => onPick(p)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onPick(p);
                    }
                  }}
                />
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 min-h-[2.5rem] font-mono text-xs">
        {active ? (
          <p className="tabular text-muted">
            <span className="text-bone">{compact(active.latency)} cycles</span>
            <span className="text-muted-dim"> ({cyclesToTime(active.latency)} @ 100 MHz)</span>
            {" · "}
            {active.dsp} DSP · {active.bram} BRAM · {compact(active.lut)} LUT · peak{" "}
            {pct(active.utilisation)} · lanes [{active.config.map((c) => c.unroll).join(", ")}]
          </p>
        ) : (
          <p className="text-muted-dim">
            {front.length} non-dominated configurations. Hover a point for its numbers.
          </p>
        )}
      </div>
    </div>
  );
}
