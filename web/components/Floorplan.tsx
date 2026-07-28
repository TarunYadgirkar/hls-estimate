"use client";

/**
 * The device floorplan — the signature element.
 *
 * A real Xilinx die is organised into vertical columns of fabric: mostly CLBs, with
 * periodic columns of DSP slices and block RAM. This draws that structure at a fixed
 * cell budget and lights cells up in proportion to the estimated usage of each
 * resource. Overflow past the die boundary is drawn outside the outline in amber:
 * the design does not fit, and you can see exactly which column ran out.
 */

import { motion, useReducedMotion } from "motion/react";
import { useMemo } from "react";
import type { Device, ResourceKey } from "@/lib/model";
import { capacityOf } from "@/lib/model";

const COLS = 34;
const ROWS = 18;

// Column layout mirrors a 7-series floorplan: DSP and BRAM columns are sparse and
// interleaved among CLB columns.
const DSP_COLUMNS = new Set([6, 15, 24, 31]);
const BRAM_COLUMNS = new Set([3, 11, 20, 28]);

type ColumnKind = "clb" | "dsp" | "bram";

function columnKind(col: number): ColumnKind {
  if (DSP_COLUMNS.has(col)) return "dsp";
  if (BRAM_COLUMNS.has(col)) return "bram";
  return "clb";
}

const KIND_COLOR: Record<ColumnKind, string> = {
  clb: "var(--trace)",
  dsp: "var(--signal)",
  bram: "#a78bfa",
};

// Which estimate drives which column type. CLB columns carry logic (LUT+FF).
const KIND_RESOURCE: Record<ColumnKind, ResourceKey> = {
  clb: "lut",
  dsp: "dsp",
  bram: "bram",
};

export type FloorplanProps = {
  device: Device;
  totals: Record<string, number>;
  fits: boolean;
};

type Cell = {
  col: number;
  row: number;
  kind: ColumnKind;
  filled: boolean;
  over: boolean;
  index: number;
};

export function Floorplan({ device, totals, fits }: FloorplanProps) {
  const reduced = useReducedMotion();

  const { cells, overflow } = useMemo(() => {
    const byKind: Record<ColumnKind, number[]> = { clb: [], dsp: [], bram: [] };
    for (let c = 0; c < COLS; c++) byKind[columnKind(c)].push(c);

    const out: Cell[] = [];
    const overflowByKind: Record<ColumnKind, number> = { clb: 0, dsp: 0, bram: 0 };
    let index = 0;

    for (const kind of ["clb", "dsp", "bram"] as ColumnKind[]) {
      const cols = byKind[kind];
      const capacityCells = cols.length * ROWS;
      const res = KIND_RESOURCE[kind];
      const used = totals[res] ?? 0;
      const fraction = used / capacityOf(device, res);
      const litCells = Math.min(capacityCells, Math.round(fraction * capacityCells));
      overflowByKind[kind] = Math.max(0, fraction - 1);

      let lit = 0;
      // Fill bottom-up within each column, column by column: the way a placer packs.
      for (const col of cols) {
        for (let r = ROWS - 1; r >= 0; r--) {
          const filled = lit < litCells;
          if (filled) lit++;
          out.push({
            col,
            row: r,
            kind,
            filled,
            over: filled && fraction > 1,
            index: index++,
          });
        }
      }
    }
    out.sort((a, b) => a.col - b.col || a.row - b.row);
    return { cells: out, overflow: overflowByKind };
  }, [device, totals]);

  const worstOverflow = Math.max(...Object.values(overflow));

  return (
    <figure className="relative">
      <svg
        viewBox="0 0 340 190"
        className="w-full"
        role="img"
        aria-label={`Device floorplan for ${device.label}: ${
          fits ? "the design fits" : "the design does not fit"
        }`}
      >
        <defs>
          <linearGradient id="dieEdge" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--edge-bright)" />
            <stop offset="100%" stopColor="var(--edge)" />
          </linearGradient>
        </defs>

        {/* Die outline */}
        <rect
          x="1"
          y="1"
          width="338"
          height="188"
          fill="none"
          stroke={fits ? "url(#dieEdge)" : "var(--over)"}
          strokeWidth="1"
          rx="2"
        />

        <g transform="translate(4 4)">
          {cells.map((cell) => {
            const x = cell.col * (332 / COLS);
            const y = cell.row * (182 / ROWS);
            const w = 332 / COLS - 1.1;
            const h = 182 / ROWS - 1.1;
            const color = cell.over ? "var(--over)" : KIND_COLOR[cell.kind];
            return (
              <motion.rect
                key={`${cell.col}-${cell.row}`}
                x={x}
                y={y}
                width={w}
                height={h}
                rx="0.6"
                // Unused fabric keeps its column colour at low alpha, so the die's
                // CLB / DSP / BRAM column structure stays legible when a design
                // occupies only a percent or two of it.
                initial={false}
                animate={{
                  fill: cell.filled ? color : KIND_COLOR[cell.kind],
                  opacity: cell.filled ? (cell.over ? 0.95 : 0.85) : 0.12,
                }}
                transition={
                  reduced
                    ? { duration: 0 }
                    : {
                        duration: 0.18,
                        delay: Math.min(0.22, cell.index * 0.0004),
                        ease: "easeOut",
                      }
                }
              />
            );
          })}
        </g>
      </svg>

      <figcaption className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] text-muted-dim">
        <Legend color="var(--trace)" label="CLB — logic" />
        <Legend color="var(--signal)" label="DSP — multipliers" />
        <Legend color="#a78bfa" label="BRAM — buffers" />
        {!fits && (
          <span className="text-over">
            overflow +{Math.round(worstOverflow * 100)}% past the die
          </span>
        )}
      </figcaption>

      <p className="mt-3 text-[13px] leading-relaxed text-muted">
        {fits ? (
          <>
            Lit cells are what this design claims. It occupies{" "}
            <span className="text-bone">
              {formatShare(totals.dsp / device.dsp)} of the multiplier columns
            </span>{" "}
            and {formatShare(totals.bram / device.bram18)} of the block RAM — the rest
            of the chip is still free.
          </>
        ) : (
          <>
            This design wants more fabric than the die has. Raise the precision, cut
            the parallelism, or move to a larger part.
          </>
        )}
      </p>
    </figure>
  );
}

function formatShare(x: number): string {
  if (x > 0 && x < 0.01) return "under 1%";
  return `${(x * 100).toFixed(x >= 0.1 ? 0 : 1)}%`;
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-block h-2 w-2 rounded-[1px]"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}

export { COLS as FLOORPLAN_COLS, ROWS as FLOORPLAN_ROWS };
