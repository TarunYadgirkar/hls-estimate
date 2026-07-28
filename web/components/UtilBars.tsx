"use client";

import { motion, useReducedMotion } from "motion/react";
import { capacityOf, RESOURCE_KEYS, type Device, type ResourceKey } from "@/lib/model";
import { compact, pct } from "@/lib/format";

const LABEL: Record<ResourceKey, string> = {
  lut: "LUT",
  ff: "FF",
  dsp: "DSP",
  bram: "BRAM18",
};

const WHAT: Record<ResourceKey, string> = {
  lut: "logic",
  ff: "registers",
  dsp: "multipliers",
  bram: "on-chip memory",
};

export function UtilBars({
  device,
  totals,
  binding,
}: {
  device: Device;
  totals: Record<string, number>;
  binding: ResourceKey;
}) {
  const reduced = useReducedMotion();
  return (
    <ul className="grid grid-cols-2 gap-x-6 gap-y-5 lg:grid-cols-4">
      {RESOURCE_KEYS.map((res) => {
        const cap = capacityOf(device, res);
        const used = totals[res] ?? 0;
        const frac = used / cap;
        const over = frac > 1;
        const isBinding = res === binding;
        return (
          <li key={res}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-mono text-[11px] tracking-wider text-muted">
                {LABEL[res]}
                <span className="ml-1.5 text-muted-dim">{WHAT[res]}</span>
              </span>
              <span
                className={`tabular font-mono text-[11px] ${
                  over ? "text-over" : isBinding ? "text-signal" : "text-muted-dim"
                }`}
              >
                {pct(frac)}
              </span>
            </div>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-[1px] bg-substrate-3">
              <motion.div
                className="h-full rounded-[1px]"
                style={{
                  background: over
                    ? "var(--over)"
                    : isBinding
                      ? "var(--signal)"
                      : "var(--trace)",
                }}
                initial={false}
                animate={{ width: `${Math.min(100, frac * 100)}%` }}
                transition={reduced ? { duration: 0 } : { duration: 0.3, ease: "easeOut" }}
              />
            </div>
            <p className="tabular mt-1.5 font-mono text-[11px] text-muted-dim">
              {compact(used)} / {compact(cap)}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
