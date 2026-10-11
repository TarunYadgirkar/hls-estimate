"use client";

import { motion, useReducedMotion } from "motion/react";
import { capacityOf, RESOURCE_KEYS, type Device, type ResourceKey } from "@/lib/model";
import { RESOURCE_META } from "@/lib/resources";
import { compact, pct } from "@/lib/format";

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
        const meta = RESOURCE_META[res];
        const cap = capacityOf(device, res);
        const used = totals[res] ?? 0;
        const frac = used / cap;
        const over = frac > 1;
        const isBinding = res === binding;
        return (
          <li key={res} className="min-w-0">
            <div className="flex items-start justify-between gap-2">
              <span className="text-xs leading-snug text-muted">
                <span className="font-mono font-medium text-bone">{meta.label}</span>{" "}
                {meta.what}
              </span>
              <span
                className={`tabular font-mono text-sm ${over ? "text-over" : "text-bone"}`}
              >
                {pct(frac)}
              </span>
            </div>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-[2px] bg-substrate-3">
              <motion.div
                className="h-full rounded-[2px]"
                style={{ background: over ? "var(--over)" : meta.color }}
                initial={false}
                animate={{ width: `${Math.min(100, frac * 100)}%` }}
                transition={reduced ? { duration: 0 } : { duration: 0.3, ease: "easeOut" }}
              />
            </div>
            <p className="tabular mt-1.5 flex justify-between gap-2 font-mono text-[11px] text-muted-dim">
              <span>
                {compact(used)} / {compact(cap)}
              </span>
              {isBinding && <span className={over ? "text-over" : "text-bone"}>tightest</span>}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
