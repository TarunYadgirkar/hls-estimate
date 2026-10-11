"use client";

import { CheckCircle, WarningOctagon } from "@phosphor-icons/react";
import { capacityOf, type Device, type ResourceKey } from "@/lib/model";
import { RESOURCE_META } from "@/lib/resources";
import { compact, cyclesToTime, pct } from "@/lib/format";

export function Verdict({
  exampleLabel,
  device,
  totals,
  fits,
  binding,
  bottleneck,
}: {
  exampleLabel: string;
  device: Device;
  totals: Record<string, number>;
  fits: boolean;
  binding: ResourceKey;
  bottleneck: string;
}) {
  const share = totals[binding] / capacityOf(device, binding);
  const resource = RESOURCE_META[binding].label;
  const chip = device.label.split(" (")[0];

  return (
    <div className="min-w-0">
      <p className="text-[13px] text-muted">
        <span className="font-mono text-bone">{exampleLabel}</span> on {device.label}
      </p>
      <h1 className="mt-2 flex items-start gap-3 font-display sm:gap-4 text-[2.5rem] font-semibold leading-[1.02] tracking-[-0.01em] sm:text-[3.5rem]">
        {fits ? (
          <CheckCircle aria-hidden weight="fill" className="mt-[0.12em] shrink-0 text-ok" size="0.8em" />
        ) : (
          <WarningOctagon aria-hidden weight="fill" className="mt-[0.12em] shrink-0 text-over" size="0.8em" />
        )}
        <span>{fits ? `Fits on the ${chip}` : `Does not fit on the ${chip}`}</span>
      </h1>
      <p className="tabular mt-3 text-[15px] leading-relaxed text-muted">
        <span className={fits ? "text-bone" : "text-over"}>
          {resource} is the tightest resource at {pct(share)}
        </span>
        . One inference takes {compact(totals.latency)} cycles (
        {cyclesToTime(totals.latency)} at 100 MHz), paced by {bottleneck}.
      </p>
      <p className="sr-only" aria-live="polite">
        {fits ? "Design fits." : "Design does not fit."}
      </p>
    </div>
  );
}
