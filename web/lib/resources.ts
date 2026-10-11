import type { ResourceKey } from "@/lib/model";

type ResourceMeta = { label: string; what: string; color: string };

// One definition of how each resource is named and coloured. FF lives in the same
// CLB columns as LUTs on the die, so it shares the logic hue.
export const RESOURCE_META: Record<ResourceKey, ResourceMeta> = {
  lut: { label: "LUT", what: "logic", color: "var(--res-logic)" },
  ff: { label: "FF", what: "registers", color: "var(--res-logic)" },
  dsp: { label: "DSP", what: "multipliers", color: "var(--res-dsp)" },
  bram: { label: "BRAM18", what: "on-chip memory", color: "var(--res-bram)" },
};
