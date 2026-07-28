"use client";

import { Cpu, Layers, Rows3, Zap } from "lucide-react";
import { EXAMPLE_LIST, type Example } from "@/lib/examples";
import { DEVICES, type Device } from "@/lib/model";

const BIT_CHOICES = [4, 8, 16] as const;

export type ControlsProps = {
  example: Example;
  onExample: (key: string) => void;
  device: Device;
  onDevice: (key: string) => void;
  unroll: number;
  onUnroll: (n: number) => void;
  maxUnroll: number;
  bits: number | null;
  onBits: (b: number | null) => void;
  nativeBits: number[];
};

export function Controls({
  example,
  onExample,
  device,
  onDevice,
  unroll,
  onUnroll,
  maxUnroll,
  bits,
  onBits,
  nativeBits,
}: ControlsProps) {
  const steps = Math.max(1, Math.floor(Math.log2(maxUnroll)) + 1);
  const sliderValue = Math.round(Math.log2(unroll));

  return (
    <div className="flex flex-col gap-6">
      <Field icon={<Layers size={13} />} label="network">
        <select
          value={example.key}
          onChange={(e) => onExample(e.target.value)}
          className="w-full cursor-pointer rounded border border-edge bg-substrate-3 px-3 py-2 font-mono text-sm text-bone transition-colors hover:border-edge-bright"
        >
          {EXAMPLE_LIST.map((ex) => (
            <option key={ex.key} value={ex.key}>
              {ex.label}
            </option>
          ))}
        </select>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">{example.blurb}</p>
      </Field>

      <Field icon={<Cpu size={13} />} label="target device">
        <select
          value={device.name}
          onChange={(e) => onDevice(e.target.value)}
          className="w-full cursor-pointer rounded border border-edge bg-substrate-3 px-3 py-2 font-mono text-sm text-bone transition-colors hover:border-edge-bright"
        >
          {Object.values(DEVICES).map((d) => (
            <option key={d.name} value={d.name}>
              {d.label}
            </option>
          ))}
        </select>
      </Field>

      <Field icon={<Rows3 size={13} />} label="parallelism (unroll)">
        <div className="flex items-baseline justify-between">
          <span className="tabular font-mono text-2xl font-semibold text-signal">
            {unroll}×
          </span>
          <span className="font-mono text-[11px] text-muted-dim">
            max {maxUnroll}× for this network
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={steps - 1}
          step={1}
          value={sliderValue}
          onChange={(e) => onUnroll(2 ** Number(e.target.value))}
          aria-label="MAC lanes per layer"
          className="mt-2"
        />
        <p className="mt-1 text-[13px] leading-relaxed text-muted">
          How many multiply lanes each layer instantiates. More lanes finish sooner and
          cost more silicon.
        </p>
      </Field>

      <Field icon={<Zap size={13} />} label="weight precision">
        <div
          role="group"
          aria-label="Weight bit width"
          className="flex gap-1 rounded border border-edge bg-substrate-3 p-1"
        >
          <BitButton
            active={bits === null}
            onClick={() => onBits(null)}
            label={`native (${[...new Set(nativeBits)].map((b) => `int${b}`).join(" + ")})`}
          />
          {BIT_CHOICES.map((b) => (
            <BitButton
              key={b}
              active={bits === b}
              onClick={() => onBits(b)}
              label={`int${b}`}
            />
          ))}
        </div>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">
          Narrower weights pack more multiplies into one DSP: two at int4, four at int2.
        </p>
      </Field>
    </div>
  );
}

function BitButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`flex-1 whitespace-nowrap rounded-[2px] px-2 py-1.5 font-mono text-xs transition-colors ${
        active
          ? "bg-bone text-substrate"
          : "text-muted hover:bg-substrate-2 hover:text-bone"
      }`}
    >
      {label}
    </button>
  );
}

function Field({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="eyebrow mb-2 flex items-center gap-1.5">
        <span className="text-muted-dim">{icon}</span>
        {label}
      </div>
      {children}
    </div>
  );
}
