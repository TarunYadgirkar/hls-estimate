"use client";

import { EXAMPLE_LIST, type Example } from "@/lib/examples";
import { DEVICES, type Device } from "@/lib/model";

const BIT_CHOICES = [4, 8, 16] as const;

const SELECT =
  "w-full cursor-pointer rounded border border-edge-bright bg-substrate-3 px-3 py-2 font-mono text-sm text-bone transition-colors hover:border-muted-dim";

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
      <Field label="Network" htmlFor="ctl-network">
        <select
          id="ctl-network"
          value={example.key}
          onChange={(e) => onExample(e.target.value)}
          className={SELECT}
        >
          {EXAMPLE_LIST.map((ex) => (
            <option key={ex.key} value={ex.key}>
              {ex.label}
            </option>
          ))}
        </select>
        <p className="mt-2 text-[13px] leading-relaxed text-muted">{example.blurb}</p>
      </Field>

      <Field label="Target device" htmlFor="ctl-device">
        <select
          id="ctl-device"
          value={device.name}
          onChange={(e) => onDevice(e.target.value)}
          className={SELECT}
        >
          {Object.values(DEVICES).map((d) => (
            <option key={d.name} value={d.name}>
              {d.label}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Parallelism (unroll)" htmlFor="ctl-unroll">
        <div className="flex items-baseline justify-between">
          <output
            htmlFor="ctl-unroll"
            className="tabular font-mono text-2xl font-semibold text-bone"
          >
            {unroll}×
          </output>
          <span className="tabular text-xs text-muted-dim">
            up to {maxUnroll}× for this network
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={steps - 1}
          step={1}
          value={sliderValue}
          onChange={(e) => onUnroll(2 ** Number(e.target.value))}
          id="ctl-unroll"
          aria-valuetext={`${unroll} multiply lanes per layer`}
          className="mt-2"
        />
        <p className="mt-1 text-[13px] leading-relaxed text-muted">
          How many multiply lanes each layer instantiates. More lanes finish sooner and
          cost more silicon.
        </p>
      </Field>

      <Field label="Weight precision">
        <div
          role="group"
          aria-label="Weight bit width"
          className="flex gap-1 rounded bg-substrate-3 p-1 shadow-[inset_0_0_0_1px_var(--edge)]"
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
      className={`min-h-8 flex-1 whitespace-nowrap rounded-[3px] px-2 py-1.5 font-mono text-xs transition-colors ${
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
  label,
  htmlFor,
  children,
}: {
  label: string;
  htmlFor?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      {htmlFor ? (
        <label htmlFor={htmlFor} className="label mb-2 block">
          {label}
        </label>
      ) : (
        <p className="label mb-2">{label}</p>
      )}
      {children}
    </div>
  );
}
