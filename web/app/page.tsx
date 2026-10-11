"use client";

import { useMemo, useState } from "react";
import Link from "next/link";

import { Controls } from "@/components/Controls";
import { Floorplan } from "@/components/Floorplan";
import { LayerTable } from "@/components/LayerTable";
import { CodePanel } from "@/components/CodePanel";
import { ParetoChart } from "@/components/ParetoChart";
import { UtilBars } from "@/components/UtilBars";
import { Verdict } from "@/components/Verdict";
import { Workbench, type Tab } from "@/components/Workbench";

import { EXAMPLES } from "@/lib/examples";
import { emitGraph } from "@/lib/codegen";
import {
  explore,
  isTruncated,
  maxParallel,
  spaceSize,
  type DesignPoint,
} from "@/lib/dse";
import {
  bindingResource,
  DEVICES,
  estimateGraph,
  fitsDevice,
  type Graph,
} from "@/lib/model";
import { compact, cyclesToTime } from "@/lib/format";

export default function Home() {
  const [exampleKey, setExampleKey] = useState("conv_relu_pool");
  const [deviceKey, setDeviceKey] = useState("zynq-7020");
  const [unroll, setUnroll] = useState(16);
  const [bits, setBits] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("layers");
  const [pinned, setPinned] = useState<number[] | null>(null);

  const example = EXAMPLES[exampleKey];
  const device = DEVICES[deviceKey];

  const { graph, nativeBits, maxUnroll } = useMemo(() => {
    const g: Graph = example.build();
    const native = g.nodes
      .filter((n) => n.isMac)
      .map((n) => (n.kind === "conv2d" || n.kind === "linear" ? n.wBits : 8));
    const cap = Math.max(...g.nodes.map(maxParallel));
    g.nodes.forEach((node, i) => {
      const lanes = pinned
        ? pinned[i]
        : Math.min(unroll, Math.max(1, maxParallel(node)));
      node.knobs = { unroll: lanes, tile: 0, pipeline: true };
      if (bits !== null && (node.kind === "conv2d" || node.kind === "linear")) {
        node.wBits = bits;
      }
    });
    return {
      graph: g,
      nativeBits: native,
      maxUnroll: 2 ** Math.floor(Math.log2(cap)),
    };
  }, [example, unroll, bits, pinned]);

  const estimate = useMemo(() => estimateGraph(graph), [graph]);
  const fits = fitsDevice(estimate.totals, device);
  const binding = bindingResource(estimate.totals, device);
  const source = useMemo(() => emitGraph(graph, "net"), [graph]);
  const front = useMemo(() => explore(example.build(), device), [example, device]);
  const truncatedFrom = useMemo(() => {
    const g = example.build();
    return isTruncated(g) ? spaceSize(g) : null;
  }, [example]);

  function pickPoint(pt: DesignPoint) {
    setPinned(pt.config.map((c) => c.unroll));
  }

  function setUnrollAndClear(n: number) {
    setPinned(null);
    setUnroll(n);
  }

  return (
    <main className="mx-auto max-w-[1400px] px-4 pb-4 pt-8 sm:px-8 sm:pt-10">
      <section className="grid items-end gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)] lg:gap-14">
        <Verdict
          exampleLabel={example.label}
          device={device}
          totals={estimate.totals}
          fits={fits}
          binding={binding}
          bottleneck={estimate.bottleneck}
        />
        <p className="max-w-[60ch] text-[15px] leading-relaxed text-muted">
          Pick a small quantized network and a target FPGA. <span className="whitespace-nowrap">hls-estimate</span> predicts the
          LUTs, flip-flops, DSPs, block RAM and cycles it will cost, and writes the
          synthesizable HLS C++, without running synthesis.
        </p>
      </section>

      <section className="mt-8 grid gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-10">
        <div className="panel h-fit p-5">
          <h2 className="mb-5 text-sm font-semibold text-bone">Design</h2>
          <Controls
            example={example}
            onExample={(k) => {
              setPinned(null);
              setExampleKey(k);
            }}
            device={device}
            onDevice={setDeviceKey}
            unroll={pinned ? Math.max(...pinned) : unroll}
            onUnroll={setUnrollAndClear}
            maxUnroll={maxUnroll}
            bits={bits}
            onBits={setBits}
            nativeBits={nativeBits}
          />
          {pinned && (
            <button
              type="button"
              onClick={() => setPinned(null)}
              className="link mt-5 text-left text-[13px]"
            >
              Clear the configuration loaded from the design space
            </button>
          )}
        </div>

        <div className="panel substrate-grid min-w-0 p-5 sm:p-7">
          <div className="mb-5 flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold text-bone">{device.label} fabric</h2>
            <p className="text-[13px] text-muted">Each column type lights up in proportion to its estimated use.</p>
          </div>

          <UtilBars device={device} totals={estimate.totals} binding={binding} />

          <div className="mt-7">
            <Floorplan device={device} totals={estimate.totals} fits={fits} />
          </div>

          <dl className="mt-7 grid grid-cols-2 gap-6 pt-6 shadow-[0_-1px_0_var(--edge)] sm:grid-cols-3">
            <Stat
              label="Cycles per inference"
              value={compact(estimate.totals.latency)}
              sub={`${cyclesToTime(estimate.totals.latency)} at 100 MHz`}
            />
            <Stat
              label="Multiply-accumulates"
              value={compact(estimate.totals.macs)}
              sub="per inference"
            />
            <Stat
              label="Slowest stage"
              value={estimate.bottleneck}
              sub="sets the throughput"
            />
          </dl>
        </div>
      </section>

      <Workbench tab={tab} onTab={setTab}>
        {tab === "layers" && <LayerTable graph={graph} estimate={estimate} />}
        {tab === "generated C++" && <CodePanel source={source} />}
        {tab === "design space" && (
          <ParetoChart
            front={front}
            currentLatency={estimate.totals.latency}
            onPick={pickPoint}
            truncatedFrom={truncatedFrom}
          />
        )}
      </Workbench>

      <section className="mt-20 pt-8 shadow-[0_-1px_0_var(--edge)]">
        <h2 className="text-base font-semibold text-bone">How wrong these numbers are</h2>
        <p className="mt-3 max-w-[65ch] text-[15px] leading-relaxed text-muted">
          Against published hls4ml results, the DSP estimate runs about 27% high on
          fully parallel designs and 113% high on a DSP-saturated CNN. The BRAM model has
          not been validated against anything yet.{" "}
          <Link href="/model" className="link">
            Read the assumptions and the measured error
          </Link>
          , or start with{" "}
          <Link href="/what-is-this" className="link">
            what this tool is for
          </Link>
          .
        </p>
      </section>
    </main>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="tabular mt-1.5 font-mono text-xl text-bone">{value}</dd>
      <dd className="mt-1 text-[13px] leading-snug text-muted-dim">{sub}</dd>
    </div>
  );
}
