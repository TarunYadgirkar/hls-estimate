"use client";

import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import { CircleCheck, TriangleAlert } from "lucide-react";

import { Controls } from "@/components/Controls";
import { Floorplan } from "@/components/Floorplan";
import { LayerTable } from "@/components/LayerTable";
import { CodePanel } from "@/components/CodePanel";
import { ParetoChart } from "@/components/ParetoChart";
import { UtilBars } from "@/components/UtilBars";

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
  capacityOf,
  DEVICES,
  estimateGraph,
  fitsDevice,
  type Graph,
} from "@/lib/model";
import { compact, cyclesToTime, pct } from "@/lib/format";

const TABS = ["layers", "generated C++", "design space"] as const;
type Tab = (typeof TABS)[number];

export default function Home() {
  const reduced = useReducedMotion();
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
    <main className="mx-auto max-w-[1400px] px-5 pb-4 pt-8 sm:px-8">
      {/* Hero: the instrument itself, not a banner about the instrument. */}
      <section className="grid gap-10 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:gap-14">
        <div>
          <h1 className="font-mono text-[2.6rem] font-semibold leading-[1.05] tracking-tight sm:text-[3.2rem]">
            Will it
            <br />
            fit?
          </h1>
          <p className="mt-5 max-w-md text-[15px] leading-relaxed text-muted">
            Give hls-estimate a small quantized network and a target FPGA. It predicts
            the LUTs, flip-flops, DSPs, block RAM and cycles you will spend — and writes
            the synthesizable HLS C++ — without running synthesis once.
          </p>

          <div className="mt-8 border-t border-edge pt-8">
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
                className="mt-4 font-mono text-xs text-trace underline underline-offset-4 hover:text-bone"
              >
                clear the configuration loaded from the design space
              </button>
            )}
          </div>
        </div>

        <div className="min-w-0">
          <div className="panel substrate-grid p-5 sm:p-7">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div className="eyebrow">{device.label} — fabric</div>
              <motion.div
                key={fits ? "fits" : "over"}
                initial={reduced ? false : { opacity: 0, y: -3 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex items-center gap-2 font-mono text-sm ${
                  fits ? "text-signal" : "text-over"
                }`}
              >
                {fits ? <CircleCheck size={15} /> : <TriangleAlert size={15} />}
                {fits ? "fits" : "does not fit"}
                <span className="text-muted-dim">
                  · {binding.toUpperCase()} binding at{" "}
                  {pct(estimate.totals[binding] / capacityOf(device, binding))}
                </span>
              </motion.div>
            </div>

            <Floorplan device={device} totals={estimate.totals} fits={fits} />

            <div className="mt-7 border-t border-edge pt-6">
              <UtilBars device={device} totals={estimate.totals} binding={binding} />
            </div>

            <dl className="mt-7 grid grid-cols-2 gap-6 border-t border-edge pt-6 sm:grid-cols-3">
              <Stat
                label="throughput"
                value={`${compact(estimate.totals.latency)} cyc`}
                sub={`${cyclesToTime(estimate.totals.latency)} at 100 MHz`}
              />
              <Stat
                label="total MACs"
                value={compact(estimate.totals.macs)}
                sub="multiply-accumulates per inference"
              />
              <Stat
                label="bottleneck"
                value={estimate.bottleneck}
                sub="slowest dataflow stage"
              />
            </dl>
          </div>
        </div>
      </section>

      {/* Workbench */}
      <section className="mt-16">
        <div className="flex flex-wrap gap-1 border-b border-edge">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              aria-current={tab === t}
              className={`relative px-3 py-2.5 font-mono text-xs transition-colors ${
                tab === t ? "text-bone" : "text-muted-dim hover:text-muted"
              }`}
            >
              {t}
              {tab === t && (
                <motion.span
                  layoutId="tab-underline"
                  className="absolute inset-x-0 -bottom-px h-px bg-signal"
                  transition={{ duration: reduced ? 0 : 0.2 }}
                />
              )}
            </button>
          ))}
        </div>

        <div className="pt-7">
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
        </div>
      </section>

      <section className="mt-20 border-t border-edge pt-8">
        <p className="max-w-3xl text-[15px] leading-relaxed text-muted">
          <span className="text-bone">
            These numbers are wrong, and by how much is written down.
          </span>{" "}
          Against published hls4ml results the DSP estimate runs about 27% high on
          fully-parallel designs and 113% high on a DSP-saturated CNN; the BRAM model
          has never been validated against anything.{" "}
          <Link
            href="/model"
            className="text-trace underline underline-offset-4 hover:text-bone"
          >
            Read the assumptions and the measured error
          </Link>
          , or start with{" "}
          <Link
            href="/what-is-this"
            className="text-trace underline underline-offset-4 hover:text-bone"
          >
            what this thing is for
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
      <dt className="eyebrow">{label}</dt>
      <dd className="tabular mt-1.5 font-mono text-xl text-bone">{value}</dd>
      <dd className="mt-1 text-[12px] leading-snug text-muted-dim">{sub}</dd>
    </div>
  );
}
