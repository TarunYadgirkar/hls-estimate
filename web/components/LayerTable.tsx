"use client";

import type { GraphEstimate, Graph } from "@/lib/model";
import { compact, pct } from "@/lib/format";

const KIND_LABEL: Record<string, string> = {
  conv2d: "conv 2d",
  linear: "dense",
  relu: "relu",
  maxpool2d: "max pool",
  add: "add",
};

export function LayerTable({
  graph,
  estimate,
}: {
  graph: Graph;
  estimate: GraphEstimate;
}) {
  const { perLayer, totals, bottleneck } = estimate;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[620px] border-collapse font-mono text-xs">
        <thead>
          <tr className="border-b border-edge text-left text-muted-dim">
            <Th className="text-left">layer</Th>
            <Th>lanes</Th>
            <Th>MACs</Th>
            <Th>DSP</Th>
            <Th>BRAM</Th>
            <Th>LUT</Th>
            <Th>FF</Th>
            <Th>cycles</Th>
          </tr>
        </thead>
        <tbody>
          {perLayer.map((layer, i) => {
            const isBottleneck = layer.name === bottleneck;
            return (
              <tr
                key={layer.name}
                className={`border-b border-edge/60 transition-colors hover:bg-substrate-3/60 ${
                  isBottleneck ? "bg-signal/[0.04]" : ""
                }`}
              >
                <td className="py-2.5 pr-4">
                  <span className="text-bone">{layer.name}</span>
                  <span className="ml-2 text-muted-dim">{KIND_LABEL[layer.kind]}</span>
                  {isBottleneck && (
                    <span className="ml-2 rounded-[2px] bg-signal/15 px-1.5 py-0.5 text-[10px] tracking-wide text-signal">
                      bottleneck
                    </span>
                  )}
                </td>
                <Td className="text-muted">{graph.nodes[i].knobs.unroll}×</Td>
                <Td className="text-muted">{compact(layer.macs)}</Td>
                <Td>{layer.dsp}</Td>
                <Td>{layer.bram}</Td>
                <Td>{compact(layer.lut)}</Td>
                <Td>{compact(layer.ff)}</Td>
                <Td className={isBottleneck ? "text-signal" : ""}>
                  {compact(layer.latency)}
                </Td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="text-bone">
            <td className="py-2.5 pr-4 text-muted-dim">total</td>
            <Td />
            <Td>{compact(totals.macs)}</Td>
            <Td>{totals.dsp}</Td>
            <Td>{totals.bram}</Td>
            <Td>{compact(totals.lut)}</Td>
            <Td>{compact(totals.ff)}</Td>
            <Td>{compact(totals.latency)}</Td>
          </tr>
        </tfoot>
      </table>
      <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-muted">
        Stages run concurrently under <Mono>DATAFLOW</Mono>, so the design&apos;s
        throughput is set by its slowest stage —{" "}
        <span className="text-signal">{bottleneck}</span> at{" "}
        {compact(totals.latency)} cycles — not by the sum of all of them (
        {compact(totals.latency_seq)}). Speed up anything else and the number above
        will not move.
      </p>
      <p className="mt-2 text-[13px] text-muted-dim">
        Share of the design&apos;s DSPs held by {bottleneck}:{" "}
        {pct(
          (perLayer.find((l) => l.name === bottleneck)?.dsp ?? 0) /
            Math.max(1, totals.dsp),
        )}
        .
      </p>
    </div>
  );
}

function Th({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return (
    <th className={`py-2 pr-4 text-right font-normal tracking-wider ${className}`}>
      {children}
    </th>
  );
}

function Td({ children, className = "" }: { children?: React.ReactNode; className?: string }) {
  return <td className={`tabular py-2.5 pr-4 text-right ${className}`}>{children}</td>;
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-[2px] bg-substrate-3 px-1 py-0.5 font-mono text-[12px] text-bone">
      {children}
    </code>
  );
}
