import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "The model — hls-estimate",
  description:
    "Every assumption behind the analytical resource model, with the measured error against published hls4ml results.",
  alternates: { canonical: "/model" },
  openGraph: {
    title: "The model — hls-estimate",
    description:
      "Every assumption behind the analytical resource model, with the measured error against published hls4ml results.",
    url: "/model",
    siteName: "hls-estimate",
    type: "website",
  },
};

type CalRow = {
  design: string;
  resource: string;
  predicted: number;
  published: number;
  independent: boolean;
};

const CALIBRATION: CalRow[] = [
  {
    design: "hls4ml jet tagger, 16-bit, RF=1",
    resource: "DSP",
    predicted: 4256,
    published: 3329,
    independent: true,
  },
  {
    design: "hls4ml jet tagger, pruned",
    resource: "DSP",
    predicted: 1205,
    published: 954,
    independent: true,
  },
  {
    design: "hls4ml SVHN CNN, 14-bit",
    resource: "DSP",
    predicted: 13552,
    published: 6377,
    independent: true,
  },
  {
    design: "hls4ml jet tagger, 16-bit, RF=1",
    resource: "LUT+FF",
    predicted: 282729,
    published: 263234,
    independent: false,
  },
  {
    design: "hls4ml jet tagger, pruned",
    resource: "LUT+FF",
    predicted: 81121,
    published: 88797,
    independent: true,
  },
  {
    design: "hls4ml SVHN CNN, 14-bit",
    resource: "LUT",
    predicted: 598316,
    published: 228823,
    independent: true,
  },
  {
    design: "hls4ml SVHN CNN, 14-bit",
    resource: "FF",
    predicted: 300858,
    published: 80278,
    independent: true,
  },
];

function errorClass(ratio: number): string {
  const err = Math.abs(ratio - 1);
  if (err <= 0.15) return "text-signal";
  if (err <= 0.5) return "text-bone";
  return "text-over";
}

export default function ModelPage() {
  return (
    <main className="mx-auto max-w-[860px] px-5 pb-4 pt-12 sm:px-8">
      <p className="eyebrow">the contribution</p>
      <h1 className="mt-3 font-mono text-[2.2rem] font-semibold leading-[1.1] tracking-tight sm:text-[2.8rem]">
        The model,
        <br />
        and where it breaks
      </h1>
      <p className="mt-6 text-[17px] leading-relaxed text-muted">
        The equations are simple. The assumptions behind them are the whole point, so
        they are listed here with the error each one causes — measured against published
        numbers, not estimated.
      </p>

      <Section title="Calibration against published results">
        <p>
          Reference designs come from the hls4ml literature, where the &ldquo;reuse
          factor&rdquo; is exactly the inverse of this tool&apos;s parallelism knob, so
          the comparison is like-for-like.
        </p>
        <div className="mt-5 overflow-x-auto">
          <table className="w-full min-w-[600px] border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-edge text-left text-muted-dim">
                <th className="py-2 pr-4 font-normal tracking-wider">reference design</th>
                <th className="py-2 pr-4 font-normal tracking-wider">resource</th>
                <th className="py-2 pr-4 text-right font-normal tracking-wider">
                  predicted
                </th>
                <th className="py-2 pr-4 text-right font-normal tracking-wider">
                  published
                </th>
                <th className="py-2 text-right font-normal tracking-wider">error</th>
              </tr>
            </thead>
            <tbody>
              {CALIBRATION.map((row, i) => {
                const ratio = row.predicted / row.published;
                const errPct = Math.round((ratio - 1) * 100);
                return (
                  <tr key={i} className="border-b border-edge/60">
                    <td className="py-2.5 pr-4 text-bone">
                      {row.design}
                      {!row.independent && (
                        <span className="ml-2 text-[10px] tracking-wide text-muted-dim">
                          fitted here
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 pr-4 text-muted">{row.resource}</td>
                    <td className="tabular py-2.5 pr-4 text-right text-muted">
                      {row.predicted.toLocaleString("en-US")}
                    </td>
                    <td className="tabular py-2.5 pr-4 text-right text-muted">
                      {row.published.toLocaleString("en-US")}
                    </td>
                    <td
                      className={`tabular py-2.5 text-right ${errorClass(ratio)}`}
                    >
                      {errPct > 0 ? "+" : ""}
                      {errPct}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-[13px] leading-relaxed text-muted-dim">
          Sources: J. Duarte et al., JINST 13 (2018) P07027 (arXiv:1804.06913), Table 2;
          T. Aarrestad et al., Mach. Learn.: Sci. Technol. 2 (2021) 045015
          (arXiv:2101.05108), Tables 1, 3 and 4.
        </p>
      </Section>

      <Section title="DSP — the model that works">
        <Formula>DSP = ceil(lanes / macs_per_dsp(bits))</Formula>
        <Formula>macs_per_dsp = &#123; 16:1, 8:1, 4:2, 2:4 &#125;</Formula>
        <p>
          One DSP retires one 8-bit multiply-accumulate. Narrower operands pack along the
          parallelism axis: a DSP48&apos;s 25×18 multiplier holds two int4 operands with
          guard bits between them, so int4 costs exactly half the DSPs of int8 at equal
          parallelism.
        </p>
        <Assumptions
          items={[
            [
              "int8 gets one MAC per DSP, not two",
              "Xilinx WP486 shows two int8 MACs on one DSP48E2, but that needs a shared operand and an UltraScale+ part. The Zynq-7020's DSP48E1 cannot. Conservative by choice.",
            ],
            [
              "No DSP saturation, no LUT spillover",
              "Real tools move multiplications into LUT fabric when DSPs run out, and prefer LUTs for narrow multiplies below about 10 bits. This is the single largest source of error and why the saturated CNN above is 113% off.",
            ],
            [
              "No constant folding",
              "Multiplies by zero, ±1 or a power of two are free in real synthesis. We charge full price — most of the residual 27%.",
            ],
          ]}
        />
      </Section>

      <Section title="LUT and FF — fitted, and fragile">
        <Formula>LUT = 120 + 2.75 · lanes · weight_bits + 90</Formula>
        <Formula>FF = 100 + 0.69 · lanes · (weight_bits + act_bits) + 64</Formula>
        <p>
          Those two coefficients were fitted to a single reference design. Before
          calibration the hand-picked constants were four times too high. They are within
          7% on the network they were fitted to, 9% on a pruned version of it, and{" "}
          <span className="text-over">161% and 275% high on a CNN they were not</span>.
        </p>
        <p>
          The CNN case fails for a specific, fixable reason: hls4ml streams a convolution
          spatially and shares one datapath across many pixels, while this model charges
          for every lane independently. Charging per distinct kernel position instead
          would likely close most of the gap. That is the highest-value fix on the list.
        </p>
      </Section>

      <Section title="BRAM — unvalidated">
        <Formula>bram18(depth, width) = ceil(width/18) · ceil(depth/1024)</Formula>
        <p>
          Weights, a line buffer and an output tile per layer, with array partitioning
          fragmenting memory so that more parallelism costs more BRAM.
        </p>
        <p className="rounded border border-over/30 bg-over/[0.06] px-4 py-3 text-[14px] text-bone">
          Neither reference design reports BRAM in a comparable form, so this number has
          never been checked against reality. It is dimensionally sane and monotonic in
          the knobs, and that is all that is tested. Do not trust it.
        </p>
      </Section>

      <Section title="Latency">
        <Formula>cycles = ceil(work / lanes) · II + 8</Formula>
        <p>
          Initiation interval is 1 when the loop is pipelined. Under <Mono>DATAFLOW</Mono>{" "}
          the stages run concurrently, so a graph&apos;s figure is the{" "}
          <em>maximum</em> over its stages, not the sum — that is throughput, not
          end-to-end latency for one inference. Real designs miss II=1 on loop-carried
          dependencies, and off-chip bandwidth is assumed free, so this is optimistic.
          Also unvalidated: it is tested only for monotonicity.
        </p>
      </Section>

      <Section title="What the tests actually prove">
        <Assumptions
          items={[
            [
              "Bit-exact codegen — the one that carries weight",
              "Emitted C++ compiles as plain C++ and matches the PyTorch model exactly on random inputs. Breaking the rounding term on purpose fails 5 of its 6 cases, so it is not vacuous.",
            ],
            [
              "Analytical sanity",
              "DSP matches the closed form exactly. Says nothing about whether the closed form matches hardware.",
            ],
            [
              "Monotonicity, over randomized configs",
              "More lanes never lowers DSP; bigger tiles never lower BRAM; more parallelism always lowers latency. Directions only, never magnitudes.",
            ],
            [
              "Budget enforcement",
              "Design space exploration never returns a configuration that exceeds the device.",
            ],
            [
              "Calibration",
              "Totals land inside the measured bands in the table above. Proves nothing about networks outside those two.",
            ],
          ]}
        />
      </Section>

      <Section title="Ranked fixes">
        <ol className="flex list-decimal flex-col gap-2 pl-5 marker:font-mono marker:text-muted-dim">
          <li>Model DSP saturation and LUT spillover — fixes the 113% CNN error.</li>
          <li>
            Charge logic per distinct kernel position for streamed convolutions — fixes
            most of the 161% and 275% logic errors.
          </li>
          <li>Validate BRAM against anything at all.</li>
          <li>Model accumulator width instead of assuming 64 bits.</li>
          <li>Model constant folding of zero and ±1 weights.</li>
        </ol>
      </Section>

      <div className="mt-14 border-t border-edge pt-8">
        <Link className="link font-mono text-sm" href="/">
          ← back to the estimator
        </Link>
      </div>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-12">
      <h2 className="font-mono text-lg font-semibold tracking-tight text-bone">
        {title}
      </h2>
      <div className="mt-4 flex flex-col gap-4 text-[15px] leading-relaxed text-muted [&_em]:text-bone [&_em]:not-italic">
        {children}
      </div>
    </section>
  );
}

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <p className="rounded border border-edge bg-substrate-2 px-4 py-2.5 font-mono text-[13px] text-bone">
      {children}
    </p>
  );
}

function Assumptions({ items }: { items: [string, string][] }) {
  return (
    <dl className="my-1 flex flex-col gap-4 border-l border-edge pl-5">
      {items.map(([term, body]) => (
        <div key={term}>
          <dt className="font-mono text-[13px] text-bone">{term}</dt>
          <dd className="mt-1 text-[14px] leading-relaxed text-muted">{body}</dd>
        </div>
      ))}
    </dl>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-[2px] bg-substrate-3 px-1.5 py-0.5 font-mono text-[13px] text-bone">
      {children}
    </code>
  );
}
