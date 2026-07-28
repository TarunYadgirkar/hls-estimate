import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "What is this? — hls-estimate",
  description:
    "A plain-language explanation of what hls-estimate does, who it is for, and what an FPGA resource estimate actually means.",
};

export default function WhatIsThis() {
  return (
    <main className="mx-auto max-w-[760px] px-5 pb-4 pt-12 sm:px-8">
      <p className="eyebrow">orientation</p>
      <h1 className="mt-3 font-mono text-[2.2rem] font-semibold leading-[1.1] tracking-tight sm:text-[2.8rem]">
        What is this?
      </h1>
      <p className="mt-6 text-[17px] leading-relaxed text-muted">
        A tool that answers one question about putting a small neural network on an
        FPGA: <span className="text-bone">will it fit, and how fast will it run?</span>{" "}
        It answers in milliseconds, in a browser, without the hours of synthesis that
        would normally be required.
      </p>

      <Section title="The problem it solves">
        <p>
          Running a neural network on an FPGA means turning it into a circuit. The usual
          way to find out whether that circuit fits on your chip is to write the code,
          run it through a synthesis tool like Vitis HLS, and wait — often tens of
          minutes for a small design, hours for a real one. Then you change one
          parameter and wait again.
        </p>
        <p>
          That loop is too slow to explore with. If you want to know how sixteen
          different configurations compare, you are looking at a day of waiting for
          numbers you could have reasoned about in advance.
        </p>
        <p>
          hls-estimate predicts those numbers from the shape of the network and a set of
          equations, so you can explore the space first and synthesize once, at the end,
          on a design you already believe in.
        </p>
      </Section>

      <Section title="The four numbers that matter">
        <p>
          An FPGA is a chip full of blank hardware you configure into whatever circuit
          you need. It comes in four currencies, and you can run out of any of them:
        </p>
        <Definitions
          items={[
            [
              "LUT",
              "Look-up tables — the general-purpose logic. Anything that is not arithmetic or memory ends up here.",
            ],
            [
              "FF",
              "Flip-flops — one-bit registers that hold values between clock ticks. Pipelining spends these.",
            ],
            [
              "DSP",
              "Dedicated multiplier blocks. Neural networks are almost entirely multiply-accumulate, so this is usually what you run out of first.",
            ],
            [
              "BRAM",
              "Block RAM — on-chip memory for weights and intermediate activations. Counted in 18-kilobit units.",
            ],
          ]}
        />
        <p>
          A Zynq-7020 — the chip on a PYNQ-Z2 board, a common starting point — has 220
          DSPs. That is the entire multiplier budget for your network.
        </p>
      </Section>

      <Section title="The one knob that matters most">
        <p>
          <span className="text-bone">Parallelism.</span> A convolution is millions of
          multiplications. You can build one multiplier and feed it every operation in
          turn — small and slow. You can build a thousand and do a thousand at once —
          fast and enormous. The <Mono>unroll</Mono> control on the estimator is exactly
          this dial.
        </p>
        <p>
          Everything else follows from it: double the lanes and you roughly halve the
          cycles and double the DSPs. The design space is the shape of that trade-off,
          and the <Link className="link" href="/">design space tab</Link> draws it.
        </p>
      </Section>

      <Section title="Why narrower numbers are cheaper">
        <p>
          Quantization means storing weights as small integers instead of 32-bit floats.
          An int8 weight needs one multiplier. Two int4 weights fit inside{" "}
          <em>one</em> multiplier at the same time, because a DSP block is wide enough to
          hold both with room to keep them from colliding.
        </p>
        <p>
          So halving the precision roughly halves the DSP bill. Switch the estimator to{" "}
          <Mono>int4</Mono> and watch the DSP column drop by half — that is not a
          decoration, it is the packing model doing its job.
        </p>
      </Section>

      <Section title="What the tool actually gives you">
        <Definitions
          items={[
            [
              "An estimate",
              "Per-layer LUT, FF, DSP, BRAM and cycle counts, plus a verdict on whether the whole thing fits your target board.",
            ],
            [
              "A bottleneck",
              "Stages run concurrently, so throughput is set by the slowest one. Optimising anything else changes nothing.",
            ],
            [
              "Synthesizable code",
              "Real Vitis HLS C++ with PIPELINE, UNROLL, ARRAY_PARTITION and DATAFLOW pragmas, verified bit-exact against the PyTorch model it came from.",
            ],
            [
              "A Pareto front",
              "Every configuration that fits, with the ones beaten on both speed and size removed.",
            ],
          ]}
        />
      </Section>

      <Section title="How much should you trust it?">
        <p>
          Less than a synthesis run, and the project says so in detail rather than
          rounding the error away. Measured against published hls4ml results, the DSP
          prediction runs about 27% high on fully-parallel designs and 113% high on a
          CNN whose DSPs had saturated. The LUT and flip-flop model was fitted to two
          data points and is 161% and 275% high on a network it was not fitted to. The
          BRAM model has never been checked against a published number at all.
        </p>
        <p>
          Use it to rank configurations and to catch a design that is off by 10× before
          you spend an afternoon on it. Do not quote its LUT count in a paper.
        </p>
        <p>
          <Link className="link" href="/model">
            Every assumption and every measured error band
          </Link>{" "}
          is written down.
        </p>
      </Section>

      <Section title="Where the correctness actually lives">
        <p>
          An estimate of wrong code is worthless, so the load-bearing test is not about
          resources at all: the generated C++ is compiled with an ordinary C++ compiler
          and its output is compared against the original PyTorch model, bit for bit, on
          random inputs. Same accumulation, same rounding, same clamping.
        </p>
        <p>
          That test is checked for teeth, too — deliberately breaking the rounding in the
          emitter makes five of its six cases fail, which is how you know it would catch
          a real bug rather than passing on principle.
        </p>
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

function Definitions({ items }: { items: [string, string][] }) {
  return (
    <dl className="my-1 flex flex-col gap-3 border-l border-edge pl-5">
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
