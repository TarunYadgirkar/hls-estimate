"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

/**
 * Minimal C++ highlighter. A full grammar is not worth the bundle here — the
 * emitted code only ever contains these token classes.
 */
const PRAGMA = /^\s*#pragma\s+HLS.*$/;
const PREPROC = /^\s*#(include|define|ifdef|endif).*$/;
const COMMENT = /(\/\/.*$|\/\*.*?\*\/)/;
const KEYWORD =
  /\b(static|const|void|int|for|if|continue|return|typedef|long|struct)\b/g;
const TYPE = /\b(data_t|acc_t|int8_t|int32_t|int64_t)\b/g;
const NUMBER = /\b(\d+)\b/g;

function highlight(line: string): React.ReactNode {
  if (PRAGMA.test(line)) {
    return <span className="text-signal">{line}</span>;
  }
  if (PREPROC.test(line)) {
    return <span className="text-trace">{line}</span>;
  }
  const comment = line.match(COMMENT);
  if (comment && comment.index !== undefined) {
    return (
      <>
        {highlight(line.slice(0, comment.index))}
        <span className="text-muted-dim">{line.slice(comment.index)}</span>
      </>
    );
  }
  const parts: React.ReactNode[] = [];
  let last = 0;
  const marks: { start: number; end: number; cls: string }[] = [];
  for (const [re, cls] of [
    [KEYWORD, "text-[#c39bf5]"],
    [TYPE, "text-[#7fd1e8]"],
    [NUMBER, "text-[#e8c07d]"],
  ] as const) {
    re.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(line))) {
      marks.push({ start: m.index, end: m.index + m[0].length, cls });
    }
  }
  marks.sort((a, b) => a.start - b.start);
  for (const mark of marks) {
    if (mark.start < last) continue;
    parts.push(line.slice(last, mark.start));
    parts.push(
      <span key={`${mark.start}-${mark.cls}`} className={mark.cls}>
        {line.slice(mark.start, mark.end)}
      </span>,
    );
    last = mark.end;
  }
  parts.push(line.slice(last));
  return <>{parts}</>;
}

export function CodePanel({ source }: { source: string }) {
  const [copied, setCopied] = useState(false);
  const lines = source.split("\n");

  async function copy() {
    await navigator.clipboard.writeText(source);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-xl text-[13px] leading-relaxed text-muted">
          This is the real emitter&apos;s output — the pragmas below change as you move
          the parallelism control. Weight literals are elided for reading; the CLI
          writes them in full.
        </p>
        <button
          type="button"
          onClick={copy}
          className="flex shrink-0 items-center gap-1.5 rounded border border-edge bg-substrate-3 px-2.5 py-1.5 font-mono text-xs text-muted transition-colors hover:border-edge-bright hover:text-bone"
        >
          {copied ? <Check size={12} className="text-signal" /> : <Copy size={12} />}
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <div className="max-h-[26rem] overflow-auto rounded border border-edge bg-[#101219]">
        <pre className="p-4 font-mono text-[11.5px] leading-[1.65]">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="flex">
                <span
                  aria-hidden
                  className="tabular mr-4 w-7 shrink-0 select-none text-right text-muted-dim/50"
                >
                  {i + 1}
                </span>
                <span className="whitespace-pre text-bone/90">{highlight(line)}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
