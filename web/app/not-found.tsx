import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Page not found — hls-estimate",
};

export default function NotFound() {
  return (
    <main className="mx-auto max-w-[760px] px-5 pb-4 pt-24 sm:px-8">
      <p className="eyebrow">404</p>
      <h1 className="mt-3 font-mono text-[2.2rem] font-semibold leading-[1.1] tracking-tight sm:text-[2.8rem]">
        This page does not fit.
      </h1>
      <p className="mt-6 text-[17px] leading-relaxed text-muted">
        Nothing is routed at this address. The estimator, the model notes, and the
        orientation page are all reachable from the header.
      </p>
      <p className="mt-8">
        <Link
          href="/"
          className="font-mono text-sm text-trace underline underline-offset-4 hover:text-bone"
        >
          back to the estimator
        </Link>
      </p>
    </main>
  );
}
