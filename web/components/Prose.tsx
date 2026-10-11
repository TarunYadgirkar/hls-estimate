import Link from "next/link";

export function PageTitle({ children }: { children: React.ReactNode }) {
  return (
    <h1 className="font-display text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.01em] sm:text-[3.25rem]">
      {children}
    </h1>
  );
}

export function Lede({ children }: { children: React.ReactNode }) {
  return <p className="mt-6 max-w-[65ch] text-[17px] leading-relaxed text-muted">{children}</p>;
}

export function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-14">
      <h2 className="text-xl font-semibold tracking-tight text-bone">{title}</h2>
      <div className="mt-4 flex flex-col gap-4 text-[15px] leading-relaxed text-muted [&>p]:max-w-[65ch] [&_em]:not-italic [&_em]:text-bone">
        {children}
      </div>
    </section>
  );
}

export function DefList({ items }: { items: [string, string][] }) {
  return (
    <dl className="my-1 flex flex-col gap-4 pl-5 shadow-[inset_1px_0_0_var(--edge-bright)]">
      {items.map(([term, body]) => (
        <div key={term}>
          <dt className="text-[14px] font-medium text-bone">{term}</dt>
          <dd className="mt-1 text-[14px] leading-relaxed text-muted">{body}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-[3px] bg-substrate-3 px-1.5 py-0.5 font-mono text-[13px] text-bone">
      {children}
    </code>
  );
}

export function BackToEstimator() {
  return (
    <div className="mt-16 pt-8 shadow-[0_-1px_0_var(--edge)]">
      <Link className="link text-sm" href="/">
        Back to the estimator
      </Link>
    </div>
  );
}
