export function compact(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`;
  if (Math.abs(n) >= 10_000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`;
  return n.toLocaleString("en-US");
}

export function pct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  if (x > 0 && x < 0.001) return "<0.1%";
  return `${(x * 100).toFixed(x >= 1 ? 0 : 1)}%`;
}

/** 100 MHz is the conservative default clock for the small parts we target. */
export const CLOCK_MHZ = 100;

export function cyclesToTime(cycles: number): string {
  const us = cycles / CLOCK_MHZ;
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  if (us >= 1) return `${us.toFixed(1)} µs`;
  return `${(us * 1000).toFixed(0)} ns`;
}
