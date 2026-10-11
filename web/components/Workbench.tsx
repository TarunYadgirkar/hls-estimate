"use client";

import { useRef } from "react";

export const TABS = ["layers", "generated C++", "design space"] as const;
export type Tab = (typeof TABS)[number];

const TAB_LABEL: Record<Tab, string> = {
  layers: "Layers",
  "generated C++": "Generated C++",
  "design space": "Design space",
};

const tabId = (t: Tab) => `tab-${t.replace(/\W+/g, "-")}`;

export function Workbench({
  tab,
  onTab,
  children,
}: {
  tab: Tab;
  onTab: (t: Tab) => void;
  children: React.ReactNode;
}) {
  const refs = useRef<Partial<Record<Tab, HTMLButtonElement | null>>>({});

  function onKeyDown(e: React.KeyboardEvent) {
    const current = TABS.indexOf(tab);
    const target: Record<string, number> = {
      ArrowRight: (current + 1) % TABS.length,
      ArrowLeft: (current - 1 + TABS.length) % TABS.length,
      Home: 0,
      End: TABS.length - 1,
    };
    if (!(e.key in target)) return;
    e.preventDefault();
    const next = TABS[target[e.key]];
    onTab(next);
    refs.current[next]?.focus();
  }

  return (
    <section className="mt-14">
      <div
        role="tablist"
        aria-label="Details"
        onKeyDown={onKeyDown}
        className="flex gap-1 overflow-x-auto shadow-[inset_0_-1px_0_var(--edge)]"
      >
        {TABS.map((t) => {
          const selected = tab === t;
          return (
            <button
              key={t}
              ref={(el) => {
                refs.current[t] = el;
              }}
              id={tabId(t)}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls="workbench-panel"
              tabIndex={selected ? 0 : -1}
              onClick={() => onTab(t)}
              className={`whitespace-nowrap px-3 py-2.5 text-[13px] font-medium transition-colors ${
                selected
                  ? "text-bone shadow-[inset_0_-2px_0_var(--bone)]"
                  : "text-muted hover:text-bone"
              }`}
            >
              {TAB_LABEL[t]}
            </button>
          );
        })}
      </div>
      <div
        id="workbench-panel"
        role="tabpanel"
        tabIndex={0}
        aria-labelledby={tabId(tab)}
        className="pt-7"
      >
        {children}
      </div>
    </section>
  );
}
