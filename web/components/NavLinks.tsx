"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function NavLinks({ items }: { items: { href: string; label: string }[] }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Site" className="flex min-w-0 items-center gap-1 overflow-x-auto">
      {items.map((item) => {
        const current = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={current ? "page" : undefined}
            className={`whitespace-nowrap rounded px-2 py-1.5 sm:px-2.5 text-[13px] transition-colors hover:bg-substrate-3 hover:text-bone ${
              current ? "bg-substrate-2 text-bone shadow-[inset_0_0_0_1px_var(--edge)]" : "text-muted"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
