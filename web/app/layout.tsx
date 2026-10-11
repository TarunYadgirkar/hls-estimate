import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, IBM_Plex_Sans_Condensed } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { NavLinks } from "@/components/NavLinks";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const plexCondensed = IBM_Plex_Sans_Condensed({
  variable: "--font-plex-condensed",
  subsets: ["latin"],
  weight: ["600"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const SITE_URL = "https://hls-estimate.vercel.app";
const SITE_TITLE = "hls-estimate: will it fit?";
const SITE_DESCRIPTION =
  "Predict FPGA resource usage and latency for a quantized neural network, and generate synthesizable HLS C++, without running synthesis.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: {
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    url: "/",
    siteName: "hls-estimate",
    type: "website",
  },
  twitter: { card: "summary" },
};

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebApplication",
  name: "hls-estimate",
  description: SITE_DESCRIPTION,
  url: SITE_URL,
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Any",
};

const NAV = [
  { href: "/", label: "Estimator" },
  { href: "/what-is-this", label: "About" },
  { href: "/model", label: "The model" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable} ${plexCondensed.variable}`}>
      <body className="min-h-screen antialiased">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
        />
        <header className="sticky top-0 z-50 bg-substrate/90 shadow-[0_1px_0_var(--edge)] backdrop-blur-md">
          <div className="mx-auto flex max-w-[1400px] items-center gap-3 px-4 py-3 sm:gap-6 sm:px-8">
            <Link
              href="/"
              className="shrink-0 whitespace-nowrap font-mono text-sm font-semibold tracking-tight text-bone"
            >
              hls-estimate
            </Link>
            <NavLinks items={NAV} />
            <a
              href="https://github.com/TarunYadgirkar/hls-estimate"
              target="_blank"
              rel="noreferrer noopener"
              className="ml-auto hidden whitespace-nowrap rounded px-2 py-1 text-[13px] text-muted transition-colors hover:text-bone sm:block"
            >
              Source on GitHub
            </a>
          </div>
        </header>
        {children}
        <footer className="mt-24 shadow-[0_-1px_0_var(--edge)]">
          <div className="mx-auto flex max-w-[1400px] flex-col gap-2 px-4 py-8 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between sm:px-8">
            <p>
              Analytical estimates, no synthesis run. Where they are wrong, and by how
              much, is{" "}
              <Link href="/model" className="link">
                written down
              </Link>
              .
            </p>
            <a
              href="https://github.com/TarunYadgirkar/hls-estimate"
              target="_blank"
              rel="noreferrer noopener"
              className="link"
            >
              Source on GitHub
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
