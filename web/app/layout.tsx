import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const SITE_URL = "https://hls-estimate.vercel.app";
const SITE_TITLE = "hls-estimate — will it fit?";
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
  { href: "/", label: "estimator" },
  { href: "/what-is-this", label: "what is this" },
  { href: "/model", label: "the model" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="min-h-screen antialiased">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
        />
        <header className="sticky top-0 z-50 border-b border-edge bg-substrate/85 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1400px] items-center gap-3 px-5 py-3 sm:gap-6 sm:px-8">
            <Link
              href="/"
              className="shrink-0 whitespace-nowrap font-mono text-sm font-semibold tracking-tight text-bone"
            >
              hls<span className="text-trace">-</span>estimate
            </Link>
            <nav className="flex min-w-0 items-center gap-1 overflow-x-auto">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="whitespace-nowrap rounded px-2.5 py-1 font-mono text-xs text-muted transition-colors hover:bg-substrate-3 hover:text-bone"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <a
              href="https://github.com/TarunYadgirkar/hls-estimate"
              target="_blank"
              rel="noreferrer noopener"
              className="ml-auto whitespace-nowrap font-mono text-xs text-muted-dim transition-colors hover:text-bone"
            >
              source ↗
            </a>
          </div>
        </header>
        {children}
        <footer className="mt-24 border-t border-edge">
          <div className="mx-auto flex max-w-[1400px] flex-col gap-2 px-5 py-8 font-mono text-xs text-muted-dim sm:flex-row sm:items-center sm:justify-between sm:px-8">
            <p>
              Analytical estimates. No synthesis was run — and the numbers are wrong
              in ways that are{" "}
              <Link
                href="/model"
                className="text-muted underline underline-offset-4 hover:text-bone"
              >
                written down
              </Link>
              .
            </p>
            <p>Zynq-7020 · Ultra96 · PYNQ-Z2</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
