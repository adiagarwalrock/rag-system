import type { Metadata } from "next";
import { AppShell } from "@/components/shell/app-shell";
import { Providers } from "@/components/providers/query-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Console",
  description: "Internal operations console for document-grounded RAG.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
