import type { Metadata } from "next";
import { ThemeProvider } from "@/components/theme/theme-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM From Scratch",
  description:
    "A comprehensive course on building large language models from scratch, covering data processing, model architecture, training, inference, and applications.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="font-mono antialiased">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
