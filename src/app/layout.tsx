import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CUDA Program Visualizer",
  description: "Simulate CUDA kernel behavior and visualize memory and execution timelines.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-neutral-100">
        <div className="mx-auto max-w-6xl px-4 py-8">
          {children}
        </div>
      </body>
    </html>
  );
}
