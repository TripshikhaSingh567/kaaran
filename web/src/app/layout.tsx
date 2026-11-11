import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "KAARAN - Uncertainty-Aware Causal World Model",
  description: "Continuous industrial heated tank simulator and causal world model for hackathon presentation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
        {children}
      </body>
    </html>
  );
}
