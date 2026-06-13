import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgroVision AI",
  description: "Edge-First Plant Disease Intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">
        {/* Simple Sidebar structure will go here later */}
        <div className="flex h-screen overflow-hidden">
          <aside className="w-64 bg-slate-900 text-white p-4 hidden md:block">
            <h1 className="text-xl font-bold text-green-400 mb-8">AgroVision AI</h1>
            <nav className="space-y-2">
              <a href="/" className="block p-2 rounded hover:bg-slate-800">Dashboard</a>
              <a href="/diagnose" className="block p-2 rounded hover:bg-slate-800 text-green-400">Diagnosis Center</a>
            </nav>
          </aside>
          <main className="flex-1 overflow-y-auto bg-slate-50">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
