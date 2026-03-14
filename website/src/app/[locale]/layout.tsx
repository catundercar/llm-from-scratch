import { Header } from "@/components/layout/header";
import { Sidebar } from "@/components/layout/sidebar";
import { Footer } from "@/components/layout/footer";
import { LocaleProvider } from "@/i18n";

export default function LocaleLayout({
  children,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  return (
    <LocaleProvider>
      <div className="min-h-screen flex flex-col bg-[var(--bg-primary)]">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-y-auto">
            <div className="max-w-4xl mx-auto px-4 py-8">{children}</div>
          </main>
        </div>
        <Footer />
      </div>
    </LocaleProvider>
  );
}
