export function generateStaticParams() {
  return Array.from({ length: 10 }, (_, i) => ({ phaseId: String(i) }));
}

export default function PhaseLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
