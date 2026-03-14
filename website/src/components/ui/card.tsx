interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export function Card({ children, className }: CardProps) {
  return (
    <div
      className={`rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4 ${className ?? ""}`}
    >
      {children}
    </div>
  );
}
