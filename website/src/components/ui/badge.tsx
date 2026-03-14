interface BadgeProps {
  children: React.ReactNode;
  color?: string;
  className?: string;
}

export function Badge({ children, color, className }: BadgeProps) {
  const colorClasses = color
    ? ""
    : "bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300";

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${colorClasses} ${className ?? ""}`}
      style={{
        fontFamily: "var(--font-geist-mono), monospace",
        ...(color ? { backgroundColor: color, color: "#fff" } : {}),
      }}
    >
      {children}
    </span>
  );
}
