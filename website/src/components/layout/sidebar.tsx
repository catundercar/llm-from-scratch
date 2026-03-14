"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  Database,
  Boxes,
  Layers,
  Dumbbell,
  Zap,
  MessageSquare,
  Globe,
  Shield,
  Sparkles,
} from "lucide-react";

const phaseIcons: Record<number, React.ElementType> = {
  0: BookOpen,
  1: Database,
  2: Boxes,
  3: Layers,
  4: Dumbbell,
  5: Zap,
  6: MessageSquare,
  7: Globe,
  8: Shield,
  9: Sparkles,
};

const layerGroups = [
  {
    label: "Overview",
    phases: [{ id: 0, title: "Foundation" }],
  },
  {
    label: "Data Processing",
    phases: [{ id: 1, title: "Data Pipeline" }],
  },
  {
    label: "Model Architecture",
    phases: [
      { id: 2, title: "Embeddings" },
      { id: 3, title: "Attention" },
    ],
  },
  {
    label: "Training",
    phases: [{ id: 4, title: "Training Loop" }],
  },
  {
    label: "Inference",
    phases: [{ id: 5, title: "Generation" }],
  },
  {
    label: "Applications",
    phases: [
      { id: 6, title: "Chat & Dialog" },
      { id: 7, title: "Multimodal" },
      { id: 8, title: "Safety" },
      { id: 9, title: "Advanced" },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden md:block w-64 shrink-0 border-r border-[var(--border)] bg-[var(--bg-secondary)] overflow-y-auto">
      <div className="p-4 flex flex-col gap-6">
        {layerGroups.map((group) => (
          <div key={group.label}>
            <h3 className="text-[10px] uppercase tracking-widest text-[var(--text-tertiary)] font-semibold mb-2 px-2">
              {group.label}
            </h3>
            <div className="flex flex-col gap-0.5">
              {group.phases.map((phase) => {
                const Icon = phaseIcons[phase.id];
                const href = `/phase/${phase.id}`;
                const isActive = pathname.startsWith(href);
                const phaseColor = `var(--phase-${phase.id})`;

                return (
                  <Link
                    key={phase.id}
                    href={href}
                    className={`flex items-center gap-2.5 px-2 py-1.5 rounded-md text-xs transition-colors ${
                      isActive
                        ? "bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
                        : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: phaseColor }}
                    />
                    <Icon size={14} className="shrink-0" />
                    <span className="truncate">{phase.title}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
