"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { getPhases } from "@/data/phases";
import { PHASE_META, LAYERS } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import type { Locale } from "@/i18n";

function DifficultyBadge({ difficulty }: { difficulty: string }) {
  const colors: Record<string, string> = {
    beginner: "#10B981",
    intermediate: "#F59E0B",
    advanced: "#EF4444",
  };
  return <Badge color={colors[difficulty] ?? "#71717A"}>{difficulty}</Badge>;
}

export default function LayersPage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const phases = getPhases(locale as Locale);

  // Build a quick lookup for phases by id
  const phaseById = new Map(phases.map((p) => [p.id, p]));

  return (
    <div>
      <div className="py-2">
        {/* Header */}
        <Link
          href={`/${locale}`}
          className="inline-flex items-center gap-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors mb-6"
        >
          <ArrowLeft size={14} />
          Back to Roadmap
        </Link>

        <h1 className="text-2xl font-bold text-[var(--text-primary)] mb-2">
          Architecture Layers
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mb-8">
          Phases grouped by architectural layer. Each layer builds on the
          previous to form a complete language model.
        </p>

        {/* Layers */}
        <div className="space-y-8">
          {LAYERS.map((layer, li) => (
            <motion.div
              key={layer.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: li * 0.1, duration: 0.3 }}
            >
              {/* Layer header bar */}
              <div
                className="rounded-t-xl px-4 py-2.5 text-sm font-semibold text-white"
                style={{ backgroundColor: layer.color }}
              >
                {layer.label}
              </div>

              {/* Phase cards within this layer */}
              <div className="border border-t-0 border-[var(--border)] rounded-b-xl divide-y divide-[var(--border)]">
                {layer.phases.map((pid) => {
                  const phase = phaseById.get(pid);
                  const meta = PHASE_META[pid];
                  if (!phase) return null;

                  return (
                    <Link
                      key={pid}
                      href={`/${locale}/phase/${pid}`}
                      className="flex items-center justify-between gap-4 px-4 py-3 hover:bg-[var(--bg-secondary)] transition-colors cursor-pointer block"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div
                          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-base"
                          style={{
                            backgroundColor: phase.color + "20",
                            color: phase.color,
                          }}
                        >
                          {phase.icon}
                        </div>
                        <div className="min-w-0">
                          <h3 className="text-sm font-semibold text-[var(--text-primary)] truncate">
                            {phase.title}
                          </h3>
                          <div className="flex items-center gap-2 mt-0.5">
                            {meta && (
                              <>
                                <span className="text-xs font-mono text-[var(--text-tertiary)]">
                                  {meta.loc} LOC
                                </span>
                                <DifficultyBadge
                                  difficulty={meta.difficulty}
                                />
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                      <ArrowRight
                        size={14}
                        className="text-[var(--text-tertiary)] shrink-0"
                      />
                    </Link>
                  );
                })}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
