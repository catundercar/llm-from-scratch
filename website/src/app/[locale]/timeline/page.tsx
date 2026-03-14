"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import { getPhases } from "@/data/phases";
import { PHASE_META, LAYERS } from "@/lib/constants";
import type { Locale } from "@/i18n";

function DifficultyDots({ difficulty }: { difficulty: string }) {
  const level =
    difficulty === "beginner" ? 1 : difficulty === "intermediate" ? 2 : 3;
  return (
    <div className="flex items-center gap-1" title={difficulty}>
      {[1, 2, 3].map((i) => (
        <span
          key={i}
          className={`inline-block h-2 w-2 rounded-full ${
            i <= level
              ? "bg-[var(--text-primary)]"
              : "bg-[var(--bg-tertiary)]"
          }`}
        />
      ))}
    </div>
  );
}

export default function TimelinePage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const phases = getPhases(locale as Locale);

  const maxLoc = Math.max(
    ...Object.values(PHASE_META).map((m) => m.loc),
    1
  );

  // Build a lookup for layer color by phase id
  const phaseLayerColor: Record<number, string> = {};
  for (const layer of LAYERS) {
    for (const pid of layer.phases) {
      phaseLayerColor[pid] = layer.color;
    }
  }

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
          Timeline
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mb-8">
          Phase progression with lines of code and difficulty levels.
        </p>

        {/* Timeline */}
        <div className="space-y-3">
          {phases.map((phase, idx) => {
            const meta = PHASE_META[phase.id];
            const layerColor = phaseLayerColor[phase.id] ?? phase.color;
            const barWidth = meta ? (meta.loc / maxLoc) * 100 : 0;

            return (
              <motion.div
                key={phase.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05, duration: 0.3 }}
              >
                <Link
                  href={`/${locale}/phase/${phase.id}`}
                  className="block"
                >
                  <div className="flex items-center gap-4 p-4 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] hover:border-[var(--border-hover)] transition-colors cursor-pointer">
                    {/* Phase icon and number */}
                    <div className="flex flex-col items-center shrink-0 w-12">
                      <div
                        className="flex h-10 w-10 items-center justify-center rounded-full text-lg"
                        style={{
                          backgroundColor: phase.color + "20",
                          color: phase.color,
                        }}
                      >
                        {phase.icon}
                      </div>
                      {/* Prerequisite dotted lines (visual indicator) */}
                      {meta && meta.prerequisites.length > 0 && (
                        <div
                          className="w-0.5 h-3 mt-1"
                          style={{
                            borderLeft: `2px dotted ${phase.color}40`,
                          }}
                        />
                      )}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-semibold text-[var(--text-primary)] truncate">
                          {phase.title}
                        </span>
                        {meta && (
                          <DifficultyDots difficulty={meta.difficulty} />
                        )}
                      </div>

                      {/* LOC bar */}
                      {meta && (
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-3 rounded-full bg-[var(--bg-tertiary)] overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ backgroundColor: layerColor }}
                              initial={{ width: 0 }}
                              animate={{ width: `${barWidth}%` }}
                              transition={{
                                delay: idx * 0.05 + 0.2,
                                duration: 0.5,
                                ease: "easeOut",
                              }}
                            />
                          </div>
                          <span className="text-xs font-mono text-[var(--text-tertiary)] shrink-0 w-16 text-right">
                            {meta.loc} LOC
                          </span>
                        </div>
                      )}

                      {/* Prerequisites */}
                      {meta &&
                        meta.prerequisites.length > 0 && (
                          <p className="text-xs text-[var(--text-tertiary)] mt-1">
                            Requires:{" "}
                            {meta.prerequisites
                              .map((p) => `Phase ${p}`)
                              .join(", ")}
                          </p>
                        )}
                    </div>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
