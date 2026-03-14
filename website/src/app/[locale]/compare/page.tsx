"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowLeftRight, Code2 } from "lucide-react";
import { getPhases } from "@/data/phases";
import { PHASE_META } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { Locale } from "@/i18n";
import type { Phase } from "@/data/types";

function DifficultyBadge({ difficulty }: { difficulty: string }) {
  const colors: Record<string, string> = {
    beginner: "#10B981",
    intermediate: "#F59E0B",
    advanced: "#EF4444",
  };
  return <Badge color={colors[difficulty] ?? "#71717A"}>{difficulty}</Badge>;
}

function PhaseSelector({
  phases,
  value,
  onChange,
  label,
}: {
  phases: Phase[];
  value: number | null;
  onChange: (id: number) => void;
  label: string;
}) {
  return (
    <div>
      <label className="block text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-2">
        {label}
      </label>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-primary)] px-3 py-2 text-sm focus:outline-none focus:border-[var(--border-hover)]"
      >
        <option value="" disabled>
          Select a phase...
        </option>
        {phases.map((p) => (
          <option key={p.id} value={p.id}>
            Phase {p.id}: {p.title}
          </option>
        ))}
      </select>
    </div>
  );
}

function PhaseDetail({ phase }: { phase: Phase }) {
  const meta = PHASE_META[phase.id];

  return (
    <Card>
      <div className="flex items-center gap-3 mb-4">
        <div
          className="flex h-10 w-10 items-center justify-center rounded-lg text-xl"
          style={{ backgroundColor: phase.color + "20", color: phase.color }}
        >
          {phase.icon}
        </div>
        <div>
          <h3 className="text-base font-semibold text-[var(--text-primary)]">
            {phase.title}
          </h3>
          <p className="text-xs text-[var(--text-tertiary)]">
            {phase.week} &middot; {phase.duration}
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="space-y-3">
        {meta && (
          <div className="flex items-center gap-2 flex-wrap">
            <Badge>
              <Code2 size={10} className="mr-1" />
              {meta.loc} LOC
            </Badge>
            <DifficultyBadge difficulty={meta.difficulty} />
            {meta.layer && <Badge>{meta.layer}</Badge>}
          </div>
        )}

        {/* Key Insight */}
        {meta?.keyInsight && (
          <div
            className="pl-3 border-l-2 text-xs italic text-[var(--text-secondary)]"
            style={{ borderColor: phase.color + "60" }}
          >
            {meta.keyInsight}
          </div>
        )}

        {/* Core Addition */}
        {meta?.coreAddition && (
          <div>
            <span className="text-xs font-semibold uppercase text-[var(--text-tertiary)]">
              Core Addition
            </span>
            <p className="text-sm text-[var(--text-primary)] font-mono mt-0.5">
              {meta.coreAddition}
            </p>
          </div>
        )}

        {/* Concepts */}
        <div>
          <span className="text-xs font-semibold uppercase text-[var(--text-tertiary)]">
            Concepts
          </span>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {phase.concepts.map((c) => (
              <span
                key={c}
                className="inline-flex items-center rounded-md px-2 py-0.5 text-xs font-mono border border-[var(--border)] text-[var(--text-secondary)]"
              >
                {c}
              </span>
            ))}
          </div>
        </div>

        {/* Prerequisites */}
        {meta && meta.prerequisites.length > 0 && (
          <div>
            <span className="text-xs font-semibold uppercase text-[var(--text-tertiary)]">
              Prerequisites
            </span>
            <p className="text-sm text-[var(--text-secondary)] mt-0.5">
              {meta.prerequisites.map((p) => `Phase ${p}`).join(", ")}
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}

export default function ComparePage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const phases = getPhases(locale as Locale);

  const [phaseA, setPhaseA] = useState<number | null>(null);
  const [phaseB, setPhaseB] = useState<number | null>(null);

  const selectedA = phases.find((p) => p.id === phaseA) ?? null;
  const selectedB = phases.find((p) => p.id === phaseB) ?? null;

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

        <div className="flex items-center gap-3 mb-2">
          <ArrowLeftRight
            size={24}
            className="text-[var(--text-tertiary)]"
          />
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">
            Compare Phases
          </h1>
        </div>
        <p className="text-sm text-[var(--text-secondary)] mb-8">
          Select two phases to compare their concepts, complexity, and key
          components side by side.
        </p>

        {/* Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <PhaseSelector
            phases={phases}
            value={phaseA}
            onChange={setPhaseA}
            label="Phase A"
          />
          <PhaseSelector
            phases={phases}
            value={phaseB}
            onChange={setPhaseB}
            label="Phase B"
          />
        </div>

        {/* Comparison */}
        {selectedA && selectedB ? (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            <PhaseDetail phase={selectedA} />
            <PhaseDetail phase={selectedB} />
          </motion.div>
        ) : (
          <div className="text-center py-16">
            <ArrowLeftRight
              size={40}
              className="mx-auto mb-4 text-[var(--text-tertiary)] opacity-30"
            />
            <p className="text-sm text-[var(--text-tertiary)]">
              Select two phases above to see a side-by-side comparison.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
