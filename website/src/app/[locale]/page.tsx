"use client";

import { useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  ArrowRight,
  BookOpen,
  Layers,
  Lightbulb,
  Code2,
} from "lucide-react";
import { getPhases } from "@/data/phases";
import { architecture, getPrinciples, dataFlowSteps } from "@/data/phases";
import { PHASE_META, LAYERS } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { Locale } from "@/i18n";

const TABS = [
  { id: "roadmap", label: "Roadmap", icon: BookOpen },
  { id: "architecture", label: "Architecture", icon: Layers },
  { id: "principles", label: "Principles", icon: Lightbulb },
] as const;

function DifficultyBadge({ difficulty }: { difficulty: string }) {
  const colors: Record<string, string> = {
    beginner: "#10B981",
    intermediate: "#F59E0B",
    advanced: "#EF4444",
  };
  return <Badge color={colors[difficulty] ?? "#71717A"}>{difficulty}</Badge>;
}

function PhaseCard({
  phase,
  locale,
}: {
  phase: ReturnType<typeof getPhases>[number];
  locale: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const meta = PHASE_META[phase.id];

  return (
    <div className="relative flex gap-4">
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-lg"
          style={{ backgroundColor: phase.color + "20", color: phase.color }}
        >
          {phase.icon}
        </div>
        <div
          className="w-0.5 flex-1 min-h-[24px]"
          style={{ backgroundColor: phase.color + "30" }}
        />
      </div>

      {/* Card content */}
      <div className="flex-1 pb-8">
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-left"
        >
          <Card className="hover:border-[var(--border-hover)] transition-colors cursor-pointer">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap mb-1">
                  <span
                    className="text-xs font-medium"
                    style={{ color: phase.color }}
                  >
                    {phase.week}
                  </span>
                  <span className="text-xs text-[var(--text-tertiary)]">
                    {phase.duration}
                  </span>
                </div>
                <h3 className="text-base font-semibold text-[var(--text-primary)]">
                  {phase.title}
                </h3>
                <p className="text-sm text-[var(--text-secondary)] mt-0.5">
                  {phase.subtitle}
                </p>

                {/* Badges row */}
                <div className="flex items-center gap-2 mt-2 flex-wrap">
                  {meta && (
                    <>
                      <Badge>
                        <Code2 size={10} className="mr-1" />
                        {meta.loc} LOC
                      </Badge>
                      <DifficultyBadge difficulty={meta.difficulty} />
                    </>
                  )}
                </div>
              </div>

              <motion.div
                animate={{ rotate: expanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronDown
                  size={18}
                  className="text-[var(--text-tertiary)] mt-1"
                />
              </motion.div>
            </div>

            {/* Key Insight quote */}
            {meta?.keyInsight && (
              <div
                className="mt-3 pl-3 border-l-2 text-xs italic text-[var(--text-secondary)]"
                style={{ borderColor: phase.color + "60" }}
              >
                {meta.keyInsight}
              </div>
            )}
          </Card>
        </button>

        {/* Expanded content */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="overflow-hidden"
            >
              <div className="mt-2 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4 space-y-4">
                {/* Goal */}
                <div>
                  <h4 className="text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-1">
                    Goal
                  </h4>
                  <p className="text-sm text-[var(--text-secondary)]">
                    {phase.goal}
                  </p>
                </div>

                {/* Concepts */}
                {phase.concepts.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-1.5">
                      Core Concepts
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {phase.concepts.map((c) => (
                        <Badge key={c} color={phase.color + "20"}>
                          <span style={{ color: phase.color }}>{c}</span>
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Readings */}
                {phase.readings.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-1">
                      References
                    </h4>
                    <ul className="text-sm text-[var(--text-secondary)] space-y-0.5">
                      {phase.readings.map((r, i) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-[var(--text-tertiary)]">-</span>
                          {r}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Deliverable */}
                <div>
                  <h4 className="text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-1">
                    Deliverable
                  </h4>
                  <p className="text-sm font-medium text-[var(--text-primary)]">
                    {phase.deliverable.name}
                  </p>
                  <p className="text-sm text-[var(--text-secondary)]">
                    {phase.deliverable.desc}
                  </p>
                </div>

                {/* Enter Lesson Button */}
                <Link
                  href={`/${locale}/phase/${phase.id}/lesson/1`}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium text-white transition-opacity hover:opacity-90"
                  style={{ backgroundColor: phase.color }}
                >
                  Enter Lesson
                  <ArrowRight size={14} />
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function RoadmapTab({ locale }: { locale: string }) {
  const phases = getPhases(locale as Locale);

  return (
    <div className="max-w-2xl mx-auto">
      {phases.map((phase) => (
        <PhaseCard key={phase.id} phase={phase} locale={locale} />
      ))}
    </div>
  );
}

function ArchitectureTab() {
  const arch = architecture;

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Architecture layers */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          Model Architecture
        </h3>
        {arch.layers.map((layer, i) => (
          <div
            key={i}
            className="rounded-lg border border-[var(--border)] overflow-hidden"
          >
            <div
              className="px-4 py-2 text-sm font-semibold text-white"
              style={{ backgroundColor: layer.color }}
            >
              {layer.name}
            </div>
            <div className="px-4 py-3 bg-[var(--bg-secondary)] flex flex-wrap gap-2">
              {layer.modules.map((mod) => (
                <span
                  key={mod}
                  className="inline-flex items-center rounded-md px-2.5 py-1 text-xs font-mono border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-secondary)]"
                >
                  {mod}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Data flow pipeline */}
      <div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-4">
          Data Flow Pipeline
        </h3>
        <div className="flex items-center gap-1 flex-wrap">
          {dataFlowSteps.map((step, i) => (
            <div key={i} className="flex items-center gap-1">
              <span className="px-3 py-1.5 rounded-md text-xs font-mono bg-[var(--bg-tertiary)] text-[var(--text-primary)] border border-[var(--border)]">
                {step}
              </span>
              {i < dataFlowSteps.length - 1 && (
                <ArrowRight
                  size={14}
                  className="text-[var(--text-tertiary)] shrink-0"
                />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PrinciplesTab({ locale }: { locale: string }) {
  const principles = getPrinciples(locale as Locale);

  return (
    <div className="max-w-3xl mx-auto grid grid-cols-1 sm:grid-cols-2 gap-4">
      {principles.map((p) => (
        <Card key={p.num} className="relative overflow-hidden">
          <div
            className="absolute top-0 left-0 w-1 h-full"
            style={{ backgroundColor: p.color }}
          />
          <div className="pl-4">
            <span
              className="text-3xl font-bold opacity-20"
              style={{ color: p.color }}
            >
              {p.num}
            </span>
            <h3 className="text-base font-semibold text-[var(--text-primary)] -mt-1">
              {p.title}
            </h3>
            <p className="text-sm text-[var(--text-secondary)] mt-1">
              {p.desc}
            </p>
          </div>
        </Card>
      ))}
    </div>
  );
}

export default function HomePage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const [activeTab, setActiveTab] = useState<string>("roadmap");

  return (
    <div>
      <div className="py-4">
        {/* Header */}
        <div className="text-center mb-10">
          <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold tracking-wider uppercase bg-[var(--bg-tertiary)] text-[var(--text-secondary)] mb-4">
            Practicum Course
          </span>
          <h1 className="text-3xl sm:text-4xl font-bold text-[var(--text-primary)] mb-3">
            LLM From Scratch
          </h1>
          <p className="text-base text-[var(--text-secondary)] max-w-lg mx-auto">
            Build a GPT-class language model from scratch with PyTorch.
            Understand every layer, every gradient, every token.
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)] p-1">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? "bg-[var(--bg-primary)] text-[var(--text-primary)] shadow-sm"
                      : "text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
                  }`}
                >
                  <Icon size={14} />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            {activeTab === "roadmap" && <RoadmapTab locale={locale} />}
            {activeTab === "architecture" && <ArchitectureTab />}
            {activeTab === "principles" && <PrinciplesTab locale={locale} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
