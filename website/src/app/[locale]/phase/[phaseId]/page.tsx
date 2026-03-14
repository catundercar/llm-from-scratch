"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, ArrowRight, BookOpen, Code2 } from "lucide-react";
import { getPhases } from "@/data/phases";
import { getPhaseContent } from "@/data/index";
import { PHASE_META } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { Locale } from "@/i18n";

function DifficultyBadge({ difficulty }: { difficulty: string }) {
  const colors: Record<string, string> = {
    beginner: "#10B981",
    intermediate: "#F59E0B",
    advanced: "#EF4444",
  };
  return <Badge color={colors[difficulty] ?? "#71717A"}>{difficulty}</Badge>;
}

export default function PhaseOverviewPage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const phaseId = Number(params?.phaseId);

  const phases = getPhases(locale as Locale);
  const phase = phases.find((p) => p.id === phaseId);
  const meta = PHASE_META[phaseId];

  // Attempt to load phase content for lesson listing
  let phaseContent: ReturnType<typeof getPhaseContent> | null = null;
  try {
    phaseContent = getPhaseContent(phaseId, locale as Locale);
  } catch {
    // Phase content may not exist yet
  }

  if (!phase) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-[var(--text-primary)] mb-2">
            Phase Not Found
          </h1>
          <p className="text-[var(--text-secondary)] mb-4">
            Phase {phaseId} does not exist.
          </p>
          <Link
            href={`/${locale}`}
            className="text-sm text-blue-500 hover:underline"
          >
            Back to Roadmap
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="py-2">
        {/* Back link */}
        <Link
          href={`/${locale}`}
          className="inline-flex items-center gap-1.5 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors mb-6"
        >
          <ArrowLeft size={14} />
          Back to Roadmap
        </Link>

        {/* Phase header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <div
              className="flex h-12 w-12 items-center justify-center rounded-xl text-2xl"
              style={{
                backgroundColor: phase.color + "20",
                color: phase.color,
              }}
            >
              {phase.icon}
            </div>
            <div>
              <span
                className="text-xs font-medium"
                style={{ color: phase.color }}
              >
                {phase.week} &middot; {phase.duration}
              </span>
              <h1 className="text-2xl font-bold text-[var(--text-primary)]">
                {phase.title}
              </h1>
            </div>
          </div>
          <p className="text-sm text-[var(--text-secondary)]">
            {phase.subtitle}
          </p>

          {/* Meta badges */}
          <div className="flex items-center gap-2 mt-4 flex-wrap">
            {meta && (
              <>
                <Badge>
                  <Code2 size={10} className="mr-1" />
                  {meta.loc} LOC
                </Badge>
                <DifficultyBadge difficulty={meta.difficulty} />
                {meta.prerequisites.length > 0 && (
                  <Badge>
                    Prerequisites:{" "}
                    {meta.prerequisites
                      .map((p) => `Phase ${p}`)
                      .join(", ")}
                  </Badge>
                )}
              </>
            )}
          </div>
        </div>

        {/* Goal */}
        <Card className="mb-6">
          <h2 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-2">
            Goal
          </h2>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
            {phase.goal}
          </p>
        </Card>

        {/* Lessons list */}
        {phaseContent && phaseContent.lessons.length > 0 && (
          <div className="mb-6">
            <h2 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
              Lessons
            </h2>
            <div className="space-y-2">
              {phaseContent.lessons.map((lesson) => (
                <Link
                  key={lesson.lessonId}
                  href={`/${locale}/phase/${phaseId}/lesson/${lesson.lessonId}`}
                  className="block"
                >
                  <Card className="hover:border-[var(--border-hover)] transition-colors cursor-pointer">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className="flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold text-white"
                          style={{ backgroundColor: phase.color }}
                        >
                          {lesson.lessonId}
                        </div>
                        <div>
                          <h3 className="text-sm font-medium text-[var(--text-primary)]">
                            {lesson.title}
                          </h3>
                          <p className="text-xs text-[var(--text-tertiary)]">
                            {lesson.type} &middot; {lesson.duration}
                          </p>
                        </div>
                      </div>
                      <ArrowRight
                        size={14}
                        className="text-[var(--text-tertiary)]"
                      />
                    </div>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* Core Concepts */}
        {phase.concepts.length > 0 && (
          <div className="mb-6">
            <h2 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
              Core Concepts
            </h2>
            <div className="flex flex-wrap gap-2">
              {phase.concepts.map((c) => (
                <span
                  key={c}
                  className="inline-flex items-center rounded-lg px-3 py-1.5 text-sm font-mono border border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-primary)]"
                >
                  {c}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* References */}
        {phase.readings.length > 0 && (
          <div className="mb-6">
            <h2 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
              References
            </h2>
            <Card>
              <ul className="space-y-1.5">
                {phase.readings.map((r, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-[var(--text-secondary)]"
                  >
                    <BookOpen
                      size={14}
                      className="mt-0.5 shrink-0 text-[var(--text-tertiary)]"
                    />
                    {r}
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )}

        {/* Deliverable */}
        <div className="mb-6">
          <h2 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
            Deliverable
          </h2>
          <Card>
            <h3
              className="text-base font-semibold text-[var(--text-primary)] mb-1"
              style={{ color: phase.color }}
            >
              {phase.deliverable.name}
            </h3>
            <p className="text-sm text-[var(--text-secondary)] mb-3">
              {phase.deliverable.desc}
            </p>
            <h4 className="text-xs font-semibold uppercase text-[var(--text-tertiary)] mb-2">
              Acceptance Criteria
            </h4>
            <ul className="space-y-1">
              {phase.deliverable.acceptance.map((a, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-sm text-[var(--text-secondary)]"
                >
                  <span style={{ color: phase.color }}>&#10003;</span>
                  {a}
                </li>
              ))}
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}
