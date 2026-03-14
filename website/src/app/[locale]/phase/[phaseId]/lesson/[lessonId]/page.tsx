"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  BookOpen,
  Eye,
  Code2,
  Search,
  ExternalLink,
} from "lucide-react";
import { getPhases } from "@/data/phases";
import { getPhaseContent, getLesson } from "@/data/index";
import { PHASE_META } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  ContentRenderer,
  SectionRenderer,
} from "@/components/content/content-renderer";
import { PhaseVisualization } from "@/components/visualizations";
import type { Locale } from "@/i18n";
import type { Lesson, ContentSection } from "@/data/types";

const LESSON_TABS = [
  { id: "learn", label: "Learn", icon: BookOpen },
  { id: "visualize", label: "Visualize", icon: Eye },
  { id: "code", label: "Code", icon: Code2 },
  { id: "deep-dive", label: "Deep Dive", icon: Search },
] as const;

function ExpandableSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors text-left"
      >
        <span className="text-sm font-semibold text-[var(--text-primary)]">
          {title}
        </span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown size={16} className="text-[var(--text-tertiary)]" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 py-3 bg-[var(--bg-primary)]">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function LearnTab({
  lesson,
  phaseColor,
}: {
  lesson: Lesson;
  phaseColor: string;
}) {
  return (
    <div className="space-y-6">
      {/* Problem Statement */}
      {lesson.problemStatement && (
        <div className="rounded-lg border border-amber-400/30 bg-amber-500/10 p-4">
          <h3 className="text-sm font-semibold text-amber-300 mb-1">
            Problem Statement
          </h3>
          <p className="text-sm text-amber-200/80">{lesson.problemStatement}</p>
        </div>
      )}

      {/* Learning Objectives */}
      {lesson.objectives.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-2">
            Learning Objectives
          </h3>
          <ul className="space-y-1.5">
            {lesson.objectives.map((obj, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-[var(--text-secondary)]"
              >
                <span style={{ color: phaseColor }} className="mt-0.5">
                  &#9679;
                </span>
                {obj}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Content Sections */}
      {lesson.sections.length > 0 && (
        <div className="space-y-3">
          {lesson.sections.map((section, i) => (
            <ExpandableSection
              key={i}
              title={section.title}
              defaultOpen={i === 0}
            >
              <ContentRenderer blocks={section.blocks} />
            </ExpandableSection>
          ))}
        </div>
      )}

      {/* Before/After Table */}
      {lesson.beforeAfter && lesson.beforeAfter.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
            Before / After
          </h3>
          <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-[var(--bg-tertiary)]">
                  <th className="px-4 py-2 text-left font-semibold text-[var(--text-primary)] border-b border-[var(--border)]">
                    Component
                  </th>
                  <th className="px-4 py-2 text-left font-semibold text-[var(--text-primary)] border-b border-[var(--border)]">
                    Before
                  </th>
                  <th className="px-4 py-2 text-left font-semibold text-[var(--text-primary)] border-b border-[var(--border)]">
                    After
                  </th>
                </tr>
              </thead>
              <tbody>
                {lesson.beforeAfter.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--border)] last:border-0"
                  >
                    <td className="px-4 py-2 font-medium text-[var(--text-primary)]">
                      {row.component}
                    </td>
                    <td className="px-4 py-2 text-[var(--text-secondary)]">
                      {row.before}
                    </td>
                    <td className="px-4 py-2 text-[var(--text-secondary)]">
                      {row.after}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Exercises */}
      {lesson.exercises.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
            Exercises
          </h3>
          <div className="space-y-3">
            {lesson.exercises.map((ex) => (
              <Card key={ex.id}>
                <h4 className="text-sm font-semibold text-[var(--text-primary)] mb-1">
                  {ex.title}
                </h4>
                <p className="text-sm text-[var(--text-secondary)] mb-3">
                  {ex.description}
                </p>

                {ex.labFile && (
                  <p className="text-xs font-mono text-[var(--text-tertiary)] mb-2">
                    Lab file: {ex.labFile}
                  </p>
                )}

                {/* Hints */}
                {ex.hints.length > 0 && (
                  <ExpandableSection title="Hints">
                    <ul className="space-y-1">
                      {ex.hints.map((h, i) => (
                        <li
                          key={i}
                          className="text-sm text-[var(--text-secondary)]"
                        >
                          {i + 1}. {h}
                        </li>
                      ))}
                    </ul>
                  </ExpandableSection>
                )}

                {/* Pseudocode */}
                {ex.pseudocode && (
                  <div className="mt-2">
                    <ExpandableSection title="Pseudocode">
                      <pre className="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap">
                        {ex.pseudocode}
                      </pre>
                    </ExpandableSection>
                  </div>
                )}
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Acceptance Criteria */}
      {lesson.acceptanceCriteria.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-2">
            Acceptance Criteria
          </h3>
          <ul className="space-y-1">
            {lesson.acceptanceCriteria.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-sm text-[var(--text-secondary)]"
              >
                <span style={{ color: phaseColor }}>&#10003;</span>
                {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* References */}
      {lesson.references.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-2">
            References
          </h3>
          <div className="space-y-2">
            {lesson.references.map((ref, i) => (
              <a
                key={i}
                href={ref.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-2 p-3 rounded-lg border border-[var(--border)] hover:border-[var(--border-hover)] bg-[var(--bg-secondary)] transition-colors"
              >
                <ExternalLink
                  size={14}
                  className="mt-0.5 shrink-0 text-[var(--text-tertiary)]"
                />
                <div>
                  <span className="text-sm font-medium text-[var(--text-primary)]">
                    {ref.title}
                  </span>
                  <p className="text-xs text-[var(--text-tertiary)]">
                    {ref.description}
                  </p>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function VisualizeTab({ phaseId }: { phaseId: number }) {
  return (
    <div>
      <PhaseVisualization phaseId={phaseId} />
      <div className="mt-6 text-center">
        <p className="text-sm text-[var(--text-tertiary)]">
          Interactive visualization for this phase. Step through the process to
          see how each component works.
        </p>
      </div>
    </div>
  );
}

function CodeTab({ phaseId }: { phaseId: number }) {
  return (
    <div className="py-12 text-center">
      <Code2 size={40} className="mx-auto mb-4 text-[var(--text-tertiary)]" />
      <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">
        Lab Code
      </h3>
      <p className="text-sm text-[var(--text-secondary)] max-w-md mx-auto mb-4">
        View the lab code in your local repository at{" "}
        <code className="font-mono text-xs bg-[var(--bg-tertiary)] px-1.5 py-0.5 rounded">
          labs/phase{phaseId}_*/
        </code>
      </p>
      <a
        href={`https://github.com/llm-from-scratch/labs/tree/main/phase${phaseId}`}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium border border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-secondary)] transition-colors"
      >
        View on GitHub
        <ExternalLink size={14} />
      </a>
    </div>
  );
}

function DeepDiveTab({ lesson }: { lesson: Lesson }) {
  return (
    <div className="space-y-6">
      <Card>
        <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
          Execution Flow
        </h3>
        <p className="text-sm text-[var(--text-secondary)]">
          Detailed execution flow analysis coming soon. This section will trace
          the data path through each component step by step.
        </p>
      </Card>

      <Card>
        <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
          Architecture Diagram
        </h3>
        <p className="text-sm text-[var(--text-secondary)]">
          Component architecture diagrams coming soon. This will show how the
          pieces of this lesson connect to the overall model.
        </p>
      </Card>

      <Card>
        <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-3">
          What&apos;s New
        </h3>
        <p className="text-sm text-[var(--text-secondary)]">
          Detailed breakdown of new concepts and code introduced in this lesson.
        </p>
      </Card>

      {/* References */}
      {lesson.references.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold uppercase text-[var(--text-tertiary)] mb-2">
            References
          </h3>
          <div className="space-y-2">
            {lesson.references.map((ref, i) => (
              <a
                key={i}
                href={ref.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-2 p-3 rounded-lg border border-[var(--border)] hover:border-[var(--border-hover)] bg-[var(--bg-secondary)] transition-colors"
              >
                <ExternalLink
                  size={14}
                  className="mt-0.5 shrink-0 text-[var(--text-tertiary)]"
                />
                <div>
                  <span className="text-sm font-medium text-[var(--text-primary)]">
                    {ref.title}
                  </span>
                  <p className="text-xs text-[var(--text-tertiary)]">
                    {ref.description}
                  </p>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function LessonPage() {
  const params = useParams();
  const locale = (params?.locale as string) ?? "en";
  const phaseId = Number(params?.phaseId);
  const lessonId = Number(params?.lessonId);
  const [activeTab, setActiveTab] = useState<string>("learn");

  const phases = getPhases(locale as Locale);
  const phase = phases.find((p) => p.id === phaseId);

  let lesson: Lesson | null = null;
  let phaseContent: ReturnType<typeof getPhaseContent> | null = null;
  try {
    lesson = getLesson(phaseId, lessonId, locale as Locale) ?? null;
    phaseContent = getPhaseContent(phaseId, locale as Locale);
  } catch {
    // Lesson data may not exist yet
  }

  if (!phase || !lesson) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-[var(--text-primary)] mb-2">
            Lesson Not Found
          </h1>
          <p className="text-[var(--text-secondary)] mb-4">
            Phase {phaseId}, Lesson {lessonId} does not exist.
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

  const phaseColor = phase.color;
  const totalLessons = phaseContent?.lessons.length ?? 0;
  const prevLessonId = lessonId > 1 ? lessonId - 1 : null;
  const nextLessonId = lessonId < totalLessons ? lessonId + 1 : null;

  return (
    <div>
      <div className="py-2">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-[var(--text-tertiary)] mb-6">
          <Link
            href={`/${locale}`}
            className="hover:text-[var(--text-secondary)] transition-colors"
          >
            Roadmap
          </Link>
          <span>/</span>
          <Link
            href={`/${locale}/phase/${phaseId}`}
            className="hover:text-[var(--text-secondary)] transition-colors"
            style={{ color: phaseColor }}
          >
            {phase.title}
          </Link>
          <span>/</span>
          <span className="text-[var(--text-secondary)]">
            Lesson {lessonId}
          </span>
        </div>

        {/* Lesson header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-2">
            <Badge color={phaseColor}>Phase {phaseId}</Badge>
            <Badge>{lesson.type}</Badge>
            <Badge>{lesson.duration}</Badge>
          </div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)] mb-1">
            {lesson.title}
          </h1>
          <p className="text-sm text-[var(--text-secondary)]">
            {lesson.subtitle}
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-[var(--border)] mb-6">
          {LESSON_TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors relative ${
                  activeTab === tab.id
                    ? "text-[var(--text-primary)]"
                    : "text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
                }`}
              >
                <Icon size={14} />
                {tab.label}
                {activeTab === tab.id && (
                  <motion.span
                    layoutId="lesson-tab-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5"
                    style={{ backgroundColor: phaseColor }}
                  />
                )}
              </button>
            );
          })}
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
            {activeTab === "learn" && (
              <LearnTab lesson={lesson} phaseColor={phaseColor} />
            )}
            {activeTab === "visualize" && <VisualizeTab phaseId={phaseId} />}
            {activeTab === "code" && <CodeTab phaseId={phaseId} />}
            {activeTab === "deep-dive" && <DeepDiveTab lesson={lesson} />}
          </motion.div>
        </AnimatePresence>

        {/* Prev/Next navigation */}
        <div className="flex items-center justify-between mt-12 pt-6 border-t border-[var(--border)]">
          {prevLessonId ? (
            <Link
              href={`/${locale}/phase/${phaseId}/lesson/${prevLessonId}`}
              className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <ArrowLeft size={14} />
              Lesson {prevLessonId}
            </Link>
          ) : (
            <Link
              href={`/${locale}/phase/${phaseId}`}
              className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <ArrowLeft size={14} />
              Phase Overview
            </Link>
          )}

          {nextLessonId ? (
            <Link
              href={`/${locale}/phase/${phaseId}/lesson/${nextLessonId}`}
              className="flex items-center gap-2 text-sm font-medium transition-colors"
              style={{ color: phaseColor }}
            >
              Lesson {nextLessonId}
              <ArrowRight size={14} />
            </Link>
          ) : (
            <span className="text-sm text-[var(--text-tertiary)]">
              Last lesson in this phase
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
