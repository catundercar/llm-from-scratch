export type { Phase, PhaseContent, Lesson, ContentBlock, ContentSection, CodeExercise, LessonReference, Architecture, ArchitectureLayer, Principle, Deliverable } from "@/data/types";
export { getPhases, architecture, getPrinciples, dataFlowSteps } from "@/data/phases";
import { getAllPhaseContent } from "@/data/lessons";
import type { PhaseContent, Lesson } from "@/data/types";
import type { Locale } from "@/i18n";

export function getPhaseContent(phaseId: number, locale: Locale = "zh-CN"): PhaseContent | undefined {
  return getAllPhaseContent(locale).find((p) => p.phaseId === phaseId);
}

export function getLesson(phaseId: number, lessonId: number, locale: Locale = "zh-CN"): Lesson | undefined {
  const pc = getPhaseContent(phaseId, locale);
  if (!pc) return undefined;
  return pc.lessons.find((l) => l.lessonId === lessonId);
}

export function hasLessons(phaseId: number, locale: Locale = "zh-CN"): boolean {
  const pc = getPhaseContent(phaseId, locale);
  return !!pc && pc.lessons.length > 0;
}

export function getLessonCount(phaseId: number, locale: Locale = "zh-CN"): number {
  const pc = getPhaseContent(phaseId, locale);
  return pc ? pc.lessons.length : 0;
}
