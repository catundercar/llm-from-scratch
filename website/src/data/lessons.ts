import type { PhaseContent } from "@/data/types";
import type { Locale } from "@/i18n";

// Phase 0: fully translated, locale-aware
import { getPhase0Content } from "@/data/lessons_p0";
// Phases 1-9: still zh-TW only, will be converted by agents
import { getPhase1Content, getPhase2Content } from "@/data/lessons_p1_p2";
import { getPhase3Content, getPhase4Content } from "@/data/lessons_p3_p4";
import { getPhase5Content, getPhase6Content } from "@/data/lessons_p5_p6";
import { getPhase7Content, getPhase8Content, getPhase9Content } from "@/data/lessons_p7_p8_p9";

export function getAllPhaseContent(locale: Locale): PhaseContent[] {
  return [
    getPhase0Content(locale),
    getPhase1Content(locale), getPhase2Content(locale), getPhase3Content(locale), getPhase4Content(locale),
    getPhase5Content(locale), getPhase6Content(locale), getPhase7Content(locale), getPhase8Content(locale), getPhase9Content(locale),
  ];
}
