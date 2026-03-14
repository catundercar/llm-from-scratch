export interface Deliverable {
  name: string;
  desc: string;
  acceptance: string[];
}

export interface Phase {
  id: number;
  week: string;
  duration: string;
  title: string;
  subtitle: string;
  icon: string;
  color: string;
  accent: string;
  goal: string;
  concepts: string[];
  readings: string[];
  deliverable: Deliverable;
  loc?: number;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  prerequisites?: number[];
  layer?: string;
  keyInsight?: string;
  coreAddition?: string;
}

export interface ArchitectureLayer {
  name: string;
  color: string;
  modules: string[];
}

export interface Architecture {
  layers: ArchitectureLayer[];
}

export interface Principle {
  num: string;
  title: string;
  desc: string;
  color: string;
}

// Rich content blocks for LessonPage
export type ContentBlock =
  | ParagraphBlock | HeadingBlock | CodeBlock | DiagramBlock
  | TableBlock | CalloutBlock | ListBlock;

export interface ParagraphBlock { type: "paragraph"; text: string; }
export interface HeadingBlock { type: "heading"; level: 2 | 3 | 4; text: string; }
export interface CodeBlock { type: "code"; language?: string; code: string; }
export interface DiagramBlock { type: "diagram"; content: string; }
export interface TableBlock { type: "table"; headers: string[]; rows: string[][]; }
export interface CalloutBlock { type: "callout"; variant: "info" | "warning" | "tip" | "quote"; text: string; }
export interface ListBlock { type: "list"; ordered: boolean; items: string[]; }

export interface ContentSection { title: string; blocks: ContentBlock[]; }

export interface LessonReference { title: string; description: string; url: string; }
export interface CodeExercise {
  id: string; title: string; description: string;
  labFile?: string; hints: string[]; pseudocode?: string;
}
export interface Lesson {
  phaseId: number; lessonId: number;
  title: string; subtitle: string;
  type: string; duration: string;
  objectives: string[];
  sections: ContentSection[];
  exercises: CodeExercise[];
  acceptanceCriteria: string[];
  references: LessonReference[];
  problemStatement?: string;
  beforeAfter?: { component: string; before: string; after: string }[];
}
export interface PhaseContent {
  phaseId: number; color: string; accent: string; lessons: Lesson[];
}
