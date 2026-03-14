"use client";

import { lazy, Suspense } from "react";

const visualizations: Record<
  number,
  React.LazyExoticComponent<React.ComponentType<{ title?: string }>>
> = {
  1: lazy(() => import("./p1-bpe-tokenizer")),
  2: lazy(() => import("./p2-attention-matrix")),
  3: lazy(() => import("./p3-transformer-block")),
  4: lazy(() => import("./p4-training-loop")),
  5: lazy(() => import("./p5-token-generation")),
  6: lazy(() => import("./p6-classification-head")),
  7: lazy(() => import("./p7-lora-decomposition")),
  8: lazy(() => import("./p8-sft-dpo-comparison")),
  9: lazy(() => import("./p9-moe-router")),
};

export function PhaseVisualization({ phaseId, title }: { phaseId: number; title?: string }) {
  const Component = visualizations[phaseId];
  if (!Component) {
    return (
      <div className="min-h-[400px] flex items-center justify-center rounded-xl border border-dashed border-[var(--border)] text-sm text-[var(--text-tertiary)]">
        No interactive visualization available for Phase {phaseId}
      </div>
    );
  }
  return (
    <Suspense
      fallback={
        <div className="min-h-[400px] animate-pulse rounded-xl bg-[var(--bg-secondary)]" />
      }
    >
      <Component title={title} />
    </Suspense>
  );
}
