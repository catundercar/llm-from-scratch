"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Step 1: Raw Text",
    description:
      'The input string "hello world" is split into individual characters. Each character becomes an initial token in our vocabulary.',
  },
  {
    title: "Step 2: Character Pairs",
    description:
      "We scan adjacent character pairs and count how often each pair appears. The most frequent pair will be merged first.",
  },
  {
    title: "Step 3: First Merge",
    description:
      'The most frequent pair "l"+"l" is merged into a single token "ll". The token sequence is updated.',
  },
  {
    title: "Step 4: Second Merge",
    description:
      'The next most frequent pair "h"+"e" merges into "he". Our vocabulary grows with each merge.',
  },
  {
    title: "Step 5: Third Merge",
    description:
      '"he"+"ll" merges into "hell". Longer tokens emerge from repeated merging of frequent pairs.',
  },
  {
    title: "Step 6: Final Vocabulary",
    description:
      "The final token sequence and full merge vocabulary are shown. BPE builds a vocabulary bottom-up from characters.",
  },
];

// Token colors by merge level
const LEVEL_COLORS = [
  { bg: "#6366f1", text: "#ffffff" }, // indigo — level 0 (chars)
  { bg: "#8b5cf6", text: "#ffffff" }, // violet — level 1
  { bg: "#a855f7", text: "#ffffff" }, // purple — level 2
  { bg: "#d946ef", text: "#ffffff" }, // fuchsia — level 3
  { bg: "#ec4899", text: "#ffffff" }, // pink — level 4
  { bg: "#f43f5e", text: "#ffffff" }, // rose — level 5
];

interface Token {
  text: string;
  level: number;
  id: string;
}

// Token sequences at each step
const SEQUENCES: Token[][] = [
  // Step 0: raw characters
  "hello world".split("").map((c, i) => ({
    text: c === " " ? "⎵" : c,
    level: 0,
    id: `c${i}`,
  })),
  // Step 1: same as step 0 (pairs highlighted in UI)
  "hello world".split("").map((c, i) => ({
    text: c === " " ? "⎵" : c,
    level: 0,
    id: `c${i}`,
  })),
  // Step 2: merge l+l → ll
  [
    { text: "h", level: 0, id: "c0" },
    { text: "e", level: 0, id: "c1" },
    { text: "ll", level: 1, id: "m1" },
    { text: "o", level: 0, id: "c4" },
    { text: "⎵", level: 0, id: "c5" },
    { text: "w", level: 0, id: "c6" },
    { text: "o", level: 0, id: "c7" },
    { text: "r", level: 0, id: "c8" },
    { text: "l", level: 0, id: "c9" },
    { text: "d", level: 0, id: "c10" },
  ],
  // Step 3: merge h+e → he
  [
    { text: "he", level: 2, id: "m2" },
    { text: "ll", level: 1, id: "m1" },
    { text: "o", level: 0, id: "c4" },
    { text: "⎵", level: 0, id: "c5" },
    { text: "w", level: 0, id: "c6" },
    { text: "o", level: 0, id: "c7" },
    { text: "r", level: 0, id: "c8" },
    { text: "l", level: 0, id: "c9" },
    { text: "d", level: 0, id: "c10" },
  ],
  // Step 4: merge he+ll → hell
  [
    { text: "hell", level: 3, id: "m3" },
    { text: "o", level: 0, id: "c4" },
    { text: "⎵", level: 0, id: "c5" },
    { text: "w", level: 0, id: "c6" },
    { text: "o", level: 0, id: "c7" },
    { text: "r", level: 0, id: "c8" },
    { text: "l", level: 0, id: "c9" },
    { text: "d", level: 0, id: "c10" },
  ],
  // Step 5: final — same as step 4 (show vocab)
  [
    { text: "hell", level: 3, id: "m3" },
    { text: "o", level: 0, id: "c4" },
    { text: "⎵", level: 0, id: "c5" },
    { text: "w", level: 0, id: "c6" },
    { text: "o", level: 0, id: "c7" },
    { text: "r", level: 0, id: "c8" },
    { text: "l", level: 0, id: "c9" },
    { text: "d", level: 0, id: "c10" },
  ],
];

// Pair frequencies for step 1
const PAIR_FREQUENCIES = [
  { pair: ["l", "l"], count: 1, highlight: true },
  { pair: ["h", "e"], count: 1, highlight: false },
  { pair: ["e", "l"], count: 1, highlight: false },
  { pair: ["l", "o"], count: 2, highlight: false },
  { pair: ["o", "⎵"], count: 1, highlight: false },
  { pair: ["⎵", "w"], count: 1, highlight: false },
  { pair: ["w", "o"], count: 1, highlight: false },
  { pair: ["o", "r"], count: 1, highlight: false },
  { pair: ["r", "l"], count: 1, highlight: false },
  { pair: ["l", "d"], count: 1, highlight: false },
];

// Merge table entries revealed per step
const MERGE_TABLE = [
  { left: "l", right: "l", result: "ll", step: 2 },
  { left: "h", right: "e", result: "he", step: 3 },
  { left: "he", right: "ll", result: "hell", step: 4 },
];

export default function BPETokenizer({ title }: { title?: string }) {
  const viz = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });

  const step = viz.currentStep;
  const tokens = SEQUENCES[step];

  // Calculate token box dimensions
  const tokenWidth = (t: Token) => Math.max(40, t.text.length * 18 + 20);
  const tokenHeight = 44;
  const gap = 8;

  // Total width of token row
  const totalTokenWidth = tokens.reduce(
    (sum, t, i) => sum + tokenWidth(t) + (i < tokens.length - 1 ? gap : 0),
    0
  );

  const svgWidth = Math.max(totalTokenWidth + 40, 500);
  const svgHeight = 100;

  // Pair highlight indices for step 1
  const pairHighlightIndices: [number, number][] =
    step === 1
      ? tokens
          .map((_, i) => [i, i + 1] as [number, number])
          .filter(([, b]) => b < tokens.length)
      : [];

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
      {title && (
        <h3 className="mb-4 text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      <div className="flex flex-col gap-4 lg:flex-row">
        {/* Main SVG area */}
        <div className="flex-1 overflow-x-auto">
          <svg
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="w-full"
            style={{ minWidth: 400, maxHeight: 120 }}
          >
            {/* Token boxes */}
            <AnimatePresence mode="popLayout">
              {(() => {
                let x = (svgWidth - totalTokenWidth) / 2;
                return tokens.map((token, i) => {
                  const w = tokenWidth(token);
                  const color = LEVEL_COLORS[token.level];
                  const cx = x;
                  x += w + gap;

                  // Check if this token is part of a highlighted pair
                  const inPair = pairHighlightIndices.some(
                    ([a, b]) => i === a || i === b
                  );

                  return (
                    <motion.g
                      key={token.id}
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{
                        opacity: 1,
                        scale: 1,
                        x: cx,
                        y: (svgHeight - tokenHeight) / 2,
                      }}
                      exit={{ opacity: 0, scale: 0.5 }}
                      transition={{
                        type: "spring",
                        stiffness: 300,
                        damping: 25,
                      }}
                    >
                      <rect
                        width={w}
                        height={tokenHeight}
                        rx={8}
                        fill={color.bg}
                        stroke={inPair ? "#facc15" : "transparent"}
                        strokeWidth={inPair ? 3 : 0}
                      />
                      <text
                        x={w / 2}
                        y={tokenHeight / 2}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fill={color.text}
                        fontSize={16}
                        fontFamily="var(--font-geist-mono), monospace"
                        fontWeight={600}
                      >
                        {token.text}
                      </text>
                    </motion.g>
                  );
                });
              })()}
            </AnimatePresence>
          </svg>
        </div>

        {/* Side panel: pair frequencies or merge table */}
        <div className="w-full shrink-0 lg:w-52">
          {step === 1 && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] p-3">
              <p
                className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]"
              >
                Pair Frequencies
              </p>
              <div className="space-y-1">
                {PAIR_FREQUENCIES.map((pf, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="flex items-center justify-between rounded px-2 py-1 text-sm"
                    style={{
                      fontFamily: "var(--font-geist-mono), monospace",
                      backgroundColor: pf.count >= 2 ? "rgba(250,204,21,0.15)" : "transparent",
                    }}
                  >
                    <span className="text-[var(--foreground)]">
                      {pf.pair[0]}+{pf.pair[1]}
                    </span>
                    <span className="text-[var(--text-secondary)]">
                      {pf.count}
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {step >= 2 && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] p-3">
              <p
                className="mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]"
              >
                Merge Vocabulary
              </p>
              <div className="space-y-1">
                {MERGE_TABLE.filter((m) => m.step <= step).map((m, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-1 rounded px-2 py-1 text-sm"
                    style={{
                      fontFamily: "var(--font-geist-mono), monospace",
                      backgroundColor:
                        m.step === step
                          ? "rgba(139,92,246,0.15)"
                          : "transparent",
                    }}
                  >
                    <span className="text-[var(--text-secondary)]">
                      {m.left}
                    </span>
                    <span className="text-[var(--text-tertiary)]">+</span>
                    <span className="text-[var(--text-secondary)]">
                      {m.right}
                    </span>
                    <span className="text-[var(--text-tertiary)]">&rarr;</span>
                    <span
                      className="font-semibold"
                      style={{
                        color: LEVEL_COLORS[i + 1]?.bg ?? LEVEL_COLORS[0].bg,
                      }}
                    >
                      {m.result}
                    </span>
                  </motion.div>
                ))}
              </div>

              {step === 5 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-3 border-t border-[var(--border)] pt-2"
                >
                  <p className="text-xs text-[var(--text-tertiary)]">
                    Vocab size: {11 + MERGE_TABLE.length} tokens
                    <br />
                    (11 base + {MERGE_TABLE.length} merges)
                  </p>
                </motion.div>
              )}
            </div>
          )}

          {step === 0 && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] p-3">
              <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-[var(--text-secondary)]">
                Input
              </p>
              <p
                className="text-sm text-[var(--foreground)]"
                style={{ fontFamily: "var(--font-geist-mono), monospace" }}
              >
                &quot;hello world&quot;
              </p>
              <p className="mt-2 text-xs text-[var(--text-tertiary)]">
                11 characters &rarr; 11 initial tokens
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="mt-5">
        <StepControls
          currentStep={viz.currentStep}
          totalSteps={viz.totalSteps}
          onPrev={viz.prev}
          onNext={viz.next}
          onReset={viz.reset}
          isPlaying={viz.isPlaying}
          onToggleAutoPlay={viz.toggleAutoPlay}
          stepTitle={STEP_INFO[step].title}
          stepDescription={STEP_INFO[step].description}
        />
      </div>
    </div>
  );
}
