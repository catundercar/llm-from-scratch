"use client";

import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Step 1: Input Tokens",
    description:
      'Four input tokens — "The", "cat", "sat", "down" — are embedded as vectors and fed into the attention mechanism.',
  },
  {
    title: "Step 2: Q, K, V Projections",
    description:
      "Each token is linearly projected into three vectors: Query (Q), Key (K), and Value (V) using learned weight matrices.",
  },
  {
    title: "Step 3: Q × K^T (Raw Scores)",
    description:
      "The raw attention scores are computed by taking the dot product of each Query with every Key. Higher scores mean stronger affinity.",
  },
  {
    title: "Step 4: Scale by √d_k",
    description:
      "Scores are divided by √d_k (square root of key dimension) to prevent softmax from saturating on large values.",
  },
  {
    title: "Step 5: Causal Mask",
    description:
      "A causal (triangular) mask sets future positions to -∞ so each token can only attend to itself and prior tokens.",
  },
  {
    title: "Step 6: Softmax",
    description:
      "Row-wise softmax converts masked scores into probability distributions. Each row sums to 1.",
  },
  {
    title: "Step 7: Attention × V",
    description:
      "The attention weights multiply the Value vectors to produce the final attended output for each token position.",
  },
];

const TOKENS = ["The", "cat", "sat", "down"];

// Raw QK^T scores (made up but plausible)
const RAW_SCORES = [
  [3.2, 1.1, 0.5, 0.8],
  [2.1, 4.0, 1.2, 0.3],
  [0.9, 2.3, 3.8, 1.5],
  [1.0, 1.4, 2.2, 3.6],
];

const DK = 64;
const SCALE = Math.sqrt(DK); // 8

// Scaled scores
const SCALED_SCORES = RAW_SCORES.map((row) =>
  row.map((v) => +(v / SCALE).toFixed(2))
);

// After causal mask (upper-right triangle → -∞)
const MASKED_SCORES = SCALED_SCORES.map((row, i) =>
  row.map((v, j) => (j > i ? -Infinity : v))
);

// Softmax per row (only over non-masked positions)
function softmaxRow(row: number[]): number[] {
  const finite = row.map((v) => (v === -Infinity ? -1e9 : v));
  const maxVal = Math.max(...finite);
  const exps = finite.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => +(e / sum).toFixed(3));
}

const SOFTMAX_SCORES = MASKED_SCORES.map(softmaxRow);

// Colors
const QUERY_COLOR = "#3b82f6"; // blue
const KEY_COLOR = "#10b981"; // green
const VALUE_COLOR = "#f59e0b"; // amber
const HEAT_COLOR = "139, 92, 246"; // purple RGB for heatmap

// Grid layout constants
const CELL_SIZE = 56;
const LABEL_OFFSET = 60;
const GRID_SIZE = TOKENS.length * CELL_SIZE;

function getCellValue(step: number, row: number, col: number): string {
  if (step <= 1) return "";
  if (step === 2) return RAW_SCORES[row][col].toFixed(1);
  if (step === 3) return SCALED_SCORES[row][col].toFixed(2);
  if (step === 4) {
    return col > row ? "-∞" : SCALED_SCORES[row][col].toFixed(2);
  }
  if (step >= 5) {
    return col > row ? "0" : SOFTMAX_SCORES[row][col].toFixed(2);
  }
  return "";
}

function getCellOpacity(step: number, row: number, col: number): number {
  if (step <= 1) return 0;
  if (step === 2) {
    // Normalize raw scores for heatmap
    return RAW_SCORES[row][col] / 5;
  }
  if (step === 3) {
    return SCALED_SCORES[row][col] / 0.6;
  }
  if (step === 4) {
    if (col > row) return 0.03;
    return SCALED_SCORES[row][col] / 0.6;
  }
  if (step >= 5) {
    if (col > row) return 0.02;
    return SOFTMAX_SCORES[row][col];
  }
  return 0;
}

function isMasked(step: number, row: number, col: number): boolean {
  return step >= 4 && col > row;
}

export default function AttentionMatrix({ title }: { title?: string }) {
  const viz = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });

  const step = viz.currentStep;

  const svgWidth = LABEL_OFFSET + GRID_SIZE + 20;
  const svgHeight = LABEL_OFFSET + GRID_SIZE + 20;

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
      {title && (
        <h3 className="mb-4 text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      <div className="flex flex-col items-center gap-4">
        {/* Token row (step 0+) */}
        <div className="flex items-center gap-2">
          {TOKENS.map((token, i) => (
            <motion.div
              key={token}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-2 text-sm font-semibold text-[var(--foreground)]"
              style={{ fontFamily: "var(--font-geist-mono), monospace" }}
            >
              {token}
            </motion.div>
          ))}
        </div>

        {/* QKV Projection indicators (step 1) */}
        {step === 1 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-6"
          >
            {[
              { label: "Q", color: QUERY_COLOR },
              { label: "K", color: KEY_COLOR },
              { label: "V", color: VALUE_COLOR },
            ].map(({ label, color }) => (
              <div key={label} className="flex items-center gap-2">
                <div
                  className="h-3 w-3 rounded-sm"
                  style={{ backgroundColor: color }}
                />
                <span
                  className="text-sm font-semibold"
                  style={{
                    color,
                    fontFamily: "var(--font-geist-mono), monospace",
                  }}
                >
                  {label}
                </span>
                <div className="flex gap-0.5">
                  {TOKENS.map((_, j) => (
                    <motion.div
                      key={j}
                      initial={{ scaleY: 0 }}
                      animate={{ scaleY: 1 }}
                      transition={{ delay: 0.2 + j * 0.08 }}
                      className="h-5 w-6 rounded-sm"
                      style={{
                        backgroundColor: color,
                        opacity: 0.3 + j * 0.15,
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </motion.div>
        )}

        {/* Main attention grid (step 2+) */}
        {step >= 2 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="overflow-x-auto"
          >
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full"
              style={{ maxWidth: 380, maxHeight: 380 }}
            >
              {/* Column labels (Keys) */}
              {TOKENS.map((token, j) => (
                <text
                  key={`col-${j}`}
                  x={LABEL_OFFSET + j * CELL_SIZE + CELL_SIZE / 2}
                  y={LABEL_OFFSET - 10}
                  textAnchor="middle"
                  fontSize={12}
                  fontFamily="var(--font-geist-mono), monospace"
                  fontWeight={600}
                  fill="var(--text-secondary)"
                >
                  {token}
                </text>
              ))}

              {/* Row labels (Queries) */}
              {TOKENS.map((token, i) => (
                <text
                  key={`row-${i}`}
                  x={LABEL_OFFSET - 10}
                  y={LABEL_OFFSET + i * CELL_SIZE + CELL_SIZE / 2}
                  textAnchor="end"
                  dominantBaseline="central"
                  fontSize={12}
                  fontFamily="var(--font-geist-mono), monospace"
                  fontWeight={600}
                  fill="var(--text-secondary)"
                >
                  {token}
                </text>
              ))}

              {/* Axis labels */}
              <text
                x={LABEL_OFFSET + GRID_SIZE / 2}
                y={12}
                textAnchor="middle"
                fontSize={10}
                fill="var(--text-tertiary)"
                fontFamily="var(--font-geist-mono), monospace"
              >
                Keys
              </text>
              <text
                x={10}
                y={LABEL_OFFSET + GRID_SIZE / 2}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={10}
                fill="var(--text-tertiary)"
                fontFamily="var(--font-geist-mono), monospace"
                transform={`rotate(-90, 10, ${LABEL_OFFSET + GRID_SIZE / 2})`}
              >
                Queries
              </text>

              {/* Grid cells */}
              {TOKENS.map((_, i) =>
                TOKENS.map((_, j) => {
                  const x = LABEL_OFFSET + j * CELL_SIZE;
                  const y = LABEL_OFFSET + i * CELL_SIZE;
                  const value = getCellValue(step, i, j);
                  const opacity = getCellOpacity(step, i, j);
                  const masked = isMasked(step, i, j);

                  return (
                    <motion.g key={`${i}-${j}`}>
                      {/* Cell background */}
                      <motion.rect
                        x={x + 1}
                        y={y + 1}
                        width={CELL_SIZE - 2}
                        height={CELL_SIZE - 2}
                        rx={4}
                        initial={{ opacity: 0 }}
                        animate={{
                          opacity: 1,
                          fill: masked
                            ? "var(--bg-primary)"
                            : `rgba(${HEAT_COLOR}, ${Math.min(opacity, 1)})`,
                        }}
                        transition={{ duration: 0.4 }}
                      />

                      {/* Cell border */}
                      <rect
                        x={x + 1}
                        y={y + 1}
                        width={CELL_SIZE - 2}
                        height={CELL_SIZE - 2}
                        rx={4}
                        fill="none"
                        stroke="var(--border)"
                        strokeWidth={0.5}
                      />

                      {/* Mask pattern for masked cells */}
                      {masked && step === 4 && (
                        <motion.g
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 0.3 }}
                        >
                          {/* Diagonal lines to indicate masking */}
                          <line
                            x1={x + 6}
                            y1={y + 6}
                            x2={x + CELL_SIZE - 6}
                            y2={y + CELL_SIZE - 6}
                            stroke="var(--text-tertiary)"
                            strokeWidth={1}
                          />
                          <line
                            x1={x + CELL_SIZE - 6}
                            y1={y + 6}
                            x2={x + 6}
                            y2={y + CELL_SIZE - 6}
                            stroke="var(--text-tertiary)"
                            strokeWidth={1}
                          />
                        </motion.g>
                      )}

                      {/* Cell value text */}
                      <motion.text
                        x={x + CELL_SIZE / 2}
                        y={y + CELL_SIZE / 2}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize={value === "-∞" ? 11 : 10}
                        fontFamily="var(--font-geist-mono), monospace"
                        fontWeight={500}
                        initial={{ opacity: 0 }}
                        animate={{
                          opacity: 1,
                          fill: masked
                            ? "var(--text-tertiary)"
                            : opacity > 0.5
                              ? "#ffffff"
                              : "var(--foreground)",
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {value}
                      </motion.text>
                    </motion.g>
                  );
                })
              )}
            </svg>

            {/* Step-specific label under grid */}
            <div className="mt-2 text-center">
              <span
                className="text-xs text-[var(--text-tertiary)]"
                style={{ fontFamily: "var(--font-geist-mono), monospace" }}
              >
                {step === 2 && "QK^T raw scores"}
                {step === 3 && `Scaled by 1/√${DK} = 1/${SCALE}`}
                {step === 4 && "Causal mask applied (upper triangle → -∞)"}
                {step === 5 && "Softmax probabilities (rows sum to 1)"}
                {step === 6 && "Output = Attention weights × V"}
              </span>
            </div>
          </motion.div>
        )}

        {/* Output representation (step 6) */}
        {step === 6 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2"
          >
            <span
              className="text-xs text-[var(--text-secondary)]"
              style={{ fontFamily: "var(--font-geist-mono), monospace" }}
            >
              Output:
            </span>
            {TOKENS.map((token, i) => (
              <motion.div
                key={token}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.12 }}
                className="rounded-lg px-3 py-1.5 text-xs font-semibold text-white"
                style={{
                  backgroundColor: VALUE_COLOR,
                  fontFamily: "var(--font-geist-mono), monospace",
                }}
              >
                {token}&apos;
              </motion.div>
            ))}
          </motion.div>
        )}
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
