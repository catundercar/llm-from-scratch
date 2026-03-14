"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Raw Logits",
    description:
      "The model outputs a raw score (logit) for each token in the vocabulary. Higher logits mean the model thinks that token is more likely.",
  },
  {
    title: "Softmax",
    description:
      "Softmax converts logits into a valid probability distribution: all values are positive and sum to 1.",
  },
  {
    title: "Temperature = 0.7",
    description:
      "Lower temperature sharpens the distribution, making the model more confident and deterministic in its choices.",
  },
  {
    title: "Temperature = 1.5",
    description:
      "Higher temperature flattens the distribution, increasing randomness and diversity in generation.",
  },
  {
    title: "Top-k (k = 3)",
    description:
      "Only the top-k most probable tokens are kept. All other tokens are zeroed out and excluded from sampling.",
  },
  {
    title: "Top-p (p = 0.9)",
    description:
      "Tokens are kept in descending probability order until their cumulative probability reaches p. This adapts the cutoff dynamically.",
  },
  {
    title: "Sample",
    description:
      "A token is randomly sampled from the filtered distribution. The selected token becomes the next output.",
  },
];

const TOKENS = ["the", "a", "cat", "dog", "sat", "ran", "big", "red"];
const RAW_LOGITS = [3.2, 2.8, 2.1, 1.5, 0.8, 0.3, -0.5, -1.2];

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function applyTemperature(logits: number[], temp: number): number[] {
  return logits.map((l) => l / temp);
}

function topK(probs: number[], k: number): number[] {
  const indexed = probs.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => b.p - a.p);
  const topIndices = new Set(indexed.slice(0, k).map((x) => x.i));
  return probs.map((p, i) => (topIndices.has(i) ? p : 0));
}

function topP(probs: number[], p: number): number[] {
  const indexed = probs.map((prob, i) => ({ prob, i }));
  indexed.sort((a, b) => b.prob - a.prob);
  let cumulative = 0;
  const keepIndices = new Set<number>();
  for (const item of indexed) {
    if (cumulative >= p) break;
    keepIndices.add(item.i);
    cumulative += item.prob;
  }
  return probs.map((prob, i) => (keepIndices.has(i) ? prob : 0));
}

function renormalize(probs: number[]): number[] {
  const sum = probs.reduce((a, b) => a + b, 0);
  if (sum === 0) return probs;
  return probs.map((p) => p / sum);
}

// Deterministic "sample" — pick the token with index 0 ("the") since it has highest prob
const SELECTED_INDEX = 0;

export default function TokenGenerationVisualization({
  title,
}: {
  title?: string;
}) {
  const stepper = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });
  const step = stepper.currentStep;

  // Compute values for each step
  const { values, isProb, filtered, maxVal } = useMemo(() => {
    switch (step) {
      case 0: {
        // Raw logits
        return {
          values: RAW_LOGITS,
          isProb: false,
          filtered: Array(8).fill(false),
          maxVal: Math.max(...RAW_LOGITS.map(Math.abs)),
        };
      }
      case 1: {
        // Softmax
        const probs = softmax(RAW_LOGITS);
        return {
          values: probs,
          isProb: true,
          filtered: Array(8).fill(false),
          maxVal: Math.max(...probs),
        };
      }
      case 2: {
        // Temperature 0.7
        const probs = softmax(applyTemperature(RAW_LOGITS, 0.7));
        return {
          values: probs,
          isProb: true,
          filtered: Array(8).fill(false),
          maxVal: Math.max(...probs),
        };
      }
      case 3: {
        // Temperature 1.5
        const probs = softmax(applyTemperature(RAW_LOGITS, 1.5));
        return {
          values: probs,
          isProb: true,
          filtered: Array(8).fill(false),
          maxVal: Math.max(...probs),
        };
      }
      case 4: {
        // Top-k = 3
        const probs = softmax(applyTemperature(RAW_LOGITS, 0.7));
        const topKProbs = topK(probs, 3);
        const renormed = renormalize(topKProbs);
        const filt = topKProbs.map((p) => p === 0);
        return {
          values: renormed,
          isProb: true,
          filtered: filt,
          maxVal: Math.max(...renormed),
        };
      }
      case 5: {
        // Top-p = 0.9
        const probs = softmax(applyTemperature(RAW_LOGITS, 0.7));
        const topPProbs = topP(probs, 0.9);
        const renormed = renormalize(topPProbs);
        const filt = topPProbs.map((p) => p === 0);
        return {
          values: renormed,
          isProb: true,
          filtered: filt,
          maxVal: Math.max(...renormed),
        };
      }
      case 6: {
        // Sample — same distribution as top-p, with selection
        const probs = softmax(applyTemperature(RAW_LOGITS, 0.7));
        const topPProbs = topP(probs, 0.9);
        const renormed = renormalize(topPProbs);
        const filt = topPProbs.map((p) => p === 0);
        return {
          values: renormed,
          isProb: true,
          filtered: filt,
          maxVal: Math.max(...renormed),
        };
      }
      default:
        return {
          values: RAW_LOGITS,
          isProb: false,
          filtered: Array(8).fill(false),
          maxVal: 4,
        };
    }
  }, [step]);

  // Chart dimensions
  const chartW = 560;
  const chartH = 240;
  const barAreaTop = 30;
  const barAreaBottom = chartH - 40;
  const barAreaH = barAreaBottom - barAreaTop;
  const barCount = TOKENS.length;
  const barGap = 16;
  const totalBarSpace = chartW - 80;
  const barW = (totalBarSpace - barGap * (barCount - 1)) / barCount;
  const xStart = 40;

  // For logits, center 0 line in bar area
  const zeroY = isProb ? barAreaBottom : barAreaTop + barAreaH * 0.65;

  function barHeight(val: number): number {
    if (isProb) {
      return (val / maxVal) * (barAreaH - 10);
    }
    // logits: scale relative to max abs
    return (val / maxVal) * (barAreaH * 0.55);
  }

  return (
    <div className="flex flex-col gap-4">
      {title && (
        <h3 className="text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
        <svg
          viewBox={`0 0 ${chartW} ${chartH}`}
          className="w-full"
          style={{ minWidth: 400 }}
        >
          {/* Zero / baseline */}
          <line
            x1={xStart - 5}
            y1={zeroY}
            x2={xStart + totalBarSpace + 5}
            y2={zeroY}
            stroke="#64748b"
            strokeWidth={1}
            opacity={0.5}
          />

          {/* Y-axis label */}
          <text
            x={8}
            y={barAreaTop + barAreaH / 2}
            fontSize={10}
            fill="#94a3b8"
            textAnchor="middle"
            transform={`rotate(-90, 8, ${barAreaTop + barAreaH / 2})`}
          >
            {isProb ? "probability" : "logit value"}
          </text>

          {/* Bars */}
          {values.map((val, i) => {
            const x = xStart + i * (barW + barGap);
            const h = Math.abs(barHeight(val));
            const y = isProb ? zeroY - h : val >= 0 ? zeroY - h : zeroY;
            const isFiltered = filtered[i];
            const isSelected = step === 6 && i === SELECTED_INDEX;

            const barColor = isFiltered
              ? "#94a3b8"
              : isSelected
                ? "#f59e0b"
                : "#7c3aed";

            return (
              <g key={i}>
                {/* Bar */}
                <motion.rect
                  x={x}
                  rx={3}
                  fill={barColor}
                  animate={{
                    y,
                    height: h,
                    opacity: isFiltered ? 0.25 : 1,
                  }}
                  transition={{ duration: 0.5, ease: "easeInOut" }}
                />

                {/* Value label */}
                <motion.text
                  x={x + barW / 2}
                  textAnchor="middle"
                  fontSize={9}
                  fontFamily="var(--font-geist-mono), monospace"
                  fill={isFiltered ? "#94a3b8" : "var(--foreground, #e2e8f0)"}
                  animate={{
                    y: isProb ? zeroY - h - 6 : val >= 0 ? zeroY - h - 6 : zeroY + h + 12,
                    opacity: isFiltered && val === 0 ? 0 : 1,
                  }}
                  transition={{ duration: 0.5 }}
                >
                  {isProb
                    ? val === 0
                      ? "0"
                      : val.toFixed(3)
                    : val.toFixed(1)}
                </motion.text>

                {/* Token label */}
                <text
                  x={x + barW / 2}
                  y={barAreaBottom + 16}
                  textAnchor="middle"
                  fontSize={10}
                  fontWeight={isSelected ? 700 : 400}
                  fill={
                    isFiltered
                      ? "#64748b"
                      : isSelected
                        ? "#f59e0b"
                        : "var(--foreground, #e2e8f0)"
                  }
                  fontFamily="var(--font-geist-mono), monospace"
                >
                  {TOKENS[i]}
                </text>

                {/* Selection arrow for step 6 */}
                {isSelected && (
                  <motion.g
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.4 }}
                  >
                    <polygon
                      points={`${x + barW / 2},${zeroY - h - 20} ${x + barW / 2 - 6},${zeroY - h - 30} ${x + barW / 2 + 6},${zeroY - h - 30}`}
                      fill="#f59e0b"
                    />
                    <text
                      x={x + barW / 2}
                      y={zeroY - h - 34}
                      textAnchor="middle"
                      fontSize={9}
                      fontWeight={700}
                      fill="#f59e0b"
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      selected
                    </text>
                  </motion.g>
                )}

                {/* "filtered" x mark */}
                {isFiltered && (
                  <motion.text
                    x={x + barW / 2}
                    y={zeroY - 10}
                    textAnchor="middle"
                    fontSize={14}
                    fill="#94a3b8"
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 0.6, scale: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    x
                  </motion.text>
                )}
              </g>
            );
          })}

          {/* Cumulative probability annotation for top-p step */}
          {step === 5 && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              {(() => {
                const probs = softmax(applyTemperature(RAW_LOGITS, 0.7));
                const indexed = probs
                  .map((p, i) => ({ p, i }))
                  .sort((a, b) => b.p - a.p);
                let cum = 0;
                const annotations: { i: number; cum: number }[] = [];
                for (const item of indexed) {
                  cum += item.p;
                  annotations.push({ i: item.i, cum });
                  if (cum >= 0.9) break;
                }
                return annotations.map(({ i, cum: c }) => {
                  const x = xStart + i * (barW + barGap) + barW / 2;
                  return (
                    <text
                      key={i}
                      x={x}
                      y={barAreaBottom + 30}
                      textAnchor="middle"
                      fontSize={7}
                      fill="#22c55e"
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      cum={c.toFixed(2)}
                    </text>
                  );
                });
              })()}
              <text
                x={chartW / 2}
                y={chartH - 2}
                textAnchor="middle"
                fontSize={9}
                fill="#22c55e"
                fontFamily="var(--font-geist-mono), monospace"
              >
                cumulative cutoff at p = 0.9
              </text>
            </motion.g>
          )}

          {/* Title annotation in top-right */}
          <text
            x={chartW - 10}
            y={16}
            textAnchor="end"
            fontSize={11}
            fontWeight={600}
            fill="#94a3b8"
            fontFamily="var(--font-geist-mono), monospace"
          >
            {STEP_INFO[step].title}
          </text>
        </svg>
      </div>

      <StepControls
        currentStep={stepper.currentStep}
        totalSteps={stepper.totalSteps}
        onPrev={stepper.prev}
        onNext={stepper.next}
        onReset={stepper.reset}
        isPlaying={stepper.isPlaying}
        onToggleAutoPlay={stepper.toggleAutoPlay}
        stepTitle={`Step ${stepper.currentStep + 1}: ${STEP_INFO[stepper.currentStep].title}`}
        stepDescription={STEP_INFO[stepper.currentStep].description}
      />
    </div>
  );
}
