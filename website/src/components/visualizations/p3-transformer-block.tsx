"use client";

import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Step 1: Input",
    description:
      "The input tensor (token embeddings + positional encodings) enters the Transformer block from the top.",
  },
  {
    title: "Step 2: Layer Norm 1",
    description:
      "The input is normalized using Layer Normalization — centering and scaling each token's features independently.",
  },
  {
    title: "Step 3: Multi-Head Attention",
    description:
      "The normalized input is projected into Q, K, V and fed through multi-head self-attention to capture token relationships.",
  },
  {
    title: "Step 4: Residual Connection 1",
    description:
      "The attention output is added back to the original input via a skip connection, preserving gradient flow.",
  },
  {
    title: "Step 5: Layer Norm 2",
    description:
      "The residual sum is normalized again before entering the feed-forward network.",
  },
  {
    title: "Step 6: Feed-Forward Network",
    description:
      "A two-layer MLP with GELU activation expands and contracts the representation (e.g. 768 → 3072 → 768).",
  },
  {
    title: "Step 7: Residual Connection 2",
    description:
      "The FFN output is added to the input of this sub-block via a second skip connection, producing the final output.",
  },
];

// Block definitions for the vertical flow diagram
interface Block {
  id: string;
  label: string;
  sublabel?: string;
  activeStep: number; // Step index where this block is highlighted
  color: string; // Highlight color
}

const BLOCKS: Block[] = [
  { id: "input", label: "Input", sublabel: "x", activeStep: 0, color: "#6366f1" },
  { id: "ln1", label: "Layer Norm 1", sublabel: "μ=0, σ=1", activeStep: 1, color: "#8b5cf6" },
  { id: "mha", label: "Multi-Head Attention", sublabel: "Q, K, V", activeStep: 2, color: "#3b82f6" },
  { id: "add1", label: "Add", sublabel: "x + attn(x)", activeStep: 3, color: "#10b981" },
  { id: "ln2", label: "Layer Norm 2", sublabel: "μ=0, σ=1", activeStep: 4, color: "#8b5cf6" },
  { id: "ffn", label: "Feed-Forward", sublabel: "GELU", activeStep: 5, color: "#f59e0b" },
  { id: "add2", label: "Add", sublabel: "x + ffn(x)", activeStep: 6, color: "#10b981" },
];

// Layout constants
const BOX_WIDTH = 200;
const BOX_HEIGHT = 48;
const BOX_GAP = 24;
const LEFT_MARGIN = 100;
const TOP_MARGIN = 30;
const SVG_WIDTH = 420;
const ARROW_HEAD_SIZE = 6;

function getBoxY(index: number): number {
  return TOP_MARGIN + index * (BOX_HEIGHT + BOX_GAP);
}

function getBoxCenterX(): number {
  return LEFT_MARGIN + BOX_WIDTH / 2;
}

const SVG_HEIGHT = TOP_MARGIN + BLOCKS.length * (BOX_HEIGHT + BOX_GAP) + 20;

// Residual connection paths (curved arcs on the left side)
function getResidualPath(fromIndex: number, toIndex: number): string {
  const fromY = getBoxY(fromIndex) + BOX_HEIGHT / 2;
  const toY = getBoxY(toIndex) + BOX_HEIGHT / 2;
  const x = LEFT_MARGIN;
  const arcX = x - 50;

  return `M ${x} ${fromY} C ${arcX} ${fromY}, ${arcX} ${toY}, ${x} ${toY}`;
}

export default function TransformerBlock({ title }: { title?: string }) {
  const viz = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });

  const step = viz.currentStep;

  // Determine which data-flow arrows are "active" (data has passed through)
  const activeArrowUpTo = step; // Arrows up to this block index are filled

  return (
    <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5">
      {title && (
        <h3 className="mb-4 text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      <div className="flex justify-center overflow-x-auto">
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="w-full"
          style={{ maxWidth: 420, maxHeight: SVG_HEIGHT }}
        >
          {/* Residual connection 1: input → add1 (index 0 → 3) */}
          <motion.path
            d={getResidualPath(0, 3)}
            fill="none"
            strokeWidth={2}
            strokeDasharray="6 4"
            strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{
              pathLength: step >= 3 ? 1 : 0,
              opacity: step >= 3 ? 1 : 0.15,
              stroke: step === 3 ? "#10b981" : "var(--text-tertiary)",
            }}
            transition={{ duration: 0.6 }}
          />

          {/* Residual connection 2: add1 → add2 (index 3 → 6) */}
          <motion.path
            d={getResidualPath(3, 6)}
            fill="none"
            strokeWidth={2}
            strokeDasharray="6 4"
            strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{
              pathLength: step >= 6 ? 1 : 0,
              opacity: step >= 6 ? 1 : 0.15,
              stroke: step === 6 ? "#10b981" : "var(--text-tertiary)",
            }}
            transition={{ duration: 0.6 }}
          />

          {/* Blocks and arrows */}
          {BLOCKS.map((block, i) => {
            const y = getBoxY(i);
            const cx = getBoxCenterX();
            const isActive = step === block.activeStep;
            const isPast = step > block.activeStep;

            return (
              <motion.g key={block.id}>
                {/* Glow effect behind active block */}
                {isActive && (
                  <motion.rect
                    x={LEFT_MARGIN - 4}
                    y={y - 4}
                    width={BOX_WIDTH + 8}
                    height={BOX_HEIGHT + 8}
                    rx={14}
                    fill="none"
                    stroke={block.color}
                    strokeWidth={2}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0.4, 0.8, 0.4] }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />
                )}

                {/* Box */}
                <motion.rect
                  x={LEFT_MARGIN}
                  y={y}
                  width={BOX_WIDTH}
                  height={BOX_HEIGHT}
                  rx={10}
                  initial={{ fill: "var(--bg-primary)" }}
                  animate={{
                    fill: isActive
                      ? block.color
                      : "var(--bg-primary)",
                    stroke: isActive
                      ? block.color
                      : isPast
                        ? block.color
                        : "var(--border)",
                    strokeWidth: isActive ? 2.5 : 1.5,
                    opacity: isActive || isPast ? 1 : 0.5,
                  }}
                  transition={{ duration: 0.3 }}
                />

                {/* Label */}
                <motion.text
                  x={cx}
                  y={y + (block.sublabel ? BOX_HEIGHT / 2 - 7 : BOX_HEIGHT / 2)}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={13}
                  fontWeight={600}
                  fontFamily="var(--font-geist-mono), monospace"
                  animate={{
                    fill: isActive ? "#ffffff" : "var(--foreground)",
                    opacity: isActive || isPast ? 1 : 0.5,
                  }}
                  transition={{ duration: 0.3 }}
                >
                  {block.label}
                </motion.text>

                {/* Sublabel */}
                {block.sublabel && (
                  <motion.text
                    x={cx}
                    y={y + BOX_HEIGHT / 2 + 10}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize={10}
                    fontFamily="var(--font-geist-mono), monospace"
                    animate={{
                      fill: isActive
                        ? "rgba(255,255,255,0.8)"
                        : "var(--text-tertiary)",
                      opacity: isActive || isPast ? 1 : 0.4,
                    }}
                    transition={{ duration: 0.3 }}
                  >
                    {block.sublabel}
                  </motion.text>
                )}

                {/* Q/K/V labels inside MHA when active */}
                {block.id === "mha" && isActive && (
                  <>
                    {["Q", "K", "V"].map((letter, li) => {
                      const lx = LEFT_MARGIN + 36 + li * 64;
                      return (
                        <motion.g key={letter}>
                          <motion.rect
                            x={lx}
                            y={y + 8}
                            width={32}
                            height={BOX_HEIGHT - 16}
                            rx={6}
                            fill="rgba(255,255,255,0.2)"
                            initial={{ scaleX: 0 }}
                            animate={{ scaleX: 1 }}
                            transition={{ delay: 0.15 + li * 0.1 }}
                          />
                          <motion.text
                            x={lx + 16}
                            y={y + BOX_HEIGHT / 2}
                            textAnchor="middle"
                            dominantBaseline="central"
                            fontSize={12}
                            fontWeight={700}
                            fill="#ffffff"
                            fontFamily="var(--font-geist-mono), monospace"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 + li * 0.1 }}
                          >
                            {letter}
                          </motion.text>
                        </motion.g>
                      );
                    })}
                  </>
                )}

                {/* FFN expansion indicator when active */}
                {block.id === "ffn" && isActive && (
                  <motion.g>
                    {/* Small expansion arrow label to the right */}
                    <motion.text
                      x={LEFT_MARGIN + BOX_WIDTH + 12}
                      y={y + BOX_HEIGHT / 2}
                      dominantBaseline="central"
                      fontSize={9}
                      fill="var(--text-tertiary)"
                      fontFamily="var(--font-geist-mono), monospace"
                      initial={{ opacity: 0, x: LEFT_MARGIN + BOX_WIDTH + 5 }}
                      animate={{ opacity: 1, x: LEFT_MARGIN + BOX_WIDTH + 12 }}
                    >
                      768→3072→768
                    </motion.text>
                  </motion.g>
                )}

                {/* Downward arrow to next block */}
                {i < BLOCKS.length - 1 && (
                  <motion.g>
                    <motion.line
                      x1={cx}
                      y1={y + BOX_HEIGHT}
                      x2={cx}
                      y2={y + BOX_HEIGHT + BOX_GAP - ARROW_HEAD_SIZE}
                      strokeWidth={2}
                      strokeLinecap="round"
                      animate={{
                        stroke:
                          i < activeArrowUpTo
                            ? block.color
                            : "var(--text-tertiary)",
                        opacity: i < activeArrowUpTo ? 1 : 0.2,
                      }}
                      transition={{ duration: 0.3 }}
                    />
                    {/* Arrow head */}
                    <motion.polygon
                      points={`${cx - ARROW_HEAD_SIZE},${y + BOX_HEIGHT + BOX_GAP - ARROW_HEAD_SIZE} ${cx + ARROW_HEAD_SIZE},${y + BOX_HEIGHT + BOX_GAP - ARROW_HEAD_SIZE} ${cx},${y + BOX_HEIGHT + BOX_GAP}`}
                      animate={{
                        fill:
                          i < activeArrowUpTo
                            ? block.color
                            : "var(--text-tertiary)",
                        opacity: i < activeArrowUpTo ? 1 : 0.2,
                      }}
                      transition={{ duration: 0.3 }}
                    />
                  </motion.g>
                )}

                {/* Data flow pulse animation on active arrow */}
                {i < BLOCKS.length - 1 && i === activeArrowUpTo - 1 && (
                  <motion.circle
                    cx={cx}
                    r={3}
                    fill={block.color}
                    initial={{ cy: y + BOX_HEIGHT, opacity: 1 }}
                    animate={{
                      cy: y + BOX_HEIGHT + BOX_GAP,
                      opacity: [1, 1, 0],
                    }}
                    transition={{
                      duration: 0.8,
                      repeat: Infinity,
                      ease: "easeIn",
                    }}
                  />
                )}
              </motion.g>
            );
          })}

          {/* Output label after last block */}
          {step === 6 && (
            <motion.text
              x={getBoxCenterX()}
              y={getBoxY(BLOCKS.length - 1) + BOX_HEIGHT + BOX_GAP - 2}
              textAnchor="middle"
              fontSize={12}
              fontWeight={600}
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              fill="#10b981"
            >
              Output
            </motion.text>
          )}

          {/* Residual connection labels */}
          {step >= 3 && (
            <motion.text
              x={LEFT_MARGIN - 55}
              y={(getBoxY(0) + getBoxY(3)) / 2 + BOX_HEIGHT / 2}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={8}
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0 }}
              animate={{
                opacity: 0.7,
                fill: step === 3 ? "#10b981" : "var(--text-tertiary)",
              }}
            >
              skip
            </motion.text>
          )}

          {step >= 6 && (
            <motion.text
              x={LEFT_MARGIN - 55}
              y={(getBoxY(3) + getBoxY(6)) / 2 + BOX_HEIGHT / 2}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={8}
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0 }}
              animate={{
                opacity: 0.7,
                fill: step === 6 ? "#10b981" : "var(--text-tertiary)",
              }}
            >
              skip
            </motion.text>
          )}
        </svg>
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
