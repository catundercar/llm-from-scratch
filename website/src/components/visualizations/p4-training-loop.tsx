"use client";

import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Training Data",
    description:
      "A batch of token sequences is sampled from the dataset and fed into the model as input.",
  },
  {
    title: "Forward Pass",
    description:
      "Input tokens flow through embedding, transformer blocks, and the LM head to produce logits.",
  },
  {
    title: "Compute Loss",
    description:
      "Cross-entropy loss measures how far predicted token probabilities are from the actual next tokens.",
  },
  {
    title: "Backward Pass",
    description:
      "Gradients of the loss are computed with respect to every parameter via backpropagation.",
  },
  {
    title: "Gradient Clipping",
    description:
      "The global gradient norm is clipped to a maximum value (e.g. 1.0) to prevent exploding gradients.",
  },
  {
    title: "Optimizer Step",
    description:
      "AdamW applies the clipped gradients with a cosine-decayed learning rate to update all weights.",
  },
  {
    title: "Repeat",
    description:
      "The loop repeats for thousands of iterations. The loss curve trends downward over time.",
  },
];

// Loss curve data points (decreasing trend with noise)
const LOSS_POINTS = [
  4.2, 3.8, 3.5, 3.6, 3.2, 3.0, 2.8, 2.9, 2.6, 2.4, 2.5, 2.3, 2.1, 2.0,
  1.9, 1.85, 1.8, 1.78, 1.75, 1.72,
];

// Cosine LR schedule
function cosineSchedule(step: number, total: number): number {
  return 0.5 * (1 + Math.cos(Math.PI * step / total));
}

function buildLossPath(count: number): string {
  if (count < 2) return "";
  const xScale = 260 / (LOSS_POINTS.length - 1);
  const yMin = 1.5;
  const yMax = 4.5;
  const h = 80;
  const parts: string[] = [];
  for (let i = 0; i < count; i++) {
    const x = 20 + i * xScale;
    const y = 10 + h - ((LOSS_POINTS[i] - yMin) / (yMax - yMin)) * h;
    parts.push(i === 0 ? `M${x},${y}` : `L${x},${y}`);
  }
  return parts.join(" ");
}

function buildLRPath(): string {
  const steps = 40;
  const parts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const x = 20 + (i / steps) * 260;
    const y = 10 + 80 - cosineSchedule(i, steps) * 70;
    parts.push(i === 0 ? `M${x},${y}` : `L${x},${y}`);
  }
  return parts.join(" ");
}

// Model block component
function ModelBlock({
  x,
  y,
  width,
  height,
  label,
  active,
  color,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  active: boolean;
  color: string;
}) {
  return (
    <g>
      <motion.rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={6}
        fill={color}
        stroke={active ? "#3b82f6" : "#94a3b8"}
        strokeWidth={active ? 2.5 : 1.5}
        animate={{ opacity: active ? 1 : 0.5 }}
        transition={{ duration: 0.4 }}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={11}
        fontWeight={600}
        fill="white"
      >
        {label}
      </text>
    </g>
  );
}

// Animated arrow between blocks
function FlowArrow({
  x1,
  y1,
  x2,
  y2,
  active,
  reverse,
  color,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active: boolean;
  reverse?: boolean;
  color?: string;
}) {
  const arrowColor = color || (active ? "#3b82f6" : "#94a3b8");
  const startX = reverse ? x2 : x1;
  const startY = reverse ? y2 : y1;
  const endX = reverse ? x1 : x2;
  const endY = reverse ? y1 : y2;

  return (
    <g>
      <motion.line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke={arrowColor}
        strokeWidth={active ? 2.5 : 1.5}
        animate={{ opacity: active ? 1 : 0.3 }}
        transition={{ duration: 0.3 }}
      />
      {/* arrowhead */}
      <motion.polygon
        points={(() => {
          const dx = endX - startX;
          const dy = endY - startY;
          const len = Math.sqrt(dx * dx + dy * dy);
          const ux = dx / len;
          const uy = dy / len;
          const tipX = endX;
          const tipY = endY;
          const baseX = tipX - ux * 8;
          const baseY = tipY - uy * 8;
          const perpX = -uy * 4;
          const perpY = ux * 4;
          return `${tipX},${tipY} ${baseX + perpX},${baseY + perpY} ${baseX - perpX},${baseY - perpY}`;
        })()}
        fill={arrowColor}
        animate={{ opacity: active ? 1 : 0.3 }}
        transition={{ duration: 0.3 }}
      />
    </g>
  );
}

// Animated data dots flowing along a path
function FlowDots({
  x1,
  y1,
  x2,
  y2,
  active,
  reverse,
  color,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active: boolean;
  reverse?: boolean;
  color?: string;
}) {
  if (!active) return null;
  const dotColor = color || "#3b82f6";
  const sX = reverse ? x2 : x1;
  const sY = reverse ? y2 : y1;
  const eX = reverse ? x1 : x2;
  const eY = reverse ? y1 : y2;

  return (
    <>
      {[0, 0.33, 0.66].map((offset, i) => (
        <motion.circle
          key={i}
          r={3}
          fill={dotColor}
          initial={{ cx: sX, cy: sY, opacity: 0 }}
          animate={{ cx: eX, cy: eY, opacity: [0, 1, 1, 0] }}
          transition={{
            duration: 1.5,
            delay: offset * 1.5,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      ))}
    </>
  );
}

export default function TrainingLoopVisualization({
  title,
}: {
  title?: string;
}) {
  const stepper = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });
  const step = stepper.currentStep;

  // Pipeline geometry
  const blocks = [
    { x: 40, y: 55, w: 80, h: 36, label: "Input", color: "#6366f1" },
    { x: 160, y: 55, w: 100, h: 36, label: "Embedding", color: "#8b5cf6" },
    { x: 300, y: 55, w: 120, h: 36, label: "Transformer", color: "#7c3aed" },
    { x: 460, y: 55, w: 80, h: 36, label: "LM Head", color: "#6d28d9" },
    { x: 580, y: 55, w: 80, h: 36, label: "Logits", color: "#5b21b6" },
    { x: 700, y: 55, w: 80, h: 36, label: "Loss", color: "#dc2626" },
  ];

  const forwardActive = step === 1;
  const backwardActive = step === 3;

  // Current "iteration" for the repeat step
  const iteration = 14;
  const lrStep = 14;

  // Gradient norm bar
  const rawNorm = 2.4;
  const clippedNorm = 1.0;
  const showClipped = step >= 4;

  return (
    <div className="flex flex-col gap-4">
      {title && (
        <h3 className="text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      {/* Main pipeline diagram */}
      <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
        <svg
          viewBox="0 0 820 110"
          className="w-full"
          style={{ minWidth: 600 }}
        >
          {/* Blocks */}
          {blocks.map((b, i) => (
            <ModelBlock
              key={i}
              x={b.x}
              y={b.y}
              width={b.w}
              height={b.h}
              label={b.label}
              color={b.color}
              active={
                step === 0
                  ? i === 0
                  : step === 1
                    ? true
                    : step === 2
                      ? i >= 4
                      : step === 3
                        ? true
                        : step === 4
                          ? i >= 1 && i <= 3
                          : step === 5
                            ? i >= 1 && i <= 3
                            : true
              }
            />
          ))}

          {/* Forward arrows */}
          {blocks.slice(0, -1).map((b, i) => (
            <FlowArrow
              key={`fwd-${i}`}
              x1={b.x + b.w}
              y1={b.y + b.h / 2}
              x2={blocks[i + 1].x}
              y2={blocks[i + 1].y + blocks[i + 1].h / 2}
              active={forwardActive}
            />
          ))}

          {/* Forward flow dots */}
          {forwardActive &&
            blocks.slice(0, -1).map((b, i) => (
              <FlowDots
                key={`fdot-${i}`}
                x1={b.x + b.w}
                y1={b.y + b.h / 2}
                x2={blocks[i + 1].x}
                y2={blocks[i + 1].y + blocks[i + 1].h / 2}
                active
              />
            ))}

          {/* Backward arrows (step 3) */}
          {backwardActive &&
            blocks.slice(0, -1).map((b, i) => (
              <FlowArrow
                key={`bwd-${i}`}
                x1={b.x + b.w}
                y1={b.y + b.h + 5}
                x2={blocks[i + 1].x}
                y2={blocks[i + 1].y + blocks[i + 1].h + 5}
                active
                reverse
                color="#ef4444"
              />
            ))}

          {backwardActive &&
            blocks.slice(0, -1).map((b, i) => (
              <FlowDots
                key={`bdot-${i}`}
                x1={b.x + b.w}
                y1={b.y + b.h + 5}
                x2={blocks[i + 1].x}
                y2={blocks[i + 1].y + blocks[i + 1].h + 5}
                active
                reverse
                color="#ef4444"
              />
            ))}

          {/* Step 0: Token sequences */}
          {step === 0 && (
            <g>
              {["[the, cat, sat]", "[a, dog, ran]"].map((seq, i) => (
                <motion.text
                  key={i}
                  x={20}
                  y={20 + i * 16}
                  fontSize={10}
                  fill="#6366f1"
                  fontFamily="var(--font-geist-mono), monospace"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.2 }}
                >
                  {seq}
                </motion.text>
              ))}
            </g>
          )}

          {/* Step 2: Loss annotation */}
          {step === 2 && (
            <g>
              <motion.text
                x={700}
                y={45}
                fontSize={9}
                fill="#dc2626"
                textAnchor="middle"
                fontFamily="var(--font-geist-mono), monospace"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                CE = -log(p)
              </motion.text>
              <motion.text
                x={740}
                y={20}
                fontSize={12}
                fontWeight={700}
                fill="#dc2626"
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
              >
                L = 2.41
              </motion.text>
            </g>
          )}

          {/* Step 4: Gradient clipping bar */}
          {step === 4 && (
            <g>
              <text x={340} y={18} fontSize={9} fill="#94a3b8">
                Gradient Norm
              </text>
              {/* raw */}
              <motion.rect
                x={340}
                y={22}
                height={12}
                rx={3}
                fill="#fbbf24"
                initial={{ width: 0 }}
                animate={{ width: rawNorm * 60 }}
                transition={{ duration: 0.5 }}
              />
              {/* clipped overlay */}
              {showClipped && (
                <motion.rect
                  x={340}
                  y={22}
                  height={12}
                  rx={3}
                  fill="#22c55e"
                  initial={{ width: rawNorm * 60 }}
                  animate={{ width: clippedNorm * 60 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                />
              )}
              {/* clip threshold line */}
              <line
                x1={340 + clippedNorm * 60}
                y1={20}
                x2={340 + clippedNorm * 60}
                y2={38}
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="3,2"
              />
              <text
                x={340 + clippedNorm * 60 + 4}
                y={16}
                fontSize={8}
                fill="#ef4444"
              >
                max=1.0
              </text>
            </g>
          )}

          {/* Step 5: LR annotation */}
          {step === 5 && (
            <motion.text
              x={360}
              y={18}
              fontSize={10}
              fill="#22c55e"
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              lr = {(3e-4 * cosineSchedule(lrStep, 20)).toExponential(2)}
            </motion.text>
          )}

          {/* Step 6: iteration counter */}
          {step === 6 && (
            <motion.text
              x={410}
              y={18}
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fill="#6366f1"
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              Iteration {iteration + 1} / 20
            </motion.text>
          )}
        </svg>
      </div>

      {/* Bottom row: Loss curve + LR curve */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* Loss curve */}
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
          <p className="mb-1 text-xs font-semibold text-[var(--foreground)]">
            Loss Curve
          </p>
          <svg viewBox="0 0 300 110" className="w-full">
            {/* Axes */}
            <line
              x1={20}
              y1={95}
              x2={285}
              y2={95}
              stroke="#64748b"
              strokeWidth={1}
            />
            <line
              x1={20}
              y1={5}
              x2={20}
              y2={95}
              stroke="#64748b"
              strokeWidth={1}
            />
            <text x={2} y={15} fontSize={8} fill="#94a3b8">
              4.0
            </text>
            <text x={2} y={92} fontSize={8} fill="#94a3b8">
              1.5
            </text>
            <text x={140} y={108} fontSize={8} fill="#94a3b8" textAnchor="middle">
              iterations
            </text>

            {/* Full loss path (dimmed) */}
            <path
              d={buildLossPath(LOSS_POINTS.length)}
              fill="none"
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="3,3"
              opacity={step === 6 ? 0.4 : 0}
            />

            {/* Active loss path */}
            <motion.path
              d={buildLossPath(
                step === 6 ? iteration + 1 : Math.min(step * 3 + 1, LOSS_POINTS.length)
              )}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={2}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.8 }}
            />

            {/* Current position marker */}
            {step === 6 && (() => {
              const idx = iteration;
              const xScale = 260 / (LOSS_POINTS.length - 1);
              const yMin = 1.5;
              const yMax = 4.5;
              const cx = 20 + idx * xScale;
              const cy = 10 + 80 - ((LOSS_POINTS[idx] - yMin) / (yMax - yMin)) * 80;
              return (
                <motion.circle
                  cx={cx}
                  cy={cy}
                  r={5}
                  fill="#3b82f6"
                  stroke="white"
                  strokeWidth={2}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3 }}
                />
              );
            })()}
          </svg>
        </div>

        {/* LR Schedule */}
        <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-3">
          <p className="mb-1 text-xs font-semibold text-[var(--foreground)]">
            Learning Rate Schedule
          </p>
          <svg viewBox="0 0 300 110" className="w-full">
            {/* Axes */}
            <line
              x1={20}
              y1={95}
              x2={285}
              y2={95}
              stroke="#64748b"
              strokeWidth={1}
            />
            <line
              x1={20}
              y1={5}
              x2={20}
              y2={95}
              stroke="#64748b"
              strokeWidth={1}
            />
            <text x={2} y={22} fontSize={8} fill="#94a3b8">
              3e-4
            </text>
            <text x={2} y={92} fontSize={8} fill="#94a3b8">
              0
            </text>
            <text x={140} y={108} fontSize={8} fill="#94a3b8" textAnchor="middle">
              iterations
            </text>

            {/* Cosine curve */}
            <path
              d={buildLRPath()}
              fill="none"
              stroke="#22c55e"
              strokeWidth={2}
            />

            {/* Current position marker */}
            {(step === 5 || step === 6) && (() => {
              const progress = step === 6 ? iteration / 20 : lrStep / 20;
              const cx = 20 + progress * 260;
              const cy = 10 + 80 - cosineSchedule(progress * 40, 40) * 70;
              return (
                <motion.circle
                  cx={cx}
                  cy={cy}
                  r={5}
                  fill="#22c55e"
                  stroke="white"
                  strokeWidth={2}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2 }}
                />
              );
            })()}
          </svg>
        </div>
      </div>

      {/* Controls */}
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
