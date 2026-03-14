"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Original Weight Matrix",
    description:
      "A fully connected layer has a dense weight matrix W of shape 768×768, totaling 589,824 trainable parameters.",
  },
  {
    title: "Freeze W",
    description:
      "During LoRA fine-tuning, the original weight matrix W is frozen — its parameters are not updated by gradient descent.",
  },
  {
    title: "Add LoRA Matrices",
    description:
      "Two small low-rank matrices are introduced: A (768×8) and B (8×768). Only these are trained, with rank r=8.",
  },
  {
    title: "Low-Rank Product",
    description:
      "The product A×B yields ΔW, a 768×768 matrix. Despite its full size, it only captures a low-rank subspace of changes.",
  },
  {
    title: "Combined: W + ΔW",
    description:
      "At inference time, the effective weight is W + ΔW. The adaptation merges seamlessly with the frozen weights.",
  },
  {
    title: "Parameter Comparison",
    description:
      "LoRA trains only 12,288 parameters (768×8 + 8×768) vs 589,824 for full fine-tuning — a 98% reduction.",
  },
];

const SVG_W = 700;
const SVG_H = 400;

// Colors
const C_FROZEN = "var(--text-tertiary)";
const C_W = "#6366f1";
const C_A = "#f59e0b";
const C_B = "#10b981";
const C_DW = "#ec4899";
const C_COMBINED = "#8b5cf6";

function MatrixRect({
  x,
  y,
  w,
  h,
  fill,
  label,
  dims,
  opacity = 1,
  showGrid = false,
  gridColor = "rgba(255,255,255,0.15)",
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  label: string;
  dims: string;
  opacity?: number;
  showGrid?: boolean;
  gridColor?: string;
}) {
  const gridSpacing = 12;
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.5 }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={4}
        fill={fill}
        opacity={0.85}
        stroke={fill}
        strokeWidth={2}
      />
      {showGrid &&
        Array.from(
          { length: Math.floor(w / gridSpacing) - 1 },
          (_, i) => (
            <line
              key={`v${i}`}
              x1={x + (i + 1) * gridSpacing}
              y1={y}
              x2={x + (i + 1) * gridSpacing}
              y2={y + h}
              stroke={gridColor}
              strokeWidth={0.5}
            />
          )
        )}
      {showGrid &&
        Array.from(
          { length: Math.floor(h / gridSpacing) - 1 },
          (_, i) => (
            <line
              key={`h${i}`}
              x1={x}
              y1={y + (i + 1) * gridSpacing}
              x2={x + w}
              y2={y + (i + 1) * gridSpacing}
              stroke={gridColor}
              strokeWidth={0.5}
            />
          )
        )}
      {/* Label */}
      <text
        x={x + w / 2}
        y={y + h / 2 - 6}
        textAnchor="middle"
        fill="white"
        fontSize={14}
        fontWeight={700}
        fontFamily="var(--font-geist-mono), monospace"
      >
        {label}
      </text>
      {/* Dimensions */}
      <text
        x={x + w / 2}
        y={y + h / 2 + 12}
        textAnchor="middle"
        fill="rgba(255,255,255,0.8)"
        fontSize={11}
        fontFamily="var(--font-geist-mono), monospace"
      >
        {dims}
      </text>
    </motion.g>
  );
}

function LockIcon({ x, y }: { x: number; y: number }) {
  return (
    <motion.g
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <rect
        x={x - 12}
        y={y}
        width={24}
        height={18}
        rx={3}
        fill="#94a3b8"
        stroke="#64748b"
        strokeWidth={1.5}
      />
      <path
        d={`M${x - 7} ${y} V${y - 8} a7 7 0 0 1 14 0 V${y}`}
        fill="none"
        stroke="#64748b"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <circle cx={x} cy={y + 8} r={2.5} fill="#64748b" />
    </motion.g>
  );
}

function MultiplySign({ x, y }: { x: number; y: number }) {
  return (
    <motion.text
      x={x}
      y={y}
      textAnchor="middle"
      fill="var(--foreground)"
      fontSize={20}
      fontWeight={700}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      ×
    </motion.text>
  );
}

function EqualsSign({ x, y }: { x: number; y: number }) {
  return (
    <motion.text
      x={x}
      y={y}
      textAnchor="middle"
      fill="var(--foreground)"
      fontSize={20}
      fontWeight={700}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      =
    </motion.text>
  );
}

function PlusSign({ x, y }: { x: number; y: number }) {
  return (
    <motion.text
      x={x}
      y={y}
      textAnchor="middle"
      fill="var(--foreground)"
      fontSize={20}
      fontWeight={700}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      +
    </motion.text>
  );
}

function ParamBar({
  x,
  y,
  width,
  maxWidth,
  height,
  fill,
  label,
  value,
}: {
  x: number;
  y: number;
  width: number;
  maxWidth: number;
  height: number;
  fill: string;
  label: string;
  value: string;
}) {
  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
      <text
        x={x}
        y={y - 6}
        fill="var(--foreground)"
        fontSize={12}
        fontWeight={600}
        fontFamily="var(--font-geist-mono), monospace"
      >
        {label}
      </text>
      <rect x={x} y={y} width={maxWidth} height={height} rx={4} fill="var(--bg-secondary)" opacity={0.5} />
      <motion.rect
        x={x}
        y={y}
        height={height}
        rx={4}
        fill={fill}
        initial={{ width: 0 }}
        animate={{ width }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
      <text
        x={x + maxWidth + 8}
        y={y + height / 2 + 4}
        fill="var(--foreground)"
        fontSize={11}
        fontFamily="var(--font-geist-mono), monospace"
      >
        {value}
      </text>
    </motion.g>
  );
}

export default function P7LoraDecomposition({ title }: { title?: string }) {
  const stepper = useSteppedVisualization({ totalSteps: STEP_INFO.length, autoPlayInterval: 2500 });
  const step = stepper.currentStep;

  // Layout constants
  const wMatX = 60;
  const wMatY = 80;
  const wMatW = 160;
  const wMatH = 160;
  const wCenter = wMatX + wMatW / 2;

  // LoRA matrices positions (step 2+)
  const aX = 290;
  const aY = 80;
  const aW = 20; // thin: 768×r
  const aH = 160;

  const bX = 360;
  const bY = 140;
  const bW = 160;
  const bH = 20; // thin: r×768

  // ΔW position
  const dwX = 420;
  const dwY = 80;
  const dwW = 160;
  const dwH = 160;

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4 sm:p-6">
      {title && (
        <h3 className="text-lg font-semibold text-[var(--foreground)]">{title}</h3>
      )}

      <div className="relative w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="mx-auto w-full max-w-[700px]"
          style={{ minHeight: 280 }}
        >
          <AnimatePresence mode="wait">
            {/* Step 0: Original Weight Matrix */}
            {step === 0 && (
              <motion.g key="s0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <MatrixRect
                  x={SVG_W / 2 - 90}
                  y={60}
                  w={180}
                  h={180}
                  fill={C_W}
                  label="W"
                  dims="768 × 768"
                  showGrid
                />
                <text
                  x={SVG_W / 2}
                  y={280}
                  textAnchor="middle"
                  fill="var(--foreground)"
                  fontSize={13}
                  fontFamily="var(--font-geist-mono), monospace"
                >
                  589,824 trainable parameters
                </text>
              </motion.g>
            )}

            {/* Step 1: Freeze W */}
            {step === 1 && (
              <motion.g key="s1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <MatrixRect
                  x={SVG_W / 2 - 90}
                  y={60}
                  w={180}
                  h={180}
                  fill="#64748b"
                  label="W"
                  dims="768 × 768"
                  showGrid
                  gridColor="rgba(255,255,255,0.08)"
                />
                <LockIcon x={SVG_W / 2} y={36} />
                <text
                  x={SVG_W / 2}
                  y={275}
                  textAnchor="middle"
                  fill={C_FROZEN}
                  fontSize={13}
                  fontFamily="var(--font-geist-mono), monospace"
                >
                  589,824 params (frozen)
                </text>
                <motion.text
                  x={SVG_W / 2}
                  y={300}
                  textAnchor="middle"
                  fill="#ef4444"
                  fontSize={12}
                  fontWeight={600}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  No gradient updates
                </motion.text>
              </motion.g>
            )}

            {/* Step 2: Add LoRA Matrices */}
            {step === 2 && (
              <motion.g key="s2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {/* Frozen W */}
                <MatrixRect
                  x={wMatX}
                  y={wMatY}
                  w={wMatW}
                  h={wMatH}
                  fill="#64748b"
                  label="W"
                  dims="768×768"
                  showGrid
                  gridColor="rgba(255,255,255,0.08)"
                />
                <LockIcon x={wCenter} y={wMatY - 28} />

                {/* A matrix */}
                <MatrixRect
                  x={aX}
                  y={aY}
                  w={aW}
                  h={aH}
                  fill={C_A}
                  label=""
                  dims=""
                />
                <text x={aX + aW / 2} y={aY - 10} textAnchor="middle" fill={C_A} fontSize={13} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">A</text>
                <text x={aX + aW / 2} y={aY + aH + 18} textAnchor="middle" fill="var(--foreground)" fontSize={10} fontFamily="var(--font-geist-mono), monospace">768×8</text>

                {/* B matrix */}
                <MatrixRect
                  x={bX}
                  y={bY}
                  w={bW}
                  h={bH}
                  fill={C_B}
                  label=""
                  dims=""
                />
                <text x={bX + bW / 2} y={bY - 10} textAnchor="middle" fill={C_B} fontSize={13} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">B</text>
                <text x={bX + bW / 2} y={bY + bH + 18} textAnchor="middle" fill="var(--foreground)" fontSize={10} fontFamily="var(--font-geist-mono), monospace">8×768</text>

                <text x={350} y={340} textAnchor="middle" fill="var(--foreground)" fontSize={12} fontFamily="var(--font-geist-mono), monospace">
                  r = 8 (low rank)
                </text>
                <motion.text
                  x={350}
                  y={362}
                  textAnchor="middle"
                  fill={C_A}
                  fontSize={12}
                  fontWeight={600}
                  fontFamily="var(--font-geist-mono), monospace"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  Trainable: 6,144 + 6,144 = 12,288
                </motion.text>
              </motion.g>
            )}

            {/* Step 3: Low-Rank Product */}
            {step === 3 && (
              <motion.g key="s3" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {/* A */}
                <MatrixRect x={80} y={100} w={24} h={160} fill={C_A} label="" dims="" />
                <text x={92} y={88} textAnchor="middle" fill={C_A} fontSize={13} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">A</text>
                <text x={92} y={280} textAnchor="middle" fill="var(--foreground)" fontSize={10} fontFamily="var(--font-geist-mono), monospace">768×8</text>

                <MultiplySign x={128} y={186} />

                {/* B */}
                <MatrixRect x={148} y={155} w={160} h={24} fill={C_B} label="" dims="" />
                <text x={228} y={148} textAnchor="middle" fill={C_B} fontSize={13} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">B</text>
                <text x={228} y={200} textAnchor="middle" fill="var(--foreground)" fontSize={10} fontFamily="var(--font-geist-mono), monospace">8×768</text>

                <EqualsSign x={340} y={186} />

                {/* ΔW */}
                <MatrixRect
                  x={370}
                  y={100}
                  w={160}
                  h={160}
                  fill={C_DW}
                  label="ΔW"
                  dims="768×768"
                  opacity={0.7}
                  showGrid
                  gridColor="rgba(255,255,255,0.1)"
                />

                {/* Animated flow lines */}
                {[0, 1, 2].map((i) => (
                  <motion.line
                    key={`flow${i}`}
                    x1={104}
                    y1={140 + i * 40}
                    x2={370}
                    y2={140 + i * 40}
                    stroke={C_DW}
                    strokeWidth={1.5}
                    strokeDasharray="4,4"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.5 }}
                    transition={{ delay: 0.3 + i * 0.15, duration: 0.6 }}
                  />
                ))}

                <text x={350} y={300} textAnchor="middle" fill="var(--foreground)" fontSize={12} fontFamily="var(--font-geist-mono), monospace">
                  Low-rank approximation of weight update
                </text>
              </motion.g>
            )}

            {/* Step 4: Combined W + ΔW */}
            {step === 4 && (
              <motion.g key="s4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {/* W frozen */}
                <MatrixRect x={60} y={100} w={150} h={150} fill="#64748b" label="W" dims="frozen" showGrid gridColor="rgba(255,255,255,0.08)" />

                <PlusSign x={240} y={180} />

                {/* ΔW */}
                <MatrixRect x={270} y={100} w={150} h={150} fill={C_DW} label="ΔW" dims="A×B" opacity={0.7} showGrid gridColor="rgba(255,255,255,0.1)" />

                <EqualsSign x={450} y={180} />

                {/* W_eff */}
                <MatrixRect x={480} y={100} w={150} h={150} fill={C_COMBINED} label="W_eff" dims="768×768" showGrid />

                {/* Arrow showing forward pass */}
                <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
                  <text x={555} y={90} textAnchor="middle" fill={C_COMBINED} fontSize={11} fontWeight={600} fontFamily="var(--font-geist-mono), monospace">
                    Used in
                  </text>
                  <text x={555} y={280} textAnchor="middle" fill={C_COMBINED} fontSize={11} fontWeight={600} fontFamily="var(--font-geist-mono), monospace">
                    forward pass
                  </text>
                </motion.g>

                <text x={350} y={310} textAnchor="middle" fill="var(--foreground)" fontSize={12} fontFamily="var(--font-geist-mono), monospace">
                  h = (W + ΔW)x = Wx + ΔWx
                </text>
              </motion.g>
            )}

            {/* Step 5: Parameter Comparison */}
            {step === 5 && (
              <motion.g key="s5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <text x={350} y={40} textAnchor="middle" fill="var(--foreground)" fontSize={15} fontWeight={700}>
                  Parameter Comparison
                </text>

                <ParamBar
                  x={80}
                  y={80}
                  width={480}
                  maxWidth={480}
                  height={40}
                  fill={C_W}
                  label="Full Fine-Tuning"
                  value="589,824"
                />

                <ParamBar
                  x={80}
                  y={160}
                  width={Math.round(480 * (12288 / 589824))}
                  maxWidth={480}
                  height={40}
                  fill={C_A}
                  label="LoRA (r=8)"
                  value="12,288"
                />

                {/* Reduction badge */}
                <motion.g
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.6, type: "spring" }}
                >
                  <rect x={250} y={230} width={200} height={50} rx={25} fill="#10b981" opacity={0.15} />
                  <rect x={250} y={230} width={200} height={50} rx={25} fill="none" stroke="#10b981" strokeWidth={2} />
                  <text x={350} y={252} textAnchor="middle" fill="#10b981" fontSize={18} fontWeight={800} fontFamily="var(--font-geist-mono), monospace">
                    98% fewer
                  </text>
                  <text x={350} y={270} textAnchor="middle" fill="#10b981" fontSize={12} fontFamily="var(--font-geist-mono), monospace">
                    trainable parameters
                  </text>
                </motion.g>

                {/* Breakdown */}
                <text x={350} y={320} textAnchor="middle" fill="var(--text-tertiary)" fontSize={11} fontFamily="var(--font-geist-mono), monospace">
                  A: 768×8 = 6,144 + B: 8×768 = 6,144 = 12,288 total
                </text>
              </motion.g>
            )}
          </AnimatePresence>
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
        stepTitle={STEP_INFO[stepper.currentStep].title}
        stepDescription={STEP_INFO[stepper.currentStep].description}
      />
    </div>
  );
}
