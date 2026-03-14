"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Base Model",
    description:
      "A pre-trained GPT model can generate text, but its outputs are unstructured and may not follow instructions reliably.",
  },
  {
    title: "SFT Training Data",
    description:
      "Supervised Fine-Tuning uses curated instruction-response pairs to teach the model the expected conversational format.",
  },
  {
    title: "After SFT",
    description:
      "The model now follows the instruction format and produces structured, helpful responses — but quality can still vary.",
  },
  {
    title: "DPO Training Data",
    description:
      "Direct Preference Optimization uses pairs of chosen (good) and rejected (bad) responses for the same prompt.",
  },
  {
    title: "DPO Optimization",
    description:
      "The model learns to increase the probability of chosen responses relative to rejected ones, without a separate reward model.",
  },
  {
    title: "Final Aligned Model",
    description:
      "The complete pipeline: Pre-trained → SFT → DPO produces a model that is both instruction-following and preference-aligned.",
  },
];

const SVG_W = 700;
const SVG_H = 400;

const C_BASE = "#64748b";
const C_SFT = "#3b82f6";
const C_DPO = "#8b5cf6";
const C_CHOSEN = "#10b981";
const C_REJECTED = "#ef4444";
const C_FINAL = "#06b6d4";

function ModelBox({
  x,
  y,
  w,
  h,
  fill,
  label,
  sublabel,
  glow = false,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  label: string;
  sublabel?: string;
  glow?: boolean;
}) {
  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.85 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      {glow && (
        <rect
          x={x - 4}
          y={y - 4}
          width={w + 8}
          height={h + 8}
          rx={14}
          fill="none"
          stroke={fill}
          strokeWidth={2}
          opacity={0.4}
        />
      )}
      <rect x={x} y={y} width={w} height={h} rx={10} fill={fill} opacity={0.15} />
      <rect x={x} y={y} width={w} height={h} rx={10} fill="none" stroke={fill} strokeWidth={2} />
      {/* Neural network icon lines */}
      {[0, 1, 2].map((row) =>
        [0, 1, 2].map((col) => (
          <circle
            key={`n${row}${col}`}
            cx={x + w / 2 - 16 + col * 16}
            cy={y + 20 + row * 14}
            r={3}
            fill={fill}
            opacity={0.5}
          />
        ))
      )}
      <text
        x={x + w / 2}
        y={y + h / 2 + 18}
        textAnchor="middle"
        fill="var(--foreground)"
        fontSize={13}
        fontWeight={700}
        fontFamily="var(--font-geist-mono), monospace"
      >
        {label}
      </text>
      {sublabel && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 34}
          textAnchor="middle"
          fill="var(--text-tertiary)"
          fontSize={10}
          fontFamily="var(--font-geist-mono), monospace"
        >
          {sublabel}
        </text>
      )}
    </motion.g>
  );
}

function TextCard({
  x,
  y,
  w,
  h,
  lines,
  borderColor,
  delay = 0,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  lines: { text: string; color: string; bold?: boolean }[];
  borderColor: string;
  delay?: number;
}) {
  return (
    <motion.g
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <rect x={x} y={y} width={w} height={h} rx={6} fill="var(--bg-secondary)" opacity={0.8} />
      <rect x={x} y={y} width={w} height={h} rx={6} fill="none" stroke={borderColor} strokeWidth={1.5} />
      {lines.map((line, i) => (
        <text
          key={i}
          x={x + 10}
          y={y + 16 + i * 16}
          fill={line.color}
          fontSize={10}
          fontWeight={line.bold ? 700 : 400}
          fontFamily="var(--font-geist-mono), monospace"
        >
          {line.text}
        </text>
      ))}
    </motion.g>
  );
}

function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  delay = 0,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  delay?: number;
}) {
  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay, duration: 0.3 }}
    >
      <defs>
        <marker
          id={`arrowhead-${color.replace("#", "")}`}
          markerWidth="8"
          markerHeight="6"
          refX="8"
          refY="3"
          orient="auto"
        >
          <polygon points="0 0, 8 3, 0 6" fill={color} />
        </marker>
      </defs>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeWidth={2}
        markerEnd={`url(#arrowhead-${color.replace("#", "")})`}
      />
    </motion.g>
  );
}

function QualityMeter({
  x,
  y,
  w,
  level,
  label,
  color,
}: {
  x: number;
  y: number;
  w: number;
  level: number; // 0-1
  label: string;
  color: string;
}) {
  const barH = 14;
  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
      <text x={x} y={y - 6} fill="var(--foreground)" fontSize={10} fontWeight={600} fontFamily="var(--font-geist-mono), monospace">
        {label}
      </text>
      <rect x={x} y={y} width={w} height={barH} rx={7} fill="var(--bg-secondary)" opacity={0.6} />
      <motion.rect
        x={x}
        y={y}
        height={barH}
        rx={7}
        fill={color}
        initial={{ width: 0 }}
        animate={{ width: w * level }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
    </motion.g>
  );
}

function CheckIcon({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={10} fill={C_CHOSEN} opacity={0.2} />
      <path d={`M${x - 5} ${y} l3 4 l7 -8`} stroke={C_CHOSEN} strokeWidth={2.5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
}

function XIcon({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={10} fill={C_REJECTED} opacity={0.2} />
      <path d={`M${x - 4} ${y - 4} l8 8 M${x + 4} ${y - 4} l-8 8`} stroke={C_REJECTED} strokeWidth={2.5} fill="none" strokeLinecap="round" />
    </g>
  );
}

export default function P8SftDpoComparison({ title }: { title?: string }) {
  const stepper = useSteppedVisualization({ totalSteps: STEP_INFO.length, autoPlayInterval: 3000 });
  const step = stepper.currentStep;

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
            {/* Step 0: Base Model */}
            {step === 0 && (
              <motion.g key="s0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ModelBox x={250} y={60} w={200} h={100} fill={C_BASE} label="Pre-trained GPT" sublabel="next-token prediction" />

                {/* Random outputs */}
                <Arrow x1={350} y1={165} x2={350} y2={200} color={C_BASE} delay={0.3} />
                <TextCard
                  x={180}
                  y={210}
                  w={340}
                  h={110}
                  borderColor={C_BASE}
                  delay={0.4}
                  lines={[
                    { text: "Input: What is photosynthesis?", color: "var(--foreground)", bold: true },
                    { text: "", color: "transparent" },
                    { text: "Output: Photosynthesis is a", color: "var(--text-tertiary)" },
                    { text: "word that comes from Greek photo", color: "var(--text-tertiary)" },
                    { text: "meaning light and synthesis which", color: "var(--text-tertiary)" },
                    { text: "  (continues rambling...)", color: C_REJECTED },
                  ]}
                />
              </motion.g>
            )}

            {/* Step 1: SFT Training Data */}
            {step === 1 && (
              <motion.g key="s1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ModelBox x={300} y={80} w={180} h={90} fill={C_SFT} label="GPT" sublabel="being fine-tuned" />

                {/* Training data cards flowing in */}
                <TextCard
                  x={30}
                  y={30}
                  w={240}
                  h={60}
                  borderColor={C_SFT}
                  delay={0.2}
                  lines={[
                    { text: "[User] Explain gravity", color: C_SFT, bold: true },
                    { text: "[Asst] Gravity is a force...", color: "var(--foreground)" },
                  ]}
                />
                <Arrow x1={270} y1={60} x2={300} y2={100} color={C_SFT} delay={0.3} />

                <TextCard
                  x={30}
                  y={110}
                  w={240}
                  h={60}
                  borderColor={C_SFT}
                  delay={0.4}
                  lines={[
                    { text: "[User] Write a haiku", color: C_SFT, bold: true },
                    { text: "[Asst] Silent morning dew...", color: "var(--foreground)" },
                  ]}
                />
                <Arrow x1={270} y1={140} x2={300} y2={140} color={C_SFT} delay={0.5} />

                <TextCard
                  x={30}
                  y={190}
                  w={240}
                  h={60}
                  borderColor={C_SFT}
                  delay={0.6}
                  lines={[
                    { text: "[User] Summarize this text", color: C_SFT, bold: true },
                    { text: "[Asst] The key points are...", color: "var(--foreground)" },
                  ]}
                />
                <Arrow x1={270} y1={220} x2={300} y2={170} color={C_SFT} delay={0.7} />

                <text x={390} y={210} textAnchor="middle" fill={C_SFT} fontSize={12} fontWeight={600} fontFamily="var(--font-geist-mono), monospace">
                  Supervised
                </text>
                <text x={390} y={226} textAnchor="middle" fill={C_SFT} fontSize={12} fontWeight={600} fontFamily="var(--font-geist-mono), monospace">
                  Fine-Tuning
                </text>

                {/* Quality meters */}
                <QualityMeter x={300} y={280} w={200} level={0.3} label="Instruction Following" color={C_BASE} />
                <Arrow x1={500} y1={287} x2={540} y2={287} color={C_SFT} delay={0.5} />
                <QualityMeter x={560} y={280} w={80} level={0.7} label="Target" color={C_SFT} />
              </motion.g>
            )}

            {/* Step 2: After SFT */}
            {step === 2 && (
              <motion.g key="s2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ModelBox x={250} y={40} w={200} h={90} fill={C_SFT} label="SFT Model" sublabel="instruction-tuned" glow />

                <Arrow x1={350} y1={135} x2={350} y2={165} color={C_SFT} delay={0.3} />

                <TextCard
                  x={180}
                  y={175}
                  w={340}
                  h={110}
                  borderColor={C_SFT}
                  delay={0.4}
                  lines={[
                    { text: "Input: What is photosynthesis?", color: "var(--foreground)", bold: true },
                    { text: "", color: "transparent" },
                    { text: "Output: Photosynthesis is the", color: C_SFT },
                    { text: "process by which plants convert", color: C_SFT },
                    { text: "sunlight into chemical energy.", color: C_SFT },
                    { text: "  (structured & helpful!)", color: C_CHOSEN },
                  ]}
                />

                {/* Quality meter */}
                <QualityMeter x={200} y={320} w={300} level={0.6} label="Response Quality" color={C_SFT} />
                <motion.text
                  x={520}
                  y={334}
                  fill="var(--text-tertiary)"
                  fontSize={10}
                  fontFamily="var(--font-geist-mono), monospace"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8 }}
                >
                  Good, but can be better
                </motion.text>
              </motion.g>
            )}

            {/* Step 3: DPO Training Data */}
            {step === 3 && (
              <motion.g key="s3" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <text x={350} y={30} textAnchor="middle" fill="var(--foreground)" fontSize={14} fontWeight={700}>
                  Preference Pairs for DPO
                </text>

                {/* Prompt */}
                <TextCard
                  x={200}
                  y={45}
                  w={300}
                  h={30}
                  borderColor="var(--text-tertiary)"
                  delay={0.1}
                  lines={[{ text: "Prompt: \"Explain quantum computing\"", color: "var(--foreground)", bold: true }]}
                />

                {/* Chosen response */}
                <TextCard
                  x={50}
                  y={100}
                  w={280}
                  h={100}
                  borderColor={C_CHOSEN}
                  delay={0.3}
                  lines={[
                    { text: "Chosen Response ✓", color: C_CHOSEN, bold: true },
                    { text: "", color: "transparent" },
                    { text: "Quantum computing uses qubits", color: "var(--foreground)" },
                    { text: "that can exist in superposition,", color: "var(--foreground)" },
                    { text: "enabling parallel computation...", color: "var(--foreground)" },
                  ]}
                />
                <CheckIcon x={320} y={150} />

                {/* Rejected response */}
                <TextCard
                  x={370}
                  y={100}
                  w={280}
                  h={100}
                  borderColor={C_REJECTED}
                  delay={0.5}
                  lines={[
                    { text: "Rejected Response ✗", color: C_REJECTED, bold: true },
                    { text: "", color: "transparent" },
                    { text: "Quantum computing is basically", color: "var(--text-tertiary)" },
                    { text: "just really fast computers that", color: "var(--text-tertiary)" },
                    { text: "use quantum stuff to work...", color: "var(--text-tertiary)" },
                  ]}
                />
                <XIcon x={660} y={150} />

                {/* Second pair */}
                <TextCard
                  x={200}
                  y={220}
                  w={300}
                  h={30}
                  borderColor="var(--text-tertiary)"
                  delay={0.6}
                  lines={[{ text: "Prompt: \"Tips for better sleep\"", color: "var(--foreground)", bold: true }]}
                />

                <TextCard
                  x={50}
                  y={268}
                  w={280}
                  h={70}
                  borderColor={C_CHOSEN}
                  delay={0.7}
                  lines={[
                    { text: "Chosen ✓", color: C_CHOSEN, bold: true },
                    { text: "Maintain a consistent schedule,", color: "var(--foreground)" },
                    { text: "limit screen time before bed...", color: "var(--foreground)" },
                  ]}
                />

                <TextCard
                  x={370}
                  y={268}
                  w={280}
                  h={70}
                  borderColor={C_REJECTED}
                  delay={0.8}
                  lines={[
                    { text: "Rejected ✗", color: C_REJECTED, bold: true },
                    { text: "Just try to sleep more lol,", color: "var(--text-tertiary)" },
                    { text: "maybe take some pills idk...", color: "var(--text-tertiary)" },
                  ]}
                />
              </motion.g>
            )}

            {/* Step 4: DPO Optimization */}
            {step === 4 && (
              <motion.g key="s4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ModelBox x={250} y={30} w={200} h={80} fill={C_DPO} label="SFT Model" sublabel="+ DPO training" />

                {/* Chosen: probability up */}
                <motion.g
                  initial={{ opacity: 0, x: -30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                >
                  <rect x={80} y={140} width={220} height={70} rx={8} fill={C_CHOSEN} opacity={0.1} stroke={C_CHOSEN} strokeWidth={1.5} />
                  <text x={190} y={162} textAnchor="middle" fill={C_CHOSEN} fontSize={12} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">
                    Chosen Response
                  </text>
                  <text x={190} y={180} textAnchor="middle" fill={C_CHOSEN} fontSize={20} fontWeight={800}>
                    P(y_w) ↑
                  </text>
                  <text x={190} y={200} textAnchor="middle" fill={C_CHOSEN} fontSize={10} fontFamily="var(--font-geist-mono), monospace">
                    Increase probability
                  </text>
                </motion.g>

                {/* Rejected: probability down */}
                <motion.g
                  initial={{ opacity: 0, x: 30 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                >
                  <rect x={400} y={140} width={220} height={70} rx={8} fill={C_REJECTED} opacity={0.1} stroke={C_REJECTED} strokeWidth={1.5} />
                  <text x={510} y={162} textAnchor="middle" fill={C_REJECTED} fontSize={12} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">
                    Rejected Response
                  </text>
                  <text x={510} y={180} textAnchor="middle" fill={C_REJECTED} fontSize={20} fontWeight={800}>
                    P(y_l) ↓
                  </text>
                  <text x={510} y={200} textAnchor="middle" fill={C_REJECTED} fontSize={10} fontFamily="var(--font-geist-mono), monospace">
                    Decrease probability
                  </text>
                </motion.g>

                {/* Arrows from model */}
                <Arrow x1={300} y1={110} x2={190} y2={140} color={C_CHOSEN} delay={0.3} />
                <Arrow x1={400} y1={110} x2={510} y2={140} color={C_REJECTED} delay={0.5} />

                {/* DPO Loss formula */}
                <motion.g
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.7 }}
                >
                  <rect x={150} y={240} width={400} height={50} rx={8} fill="var(--bg-secondary)" opacity={0.6} />
                  <rect x={150} y={240} width={400} height={50} rx={8} fill="none" stroke={C_DPO} strokeWidth={1} strokeDasharray="4,4" />
                  <text x={350} y={262} textAnchor="middle" fill={C_DPO} fontSize={13} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">
                    L_DPO = -log σ(β(log π(y_w) - log π(y_l)))
                  </text>
                  <text x={350} y={280} textAnchor="middle" fill="var(--text-tertiary)" fontSize={10} fontFamily="var(--font-geist-mono), monospace">
                    No reward model needed — direct optimization
                  </text>
                </motion.g>

                {/* Quality improvement */}
                <QualityMeter x={200} y={320} w={300} level={0.6} label="Before DPO" color={C_SFT} />
                <QualityMeter x={200} y={355} w={300} level={0.9} label="After DPO" color={C_DPO} />
              </motion.g>
            )}

            {/* Step 5: Final Aligned Model */}
            {step === 5 && (
              <motion.g key="s5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <text x={350} y={30} textAnchor="middle" fill="var(--foreground)" fontSize={15} fontWeight={700}>
                  Complete Alignment Pipeline
                </text>

                {/* Pipeline: Base → SFT → DPO */}
                <ModelBox x={30} y={60} w={140} h={80} fill={C_BASE} label="Pre-trained" />
                <Arrow x1={170} y1={100} x2={220} y2={100} color={C_BASE} delay={0.2} />

                <ModelBox x={225} y={60} w={140} h={80} fill={C_SFT} label="+ SFT" />
                <Arrow x1={365} y1={100} x2={415} y2={100} color={C_SFT} delay={0.4} />

                <ModelBox x={420} y={60} w={140} h={80} fill={C_DPO} label="+ DPO" />
                <Arrow x1={560} y1={100} x2={600} y2={100} color={C_DPO} delay={0.6} />

                {/* Final model */}
                <motion.g
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.7, type: "spring" }}
                >
                  <rect x={605} y={65} width={70} height={70} rx={35} fill={C_FINAL} opacity={0.15} />
                  <rect x={605} y={65} width={70} height={70} rx={35} fill="none" stroke={C_FINAL} strokeWidth={2.5} />
                  <text x={640} y={96} textAnchor="middle" fill={C_FINAL} fontSize={20}>
                    ★
                  </text>
                  <text x={640} y={112} textAnchor="middle" fill={C_FINAL} fontSize={9} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">
                    Aligned
                  </text>
                </motion.g>

                {/* Quality comparison */}
                <QualityMeter x={80} y={190} w={250} level={0.25} label="Pre-trained: Instruction Following" color={C_BASE} />
                <QualityMeter x={80} y={230} w={250} level={0.65} label="After SFT: Format + Helpfulness" color={C_SFT} />
                <QualityMeter x={80} y={270} w={250} level={0.92} label="After DPO: Preference Aligned" color={C_DPO} />

                {/* Properties */}
                <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}>
                  <rect x={400} y={180} width={260} height={120} rx={8} fill="var(--bg-secondary)" opacity={0.5} />
                  <rect x={400} y={180} width={260} height={120} rx={8} fill="none" stroke={C_FINAL} strokeWidth={1} />
                  <text x={530} y={202} textAnchor="middle" fill={C_FINAL} fontSize={12} fontWeight={700} fontFamily="var(--font-geist-mono), monospace">
                    Final Model Properties
                  </text>
                  {[
                    "Follows instructions",
                    "Structured responses",
                    "Prefers helpful answers",
                    "Avoids harmful content",
                    "Human-preference aligned",
                  ].map((prop, i) => (
                    <g key={i}>
                      <circle cx={420} cy={222 + i * 16} r={3} fill={C_CHOSEN} />
                      <text
                        x={430}
                        y={226 + i * 16}
                        fill="var(--foreground)"
                        fontSize={10}
                        fontFamily="var(--font-geist-mono), monospace"
                      >
                        {prop}
                      </text>
                    </g>
                  ))}
                </motion.g>
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
