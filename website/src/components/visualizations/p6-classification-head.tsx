"use client";

import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Pre-trained Model",
    description:
      "The GPT backbone consists of a token embedding layer followed by stacked Transformer blocks. All layers are active and trained on language modeling.",
  },
  {
    title: "Freeze Backbone",
    description:
      "All backbone layers are frozen — their weights will not be updated during fine-tuning. This preserves the learned representations.",
  },
  {
    title: "Add Classification Head",
    description:
      "A new classification head (linear layer + softmax) is added on top of the frozen backbone to map representations to output classes.",
  },
  {
    title: "Forward Pass",
    description:
      "Input flows through the frozen backbone (dimmed) and then through the classification head (highlighted) to produce class predictions.",
  },
  {
    title: "Train Head Only",
    description:
      "Gradients only flow through the classification head. The frozen backbone receives no gradient updates, dramatically reducing trainable parameters.",
  },
  {
    title: "Comparison",
    description:
      "Feature extraction trains only the head (~0.1% of params). Full fine-tuning updates everything but risks catastrophic forgetting.",
  },
];

const CLASSES = ["Positive", "Negative", "Neutral"];
const CLASS_COLORS = ["#22c55e", "#ef4444", "#f59e0b"];

interface LayerBlockProps {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color: string;
  frozen: boolean;
  dimmed: boolean;
  active: boolean;
}

function LayerBlock({
  x,
  y,
  width,
  height,
  label,
  color,
  frozen,
  dimmed,
  active,
}: LayerBlockProps) {
  return (
    <g>
      <motion.rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={8}
        fill={color}
        stroke={active ? "#06b6d4" : frozen ? "#475569" : "#94a3b8"}
        strokeWidth={active ? 2.5 : 1.5}
        animate={{ opacity: dimmed ? 0.35 : 1 }}
        transition={{ duration: 0.4 }}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 + 1}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={12}
        fontWeight={600}
        fill="white"
      >
        {label}
      </text>

      {/* Lock icon when frozen */}
      {frozen && (
        <motion.g
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          {/* Lock body */}
          <rect
            x={x + width - 22}
            y={y + 6}
            width={14}
            height={10}
            rx={2}
            fill="none"
            stroke="white"
            strokeWidth={1.5}
            opacity={0.8}
          />
          {/* Lock shackle */}
          <path
            d={`M${x + width - 18},${y + 6} V${y + 3} a4,4 0 0 1 6,0 V${y + 6}`}
            fill="none"
            stroke="white"
            strokeWidth={1.5}
            opacity={0.8}
          />
        </motion.g>
      )}
    </g>
  );
}

function VerticalArrow({
  x,
  y1,
  y2,
  active,
  color,
  reverse,
}: {
  x: number;
  y1: number;
  y2: number;
  active: boolean;
  color?: string;
  reverse?: boolean;
}) {
  const arrowColor = color || (active ? "#06b6d4" : "#64748b");
  const startY = reverse ? y2 : y1;
  const endY = reverse ? y1 : y2;
  const dir = endY > startY ? 1 : -1;

  return (
    <g>
      <motion.line
        x1={x}
        y1={startY}
        x2={x}
        y2={endY}
        stroke={arrowColor}
        strokeWidth={active ? 2 : 1.5}
        animate={{ opacity: active ? 1 : 0.4 }}
        transition={{ duration: 0.3 }}
      />
      <motion.polygon
        points={`${x},${endY} ${x - 4},${endY - dir * 8} ${x + 4},${endY - dir * 8}`}
        fill={arrowColor}
        animate={{ opacity: active ? 1 : 0.4 }}
        transition={{ duration: 0.3 }}
      />
    </g>
  );
}

function FlowDots({
  x,
  y1,
  y2,
  active,
  color,
  reverse,
}: {
  x: number;
  y1: number;
  y2: number;
  active: boolean;
  color?: string;
  reverse?: boolean;
}) {
  if (!active) return null;
  const dotColor = color || "#06b6d4";
  const sY = reverse ? y2 : y1;
  const eY = reverse ? y1 : y2;

  return (
    <>
      {[0, 0.35, 0.7].map((offset, i) => (
        <motion.circle
          key={i}
          cx={x}
          r={3}
          fill={dotColor}
          initial={{ cy: sY, opacity: 0 }}
          animate={{ cy: eY, opacity: [0, 1, 1, 0] }}
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

export default function ClassificationHeadVisualization({
  title,
}: {
  title?: string;
}) {
  const stepper = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });
  const step = stepper.currentStep;

  const showHead = step >= 2;
  const frozen = step >= 1;
  const forwardPass = step === 3;
  const trainHead = step === 4;
  const comparison = step === 5;

  // Layer definitions
  const centerX = comparison ? 30 : 100;
  const layerW = 200;
  const layerH = 36;
  const gap = 10;

  const layers = [
    { label: "Token Embedding", color: "#6366f1" },
    { label: "Transformer Block 1", color: "#7c3aed" },
    { label: "Transformer Block 2", color: "#7c3aed" },
    { label: "Transformer Block 3", color: "#7c3aed" },
    { label: "Transformer Block 4", color: "#7c3aed" },
    { label: "Layer Norm", color: "#8b5cf6" },
  ];

  const startY = 30;
  const layerPositions = layers.map((_, i) => startY + i * (layerH + gap));
  const lastBackboneY =
    layerPositions[layerPositions.length - 1] + layerH;
  const headY = lastBackboneY + gap + 8;
  const classY = headY + layerH + 20;

  const svgH = comparison ? 380 : classY + 30;
  const svgW = comparison ? 520 : 400;

  return (
    <div className="flex flex-col gap-4">
      {title && (
        <h3 className="text-lg font-semibold text-[var(--foreground)]">
          {title}
        </h3>
      )}

      <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="mx-auto w-full"
          style={{ minWidth: 320, maxWidth: 560 }}
        >
          {/* Input label */}
          <motion.text
            x={centerX + layerW / 2}
            y={startY - 10}
            textAnchor="middle"
            fontSize={10}
            fill="#94a3b8"
            fontFamily="var(--font-geist-mono), monospace"
            animate={{ opacity: 1 }}
          >
            Input Tokens
          </motion.text>

          {/* Backbone layers */}
          {layers.map((layer, i) => (
            <LayerBlock
              key={i}
              x={centerX}
              y={layerPositions[i]}
              width={layerW}
              height={layerH}
              label={layer.label}
              color={layer.color}
              frozen={frozen && !comparison}
              dimmed={
                (forwardPass || trainHead) && frozen
              }
              active={step === 0}
            />
          ))}

          {/* Frozen label */}
          {frozen && !comparison && (
            <motion.text
              x={centerX - 10}
              y={startY + (layers.length * (layerH + gap)) / 2}
              textAnchor="end"
              fontSize={11}
              fontWeight={600}
              fill="#64748b"
              fontFamily="var(--font-geist-mono), monospace"
              initial={{ opacity: 0, x: centerX - 30 }}
              animate={{ opacity: 1, x: centerX - 10 }}
              transition={{ duration: 0.4 }}
            >
              FROZEN
            </motion.text>
          )}

          {/* Arrows between backbone layers */}
          {layers.slice(0, -1).map((_, i) => (
            <VerticalArrow
              key={`arr-${i}`}
              x={centerX + layerW / 2}
              y1={layerPositions[i] + layerH}
              y2={layerPositions[i + 1]}
              active={forwardPass || step === 0}
            />
          ))}

          {/* Forward flow dots through backbone */}
          {forwardPass &&
            layers.slice(0, -1).map((_, i) => (
              <FlowDots
                key={`fdot-${i}`}
                x={centerX + layerW / 2}
                y1={layerPositions[i] + layerH}
                y2={layerPositions[i + 1]}
                active
                color="#94a3b8"
              />
            ))}

          {/* Classification Head */}
          {showHead && !comparison && (
            <motion.g
              initial={{ opacity: 0, y: -15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
            >
              {/* Arrow from backbone to head */}
              <VerticalArrow
                x={centerX + layerW / 2}
                y1={lastBackboneY}
                y2={headY}
                active={forwardPass || trainHead}
                color="#06b6d4"
              />

              {forwardPass && (
                <FlowDots
                  x={centerX + layerW / 2}
                  y1={lastBackboneY}
                  y2={headY}
                  active
                  color="#06b6d4"
                />
              )}

              {/* Head block */}
              <motion.rect
                x={centerX}
                y={headY}
                width={layerW}
                height={layerH}
                rx={8}
                fill="#06b6d4"
                stroke={forwardPass || trainHead ? "#22d3ee" : "#0891b2"}
                strokeWidth={forwardPass || trainHead ? 2.5 : 1.5}
                animate={{ opacity: 1 }}
              />
              <text
                x={centerX + layerW / 2}
                y={headY + layerH / 2 + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={12}
                fontWeight={700}
                fill="white"
              >
                Classification Head
              </text>

              {/* Output classes */}
              {CLASSES.map((cls, i) => {
                const classX =
                  centerX +
                  (layerW / (CLASSES.length + 1)) * (i + 1) -
                  25;
                return (
                  <g key={cls}>
                    <line
                      x1={centerX + layerW / 2}
                      y1={headY + layerH}
                      x2={classX + 25}
                      y2={classY}
                      stroke={CLASS_COLORS[i]}
                      strokeWidth={1.5}
                      opacity={0.6}
                    />
                    <rect
                      x={classX}
                      y={classY}
                      width={50}
                      height={22}
                      rx={6}
                      fill={CLASS_COLORS[i]}
                      opacity={0.85}
                    />
                    <text
                      x={classX + 25}
                      y={classY + 12}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={9}
                      fontWeight={600}
                      fill="white"
                    >
                      {cls}
                    </text>
                  </g>
                );
              })}
            </motion.g>
          )}

          {/* Gradient arrows for train-head step */}
          {trainHead && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              {/* Gradient arrow through head */}
              <VerticalArrow
                x={centerX + layerW / 2 + 20}
                y1={headY}
                y2={headY + layerH}
                active
                color="#ef4444"
                reverse
              />
              <FlowDots
                x={centerX + layerW / 2 + 20}
                y1={headY + layerH}
                y2={headY}
                active
                color="#ef4444"
                reverse
              />

              {/* No gradient label on backbone */}
              <motion.text
                x={centerX + layerW + 15}
                y={startY + (layers.length * (layerH + gap)) / 2}
                fontSize={10}
                fill="#ef4444"
                fontFamily="var(--font-geist-mono), monospace"
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.8 }}
              >
                no gradients
              </motion.text>

              {/* Gradient label on head */}
              <motion.text
                x={centerX + layerW + 15}
                y={headY + layerH / 2 + 1}
                fontSize={10}
                fontWeight={600}
                fill="#06b6d4"
                fontFamily="var(--font-geist-mono), monospace"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                gradients flow
              </motion.text>
            </motion.g>
          )}

          {/* Step 5: Comparison view */}
          {comparison && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              {/* Divider */}
              <line
                x1={svgW / 2}
                y1={10}
                x2={svgW / 2}
                y2={svgH - 10}
                stroke="#475569"
                strokeWidth={1}
                strokeDasharray="4,4"
              />

              {/* Left: Feature Extraction labels */}
              <text
                x={centerX + layerW / 2}
                y={svgH - 60}
                textAnchor="middle"
                fontSize={12}
                fontWeight={700}
                fill="#06b6d4"
              >
                Feature Extraction
              </text>
              <text
                x={centerX + layerW / 2}
                y={svgH - 44}
                textAnchor="middle"
                fontSize={10}
                fill="#94a3b8"
                fontFamily="var(--font-geist-mono), monospace"
              >
                Trainable: ~50K params
              </text>
              <text
                x={centerX + layerW / 2}
                y={svgH - 30}
                textAnchor="middle"
                fontSize={9}
                fill="#22c55e"
                fontFamily="var(--font-geist-mono), monospace"
              >
                (~0.1% of total)
              </text>

              {/* Left head block */}
              <rect
                x={centerX}
                y={headY}
                width={layerW}
                height={layerH}
                rx={8}
                fill="#06b6d4"
                stroke="#0891b2"
                strokeWidth={1.5}
              />
              <text
                x={centerX + layerW / 2}
                y={headY + layerH / 2 + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={11}
                fontWeight={700}
                fill="white"
              >
                Classification Head
              </text>

              {/* Param bar - feature extraction */}
              <rect
                x={centerX + 30}
                y={svgH - 20}
                width={layerW - 60}
                height={8}
                rx={4}
                fill="#1e293b"
              />
              <motion.rect
                x={centerX + 30}
                y={svgH - 20}
                height={8}
                rx={4}
                fill="#06b6d4"
                initial={{ width: 0 }}
                animate={{ width: (layerW - 60) * 0.05 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              />

              {/* Right: Full Fine-Tuning */}
              {(() => {
                const rightX = svgW / 2 + 30;
                const rightW = 200;

                return (
                  <g>
                    {/* Right side backbone (all active) */}
                    {layers.map((layer, i) => (
                      <g key={`right-${i}`}>
                        <rect
                          x={rightX}
                          y={layerPositions[i]}
                          width={rightW}
                          height={layerH}
                          rx={8}
                          fill={layer.color}
                          stroke="#94a3b8"
                          strokeWidth={1.5}
                        />
                        <text
                          x={rightX + rightW / 2}
                          y={layerPositions[i] + layerH / 2 + 1}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={11}
                          fontWeight={600}
                          fill="white"
                        >
                          {layer.label}
                        </text>
                      </g>
                    ))}

                    {/* Right head */}
                    <rect
                      x={rightX}
                      y={headY}
                      width={rightW}
                      height={layerH}
                      rx={8}
                      fill="#06b6d4"
                      stroke="#0891b2"
                      strokeWidth={1.5}
                    />
                    <text
                      x={rightX + rightW / 2}
                      y={headY + layerH / 2 + 1}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={11}
                      fontWeight={700}
                      fill="white"
                    >
                      Classification Head
                    </text>

                    {/* Arrows between right backbone layers */}
                    {layers.slice(0, -1).map((_, i) => (
                      <g key={`rarr-${i}`}>
                        <line
                          x1={rightX + rightW / 2}
                          y1={layerPositions[i] + layerH}
                          x2={rightX + rightW / 2}
                          y2={layerPositions[i + 1]}
                          stroke="#64748b"
                          strokeWidth={1.5}
                          opacity={0.4}
                        />
                      </g>
                    ))}

                    <text
                      x={rightX + rightW / 2}
                      y={svgH - 60}
                      textAnchor="middle"
                      fontSize={12}
                      fontWeight={700}
                      fill="#a78bfa"
                    >
                      Full Fine-Tuning
                    </text>
                    <text
                      x={rightX + rightW / 2}
                      y={svgH - 44}
                      textAnchor="middle"
                      fontSize={10}
                      fill="#94a3b8"
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      Trainable: ~50M params
                    </text>
                    <text
                      x={rightX + rightW / 2}
                      y={svgH - 30}
                      textAnchor="middle"
                      fontSize={9}
                      fill="#f59e0b"
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      (100% of total)
                    </text>

                    {/* Param bar - full fine-tuning */}
                    <rect
                      x={rightX + 30}
                      y={svgH - 20}
                      width={rightW - 60}
                      height={8}
                      rx={4}
                      fill="#1e293b"
                    />
                    <motion.rect
                      x={rightX + 30}
                      y={svgH - 20}
                      height={8}
                      rx={4}
                      fill="#a78bfa"
                      initial={{ width: 0 }}
                      animate={{ width: rightW - 60 }}
                      transition={{ duration: 0.8, delay: 0.4 }}
                    />
                  </g>
                );
              })()}
            </motion.g>
          )}
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
