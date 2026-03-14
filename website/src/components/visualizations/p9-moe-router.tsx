"use client";

import { motion } from "framer-motion";
import { useSteppedVisualization } from "@/hooks/useSteppedVisualization";
import { StepControls } from "@/components/visualizations/shared/step-controls";

const STEP_INFO = [
  {
    title: "Input Token Hidden States",
    description:
      "Each token in the sequence has a hidden state vector h of shape [hidden_dim]. These vectors are the input to the MoE layer.",
  },
  {
    title: "Router Network",
    description:
      "A small linear layer (the router/gate) takes each token's hidden state and produces logits for N experts: g(h) = W_g · h, shape [num_experts].",
  },
  {
    title: "Top-K Expert Selection",
    description:
      "Softmax is applied to the router logits, then the top-K experts (typically K=2) are selected per token. Each selected expert gets a gating weight.",
  },
  {
    title: "Sparse Expert Activation",
    description:
      "Only the selected experts process the token — the rest stay idle. This is 'conditional computation': each token uses only a fraction of total parameters.",
  },
  {
    title: "Expert Computation",
    description:
      "Each active expert is a standard FFN (Feed-Forward Network). The token's hidden state is processed independently by each selected expert.",
  },
  {
    title: "Weighted Combination",
    description:
      "The outputs from the selected experts are combined using the gating weights: output = Σ(gate_i × expert_i(h)). This is the final MoE layer output.",
  },
  {
    title: "Load Balancing",
    description:
      "An auxiliary loss encourages balanced expert utilization. Without it, the router may collapse to always picking the same experts (expert collapse).",
  },
];

const NUM_EXPERTS = 6;
const TOP_K = 2;
const SELECTED_EXPERTS = [1, 4]; // 0-indexed

const C_TOKEN = "#3b82f6";
const C_ROUTER = "#8b5cf6";
const C_EXPERT_ACTIVE = "#10b981";
const C_EXPERT_IDLE = "#71717a";
const C_GATE = "#f59e0b";
const C_OUTPUT = "#ec4899";

export default function MoeRouterVisualization({
  title,
}: {
  title?: string;
}) {
  const stepper = useSteppedVisualization({
    totalSteps: STEP_INFO.length,
    autoPlayInterval: 2500,
  });
  const step = stepper.currentStep;

  const svgW = 700;
  const svgH = 420;
  const tokenX = 80;
  const tokenY = svgH / 2;
  const routerX = 210;
  const routerY = svgH / 2;
  const expertStartX = 400;
  const expertSpacing = 55;
  const expertStartY = 45;
  const outputX = 600;
  const outputY = svgH / 2;

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
          style={{ minWidth: 360, maxWidth: 700 }}
        >
          {/* Token input */}
          <motion.rect
            x={tokenX - 35}
            y={tokenY - 28}
            width={70}
            height={56}
            rx={8}
            fill={C_TOKEN}
            fillOpacity={step >= 0 ? 0.2 : 0.05}
            stroke={C_TOKEN}
            strokeWidth={step >= 0 ? 2 : 1}
            animate={{ fillOpacity: step >= 0 ? 0.2 : 0.05 }}
          />
          <text
            x={tokenX}
            y={tokenY - 7}
            textAnchor="middle"
            fontSize={12}
            fontWeight="bold"
            fill={C_TOKEN}
            fontFamily="var(--font-geist-mono), monospace"
          >
            Token
          </text>
          <text
            x={tokenX}
            y={tokenY + 10}
            textAnchor="middle"
            fontSize={10}
            fill={C_TOKEN}
            fontFamily="var(--font-geist-mono), monospace"
            opacity={0.7}
          >
            h ∈ ℝᵈ
          </text>

          {/* Arrow: token → router */}
          {step >= 1 && (
            <motion.line
              x1={tokenX + 37}
              y1={tokenY}
              x2={routerX - 42}
              y2={routerY}
              stroke={C_ROUTER}
              strokeWidth={2}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.4 }}
            />
          )}

          {/* Router */}
          {step >= 1 && (
            <motion.g
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            >
              <rect
                x={routerX - 40}
                y={routerY - 35}
                width={80}
                height={70}
                rx={10}
                fill={C_ROUTER}
                fillOpacity={0.2}
                stroke={C_ROUTER}
                strokeWidth={2}
              />
              <text
                x={routerX}
                y={routerY - 10}
                textAnchor="middle"
                fontSize={12}
                fontWeight="bold"
                fill={C_ROUTER}
                fontFamily="var(--font-geist-mono), monospace"
              >
                Router
              </text>
              <text
                x={routerX}
                y={routerY + 8}
                textAnchor="middle"
                fontSize={10}
                fill={C_ROUTER}
                fontFamily="var(--font-geist-mono), monospace"
                opacity={0.7}
              >
                W_g · h
              </text>
            </motion.g>
          )}

          {/* Softmax / Top-K label */}
          {step >= 2 && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3, delay: 0.2 }}
            >
              <rect
                x={routerX + 50}
                y={routerY - 50}
                width={120}
                height={24}
                rx={6}
                fill={C_GATE}
                fillOpacity={0.15}
                stroke={C_GATE}
                strokeWidth={1}
              />
              <text
                x={routerX + 110}
                y={routerY - 35}
                textAnchor="middle"
                fontSize={10}
                fill={C_GATE}
                fontFamily="var(--font-geist-mono), monospace"
                fontWeight="bold"
              >
                softmax → Top-{TOP_K}
              </text>
            </motion.g>
          )}

          {/* Expert boxes */}
          {Array.from({ length: NUM_EXPERTS }, (_, i) => {
            const ey = expertStartY + i * expertSpacing;
            const isSelected = SELECTED_EXPERTS.includes(i);
            const showActive = step >= 3 && isSelected;
            const isComputing = step >= 4 && isSelected;
            const color = showActive ? C_EXPERT_ACTIVE : C_EXPERT_IDLE;

            return (
              <g key={i}>
                {/* Arrow from router to expert */}
                {step >= 2 && (
                  <motion.line
                    x1={routerX + 40}
                    y1={routerY}
                    x2={expertStartX - 42}
                    y2={ey + 20}
                    stroke={isSelected && step >= 2 ? C_GATE : C_EXPERT_IDLE}
                    strokeWidth={isSelected && step >= 2 ? 2 : 0.8}
                    strokeOpacity={isSelected || step < 3 ? 1 : 0.15}
                    strokeDasharray={isSelected ? "0" : "4 3"}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.3, delay: i * 0.05 }}
                  />
                )}

                {/* Expert box */}
                <motion.rect
                  x={expertStartX - 40}
                  y={ey}
                  width={80}
                  height={40}
                  rx={8}
                  fill={color}
                  fillOpacity={showActive ? 0.2 : 0.05}
                  stroke={color}
                  strokeWidth={showActive ? 2 : 1}
                  strokeDasharray={showActive ? "0" : "4 3"}
                  animate={{
                    fillOpacity: isComputing ? 0.35 : showActive ? 0.2 : 0.05,
                  }}
                  transition={{ duration: 0.3 }}
                />
                <text
                  x={expertStartX}
                  y={ey + 15}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={showActive ? "bold" : "normal"}
                  fill={color}
                  fontFamily="var(--font-geist-mono), monospace"
                >
                  Expert {i + 1}
                </text>
                <text
                  x={expertStartX}
                  y={ey + 29}
                  textAnchor="middle"
                  fontSize={9}
                  fill={color}
                  fontFamily="var(--font-geist-mono), monospace"
                  opacity={0.7}
                >
                  {showActive ? "FFN (active)" : "FFN (idle)"}
                </text>

                {/* Gate weight label */}
                {step >= 2 && isSelected && (
                  <motion.g
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <rect
                      x={expertStartX + 44}
                      y={ey + 2}
                      width={50}
                      height={18}
                      rx={4}
                      fill={C_GATE}
                      fillOpacity={0.15}
                    />
                    <text
                      x={expertStartX + 69}
                      y={ey + 14}
                      textAnchor="middle"
                      fontSize={9}
                      fontWeight="bold"
                      fill={C_GATE}
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      {i === SELECTED_EXPERTS[0] ? "g₁=0.7" : "g₂=0.3"}
                    </text>
                  </motion.g>
                )}

                {/* Arrow from expert to output */}
                {step >= 5 && isSelected && (
                  <motion.line
                    x1={expertStartX + 40}
                    y1={ey + 20}
                    x2={outputX - 38}
                    y2={outputY}
                    stroke={C_OUTPUT}
                    strokeWidth={2}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.4 }}
                  />
                )}
              </g>
            );
          })}

          {/* Output combination */}
          {step >= 5 && (
            <motion.g
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
            >
              <rect
                x={outputX - 38}
                y={outputY - 28}
                width={76}
                height={56}
                rx={8}
                fill={C_OUTPUT}
                fillOpacity={0.2}
                stroke={C_OUTPUT}
                strokeWidth={2}
              />
              <text
                x={outputX}
                y={outputY - 6}
                textAnchor="middle"
                fontSize={11}
                fontWeight="bold"
                fill={C_OUTPUT}
                fontFamily="var(--font-geist-mono), monospace"
              >
                Σ gᵢ·Eᵢ(h)
              </text>
              <text
                x={outputX}
                y={outputY + 12}
                textAnchor="middle"
                fontSize={10}
                fill={C_OUTPUT}
                fontFamily="var(--font-geist-mono), monospace"
                opacity={0.7}
              >
                output
              </text>
            </motion.g>
          )}

          {/* Load balancing (step 6) */}
          {step >= 6 && (
            <motion.g
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <text
                x={svgW / 2}
                y={svgH - 68}
                textAnchor="middle"
                fontSize={11}
                fontWeight="bold"
                fill="var(--text-primary)"
                fontFamily="var(--font-geist-mono), monospace"
              >
                Expert Utilization
              </text>

              {Array.from({ length: NUM_EXPERTS }, (_, i) => {
                const barX = svgW / 2 - (NUM_EXPERTS * 38) / 2 + i * 38;
                const barMaxH = 45;
                const utilization = SELECTED_EXPERTS.includes(i) ? 0.85 : 0.15;
                const actualH = barMaxH * utilization;

                return (
                  <g key={`bar-${i}`}>
                    <rect
                      x={barX}
                      y={svgH - 55}
                      width={28}
                      height={barMaxH}
                      rx={4}
                      fill="var(--bg-tertiary)"
                      opacity={0.5}
                    />
                    <motion.rect
                      x={barX}
                      y={svgH - 55 + barMaxH - actualH}
                      width={28}
                      height={actualH}
                      rx={4}
                      fill={
                        SELECTED_EXPERTS.includes(i)
                          ? C_EXPERT_ACTIVE
                          : C_EXPERT_IDLE
                      }
                      fillOpacity={0.6}
                      initial={{ height: 0 }}
                      animate={{ height: actualH }}
                      transition={{ duration: 0.5, delay: i * 0.08 }}
                    />
                    <line
                      x1={barX}
                      y1={svgH - 55 + barMaxH * 0.5}
                      x2={barX + 28}
                      y2={svgH - 55 + barMaxH * 0.5}
                      stroke={C_GATE}
                      strokeWidth={1.5}
                      strokeDasharray="3 2"
                      opacity={0.7}
                    />
                    <text
                      x={barX + 14}
                      y={svgH - 5}
                      textAnchor="middle"
                      fontSize={8}
                      fill="var(--text-tertiary)"
                      fontFamily="var(--font-geist-mono), monospace"
                    >
                      E{i + 1}
                    </text>
                  </g>
                );
              })}
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
