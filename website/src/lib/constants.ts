export const LAYERS = [
  { id: "data", label: "Data Processing", color: "#3B82F6", phases: [1] },
  { id: "architecture", label: "Model Architecture", color: "#8B5CF6", phases: [2, 3] },
  { id: "training", label: "Training", color: "#F59E0B", phases: [4] },
  { id: "inference", label: "Inference", color: "#10B981", phases: [5] },
  { id: "application", label: "Applications", color: "#EF4444", phases: [6, 7, 8, 9] },
] as const;

export const PHASE_META: Record<number, { loc: number; difficulty: string; prerequisites: number[]; layer: string; keyInsight: string; coreAddition: string }> = {
  0: { loc: 0, difficulty: "beginner", prerequisites: [], layer: "overview", keyInsight: "Before writing code, build a mental model of the complete system", coreAddition: "Mental map" },
  1: { loc: 200, difficulty: "beginner", prerequisites: [0], layer: "data", keyInsight: "Text is just bytes — BPE turns them into learnable units", coreAddition: "BPE Tokenizer + DataLoader" },
  2: { loc: 300, difficulty: "intermediate", prerequisites: [1], layer: "architecture", keyInsight: "Attention lets every token look at every other token — scaled and masked", coreAddition: "Multi-Head Causal Self-Attention" },
  3: { loc: 250, difficulty: "intermediate", prerequisites: [2], layer: "architecture", keyInsight: "A Transformer Block is just Attention + FFN + Residuals, repeated N times", coreAddition: "Stackable Transformer Block" },
  4: { loc: 350, difficulty: "intermediate", prerequisites: [1, 3], layer: "training", keyInsight: "The training loop is predict → measure loss → backprop → update, repeated millions of times", coreAddition: "Training Loop + LR Schedule" },
  5: { loc: 200, difficulty: "intermediate", prerequisites: [4], layer: "inference", keyInsight: "Generation is just next-token prediction in a loop — sampling strategy controls creativity vs coherence", coreAddition: "Temperature + Top-k + Top-p" },
  6: { loc: 150, difficulty: "intermediate", prerequisites: [4], layer: "application", keyInsight: "Transfer learning: freeze the backbone, train a tiny head — most parameters already know language", coreAddition: "Classification Head" },
  7: { loc: 200, difficulty: "advanced", prerequisites: [4], layer: "application", keyInsight: "LoRA: instead of updating all weights, add tiny low-rank matrices — 95% fewer parameters", coreAddition: "LoRA Layers + Weight Merge" },
  8: { loc: 400, difficulty: "advanced", prerequisites: [5], layer: "application", keyInsight: "SFT teaches format, DPO teaches preference — together they align the model to human intent", coreAddition: "SFT + DPO Pipeline" },
  9: { loc: 350, difficulty: "advanced", prerequisites: [3], layer: "application", keyInsight: "MoE: not every token needs every expert — sparse activation gives more capacity at same cost", coreAddition: "MoE Router + Expert Networks" },
};
