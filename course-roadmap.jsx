import React, { useState } from 'react';

// ============================================================
// LLM From Scratch — Course Roadmap (Standalone React Component)
// ============================================================

const COURSE = {
  title: 'LLM From Scratch',
  subtitle: '從零構建大型語言模型',
  description: '9 個 Phase，從 Tokenizer 到 Mixture of Experts，用 PyTorch 手把手構建完整的 GPT-like LLM',
  techStack: ['Python', 'PyTorch', 'tiktoken', 'NumPy'],
  duration: '6-8 週',
  audience: '有 Python 基礎、了解基本深度學習概念的開發者',
};

const PHASES = [
  {
    id: 1,
    title: '文本與數據',
    titleEn: 'Text & Data',
    subtitle: 'Tokenizer + DataLoader',
    description: '理解 LLM 如何看待文本：從字符到 token，從 token 到張量。構建 BPE tokenizer 和滑動窗口數據加載器。',
    concepts: ['字符編碼與 Unicode', 'Byte Pair Encoding (BPE)', 'Token Embedding', 'Positional Encoding', 'Sliding Window DataLoader'],
    labTitle: '構建 BPE Tokenizer 和 DataLoader',
    labFiles: ['tokenizer.py', 'dataloader.py', 'test_tokenizer.py', 'test_dataloader.py'],
    dependencies: [],
    color: '#3B82F6',
  },
  {
    id: 2,
    title: '注意力機制',
    titleEn: 'Attention Mechanisms',
    subtitle: 'Self-Attention → Multi-Head Attention',
    description: '從最簡單的點積注意力開始，逐步構建因果自注意力和多頭注意力模組。',
    concepts: ['點積注意力 (Dot-Product Attention)', '縮放點積注意力 (Scaled)', '因果遮罩 (Causal Mask)', 'Query-Key-Value 投影', '多頭注意力 (Multi-Head Attention)'],
    labTitle: '實現 Multi-Head Causal Attention',
    labFiles: ['attention.py', 'test_attention.py'],
    dependencies: [1],
    color: '#8B5CF6',
  },
  {
    id: 3,
    title: 'Transformer 架構',
    titleEn: 'Transformer Architecture',
    subtitle: 'GPT 模型組裝',
    description: '將注意力、前饋網路、層歸一化和殘差連接組裝成完整的 GPT 模型。',
    concepts: ['Layer Normalization', 'Feed-Forward Network (GELU)', '殘差連接 (Residual Connection)', 'Transformer Block', 'GPT 模型組裝'],
    labTitle: '組裝完整的 GPT 模型',
    labFiles: ['transformer.py', 'gpt_model.py', 'test_transformer.py', 'test_gpt_model.py'],
    dependencies: [1, 2],
    color: '#EC4899',
  },
  {
    id: 4,
    title: '預訓練',
    titleEn: 'Pretraining',
    subtitle: 'Training Loop + Loss Computation',
    description: '在真實文本上預訓練 GPT 模型：實現交叉熵損失、AdamW 優化器配置、學習率排程和訓練迴圈。',
    concepts: ['交叉熵損失 (Cross-Entropy Loss)', 'AdamW 優化器', '學習率預熱與衰減 (Warmup + Cosine Decay)', '梯度裁剪 (Gradient Clipping)', '訓練迴圈與驗證'],
    labTitle: '預訓練 GPT 模型',
    labFiles: ['train.py', 'utils.py', 'test_train.py'],
    dependencies: [1, 2, 3],
    color: '#F59E0B',
  },
  {
    id: 5,
    title: '文本生成',
    titleEn: 'Text Generation',
    subtitle: 'Inference + Sampling Strategies',
    description: '實現多種解碼策略，從貪婪搜索到 nucleus sampling，讓模型生成連貫的文本。',
    concepts: ['貪婪解碼 (Greedy Decoding)', 'Temperature Scaling', 'Top-k Sampling', 'Top-p (Nucleus) Sampling', 'KV-Cache 優化'],
    labTitle: '實現文本生成引擎',
    labFiles: ['generate.py', 'test_generate.py'],
    dependencies: [3, 4],
    color: '#10B981',
  },
  {
    id: 6,
    title: '微調分類',
    titleEn: 'Fine-Tuning for Classification',
    subtitle: 'Transfer Learning + Classification Head',
    description: '凍結預訓練權重，加入分類頭，在下游任務上微調模型。理解遷移學習的原理。',
    concepts: ['遷移學習 (Transfer Learning)', '特徵提取 vs 全模型微調', '分類頭設計', '數據集準備與標籤處理', '評估指標 (Accuracy, F1)'],
    labTitle: '微調垃圾郵件分類器',
    labFiles: ['classifier.py', 'dataset.py', 'train_classifier.py', 'test_classifier.py'],
    dependencies: [4, 5],
    color: '#06B6D4',
  },
  {
    id: 7,
    title: 'LoRA',
    titleEn: 'LoRA: Low-Rank Adaptation',
    subtitle: 'Parameter-Efficient Fine-Tuning',
    description: '實現 LoRA — 通過低秩分解大幅減少微調參數量，實現高效的模型適配。',
    concepts: ['全量微調的問題', '低秩分解 (Low-Rank Factorization)', 'LoRA 數學原理 (W + BA)', '秩 r 的選擇', 'LoRA 層的合併與部署'],
    labTitle: '實現 LoRA 微調',
    labFiles: ['lora.py', 'train_lora.py', 'test_lora.py'],
    dependencies: [6],
    color: '#F97316',
  },
  {
    id: 8,
    title: '指令微調',
    titleEn: 'Instruction Fine-Tuning',
    subtitle: 'SFT + DPO + Evaluation',
    description: '將預訓練模型轉變為能遵循指令的助手：構建指令數據集、實現 SFT 和 DPO 對齊。',
    concepts: ['指令數據集格式 (Alpaca Format)', '監督微調 (SFT)', 'Direct Preference Optimization (DPO)', 'LLM-as-Judge 評估', '對話模板 (Chat Template)'],
    labTitle: '指令微調與 DPO 對齊',
    labFiles: ['sft.py', 'dpo.py', 'evaluate.py', 'test_sft.py', 'test_dpo.py'],
    dependencies: [6, 7],
    color: '#EF4444',
  },
  {
    id: 9,
    title: 'Mixture of Experts',
    titleEn: 'Mixture of Experts (MoE)',
    subtitle: 'Sparse Expert Architecture',
    description: '實現 MoE 層 — 用門控網路動態路由 token 到不同專家，在不增加推理成本的前提下擴展模型容量。',
    concepts: ['稠密 vs 稀疏模型', 'Expert Network 設計', 'Gating / Router Network', 'Top-k 路由策略', '負載均衡損失 (Load Balancing Loss)'],
    labTitle: '實現 MoE Transformer',
    labFiles: ['moe.py', 'moe_transformer.py', 'test_moe.py'],
    dependencies: [3, 4],
    color: '#A855F7',
  },
];

const ARCHITECTURE = {
  layers: [
    { name: 'Text Input', phase: 0, desc: '原始文本' },
    { name: 'Tokenizer (BPE)', phase: 1, desc: 'Phase 1: 文本 → Token IDs' },
    { name: 'Token + Position Embedding', phase: 1, desc: 'Phase 1: IDs → 向量' },
    { name: 'Multi-Head Attention', phase: 2, desc: 'Phase 2: 注意力機制' },
    { name: 'Feed-Forward + LayerNorm', phase: 3, desc: 'Phase 3: Transformer Block' },
    { name: 'Transformer Block × N', phase: 3, desc: 'Phase 3: 堆疊多層' },
    { name: 'MoE Layer (Optional)', phase: 9, desc: 'Phase 9: 稀疏專家' },
    { name: 'Pre-Training Loop', phase: 4, desc: 'Phase 4: 損失 + 優化' },
    { name: 'Text Generation', phase: 5, desc: 'Phase 5: 解碼策略' },
    { name: 'Classification Head', phase: 6, desc: 'Phase 6: 下游任務' },
    { name: 'LoRA Adapters', phase: 7, desc: 'Phase 7: 高效微調' },
    { name: 'SFT + DPO', phase: 8, desc: 'Phase 8: 指令對齊' },
  ],
};

const PRINCIPLES = [
  {
    title: '由下而上',
    titleEn: 'Bottom-Up',
    description: '每個 Phase 只引入一個新抽象層。先理解底層，再構建上層。',
    icon: '🧱',
  },
  {
    title: '真實可運行',
    titleEn: 'Real & Runnable',
    description: '每個 Lab 都能在筆電上運行，產出真實結果。不是偽代碼，是真正的 PyTorch。',
    icon: '🚀',
  },
  {
    title: '測試驅動',
    titleEn: 'Test-Driven',
    description: '每個模組都有完整的測試套件和自動評分腳本，確保實現正確。',
    icon: '🧪',
  },
  {
    title: '漸進式複雜度',
    titleEn: 'Progressive Complexity',
    description: '從 50 行的 tokenizer 到完整的 MoE 模型，難度逐步提升。',
    icon: '📈',
  },
];

// ============================================================
// UI Components
// ============================================================

const styles = {
  container: {
    fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
    backgroundColor: '#0A0A0B',
    color: '#E4E4E7',
    minHeight: '100vh',
    padding: '2rem',
  },
  header: {
    textAlign: 'center',
    marginBottom: '3rem',
    borderBottom: '1px solid #27272A',
    paddingBottom: '2rem',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: 700,
    background: 'linear-gradient(135deg, #3B82F6, #8B5CF6, #EC4899)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '0.5rem',
  },
  subtitle: {
    fontSize: '1.1rem',
    color: '#A1A1AA',
    marginBottom: '0.5rem',
  },
  meta: {
    fontSize: '0.85rem',
    color: '#71717A',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
    gap: '1.5rem',
    maxWidth: '1200px',
    margin: '0 auto 3rem',
  },
  card: (color, isExpanded) => ({
    backgroundColor: '#18181B',
    border: `1px solid ${isExpanded ? color : '#27272A'}`,
    borderRadius: '12px',
    padding: '1.5rem',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: isExpanded ? `0 0 20px ${color}22` : 'none',
  }),
  phaseNumber: (color) => ({
    fontSize: '0.75rem',
    color: color,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    marginBottom: '0.25rem',
  }),
  cardTitle: {
    fontSize: '1.2rem',
    fontWeight: 600,
    color: '#FAFAFA',
    marginBottom: '0.25rem',
  },
  cardSubtitle: {
    fontSize: '0.85rem',
    color: '#A1A1AA',
    marginBottom: '0.75rem',
  },
  cardDescription: {
    fontSize: '0.85rem',
    color: '#A1A1AA',
    lineHeight: 1.6,
    marginBottom: '1rem',
  },
  conceptList: {
    listStyle: 'none',
    padding: 0,
    margin: '0.75rem 0',
  },
  conceptItem: (color) => ({
    fontSize: '0.8rem',
    color: '#D4D4D8',
    padding: '0.25rem 0',
    borderLeft: `2px solid ${color}`,
    paddingLeft: '0.75rem',
    marginBottom: '0.25rem',
  }),
  labSection: {
    backgroundColor: '#0A0A0B',
    borderRadius: '8px',
    padding: '0.75rem',
    marginTop: '0.75rem',
  },
  labTitle: {
    fontSize: '0.8rem',
    color: '#10B981',
    fontWeight: 600,
    marginBottom: '0.5rem',
  },
  labFile: {
    fontSize: '0.75rem',
    color: '#6EE7B7',
    fontFamily: 'monospace',
    padding: '0.15rem 0',
  },
  sectionTitle: {
    fontSize: '1.5rem',
    fontWeight: 600,
    color: '#FAFAFA',
    textAlign: 'center',
    marginBottom: '1.5rem',
  },
  architectureDiagram: {
    maxWidth: '800px',
    margin: '0 auto 3rem',
    backgroundColor: '#18181B',
    border: '1px solid #27272A',
    borderRadius: '12px',
    padding: '2rem',
  },
  archLayer: (color) => ({
    display: 'flex',
    alignItems: 'center',
    padding: '0.5rem 1rem',
    borderLeft: `3px solid ${color || '#3B82F6'}`,
    marginBottom: '0.25rem',
    fontSize: '0.85rem',
  }),
  archLayerName: {
    color: '#FAFAFA',
    fontWeight: 500,
    minWidth: '240px',
  },
  archLayerDesc: {
    color: '#71717A',
    fontSize: '0.8rem',
  },
  principlesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
    gap: '1rem',
    maxWidth: '1100px',
    margin: '0 auto 3rem',
  },
  principleCard: {
    backgroundColor: '#18181B',
    border: '1px solid #27272A',
    borderRadius: '12px',
    padding: '1.25rem',
    textAlign: 'center',
  },
  principleIcon: {
    fontSize: '2rem',
    marginBottom: '0.5rem',
  },
  principleTitle: {
    fontSize: '1rem',
    fontWeight: 600,
    color: '#FAFAFA',
    marginBottom: '0.25rem',
  },
  principleDesc: {
    fontSize: '0.8rem',
    color: '#A1A1AA',
    lineHeight: 1.5,
  },
  arrow: {
    textAlign: 'center',
    color: '#3B82F6',
    fontSize: '1.2rem',
    margin: '0.15rem 0',
  },
};

function PhaseCard({ phase }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={styles.card(phase.color, expanded)} onClick={() => setExpanded(!expanded)}>
      <div style={styles.phaseNumber(phase.color)}>Phase {phase.id}</div>
      <div style={styles.cardTitle}>{phase.title}</div>
      <div style={styles.cardSubtitle}>{phase.subtitle}</div>
      {expanded && (
        <>
          <div style={styles.cardDescription}>{phase.description}</div>
          <div style={{ fontSize: '0.8rem', color: '#A1A1AA', fontWeight: 600, marginBottom: '0.25rem' }}>
            Core Concepts
          </div>
          <ul style={styles.conceptList}>
            {phase.concepts.map((c, i) => (
              <li key={i} style={styles.conceptItem(phase.color)}>{c}</li>
            ))}
          </ul>
          <div style={styles.labSection}>
            <div style={styles.labTitle}>Lab: {phase.labTitle}</div>
            {phase.labFiles.map((f, i) => (
              <div key={i} style={styles.labFile}>
                {f}
              </div>
            ))}
          </div>
          {phase.dependencies.length > 0 && (
            <div style={{ fontSize: '0.75rem', color: '#71717A', marginTop: '0.5rem' }}>
              Depends on: {phase.dependencies.map(d => `Phase ${d}`).join(', ')}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ArchitectureDiagram() {
  const phaseColors = {};
  PHASES.forEach(p => { phaseColors[p.id] = p.color; });

  return (
    <div style={styles.architectureDiagram}>
      <h2 style={{ ...styles.sectionTitle, textAlign: 'left', fontSize: '1.1rem', marginBottom: '1rem' }}>
        GPT Architecture — Layer by Layer
      </h2>
      {ARCHITECTURE.layers.map((layer, i) => (
        <React.Fragment key={i}>
          <div style={styles.archLayer(phaseColors[layer.phase] || '#3B82F6')}>
            <span style={styles.archLayerName}>{layer.name}</span>
            <span style={styles.archLayerDesc}>{layer.desc}</span>
          </div>
          {i < ARCHITECTURE.layers.length - 1 && (
            <div style={styles.arrow}>↓</div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

function Principles() {
  return (
    <div style={styles.principlesGrid}>
      {PRINCIPLES.map((p, i) => (
        <div key={i} style={styles.principleCard}>
          <div style={styles.principleIcon}>{p.icon}</div>
          <div style={styles.principleTitle}>{p.title}</div>
          <div style={{ fontSize: '0.75rem', color: '#71717A', marginBottom: '0.5rem' }}>{p.titleEn}</div>
          <div style={styles.principleDesc}>{p.description}</div>
        </div>
      ))}
    </div>
  );
}

export default function CourseRoadmap() {
  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>{COURSE.title}</h1>
        <p style={styles.subtitle}>{COURSE.subtitle}</p>
        <p style={styles.meta}>
          {COURSE.techStack.join(' · ')} | {COURSE.duration} | {COURSE.audience}
        </p>
      </header>

      <h2 style={styles.sectionTitle}>Design Principles</h2>
      <Principles />

      <h2 style={styles.sectionTitle}>Architecture</h2>
      <ArchitectureDiagram />

      <h2 style={styles.sectionTitle}>Course Phases</h2>
      <div style={styles.grid}>
        {PHASES.map(phase => (
          <PhaseCard key={phase.id} phase={phase} />
        ))}
      </div>
    </div>
  );
}
