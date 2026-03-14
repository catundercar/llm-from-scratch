import type { Phase, Architecture, Principle } from "./types";
import type { Locale } from "../i18n";

const phasesZhTW: Phase[] = [
  {
    id: 0, week: "Phase 0", duration: "1 週", title: "全景圖：Transformer 與 LLM",
    subtitle: "從 RNN 到 GPT——理解大型語言模型的全貌",
    icon: "⊙", color: "#6366F1", accent: "#818CF8",
    goal: "在動手寫任何程式碼之前，先建立對 Transformer 與 LLM 的全局理解。你會了解一段文字如何進入模型、經過哪些處理、最終如何生成新的文字。這個 Phase 沒有程式碼作業，但它是所有後續 Phase 的心智地圖。",
    concepts: ["LLM 是什麼：文字進、文字出的機器", "完整資料流：Text → Tokens → Embeddings → Transformer Blocks → Logits → Sampling → Text", "歷史脈絡：RNN → Seq2Seq+Attention → Transformer → GPT", "Encoder-Decoder vs Decoder-Only 架構", "Self-Attention 的直覺（不含數學）", "什麼讓 LLM「大」：參數、資料、算力的三角關係"],
    readings: ["3Blue1Brown — Visual Intro to Transformers", "Brendan Bycroft — LLM Visualization (互動式 3D)", "Andrej Karpathy — Let's build GPT (2h video)", "Lilian Weng — Attention? Attention!"],
    deliverable: {
      name: "概念筆記 + 架構圖",
      desc: "手繪或數位的 Transformer 架構圖，標註每個組件的角色與資料流方向",
      acceptance: ["能用自己的話解釋 LLM 如何從文字生成文字", "能畫出完整的資料流管線", "能說明 Decoder-Only 與 Encoder-Decoder 的差異"]
    }
  },
  {
    id: 1, week: "Phase 1", duration: "1–2 週", title: "文本處理與資料管線",
    subtitle: "BPE Tokenizer、Token/Position Embedding、DataLoader",
    icon: "⬡", color: "#3B82F6", accent: "#60A5FA",
    goal: "從零實作 BPE Tokenizer，建立 Token 與 Position Embedding，並打造 DataLoader 來產生訓練用的 (x, y) 對。這是所有後續 Phase 的基石。",
    concepts: ["Byte-Pair Encoding (BPE)", "Token Embedding", "Positional Encoding", "Sliding-Window DataLoader"],
    readings: ["Sennrich et al. 2016 — BPE 原始論文", "GPT-2 Paper §2.2 — Input Representation", "tiktoken 原始碼"],
    deliverable: {
      name: "BPE Tokenizer + DataLoader",
      desc: "能對任意文本做 encode / decode，並產生固定長度的訓練批次",
      acceptance: ["vocab_size 可設定且合理", "encode → decode 完美 roundtrip", "DataLoader 產出正確的 (x, y) 偏移對"]
    }
  },
  {
    id: 2, week: "Phase 2", duration: "2–3 週", title: "注意力機制",
    subtitle: "Scaled Dot-Product、Causal Mask、Multi-Head Attention",
    icon: "◈", color: "#8B5CF6", accent: "#A78BFA",
    goal: "深入理解 QKV 語義，實作 Scaled Dot-Product Attention 與 Causal Mask，最後組裝出 Multi-Head Attention 模組。",
    concepts: ["Query / Key / Value", "Scaled Dot-Product", "Causal Masking", "Multi-Head Attention"],
    readings: ["Attention Is All You Need (Vaswani et al. 2017)", "The Illustrated Transformer", "FlashAttention Paper"],
    deliverable: {
      name: "Multi-Head Causal Self-Attention",
      desc: "完整的注意力模組，支援多頭與因果遮罩",
      acceptance: ["attention weights 行和為 1", "causal mask 正確遮蔽未來位置", "多頭輸出 shape 正確"]
    }
  },
  {
    id: 3, week: "Phase 3", duration: "1–2 週", title: "Transformer Block",
    subtitle: "LayerNorm、GELU、Feed-Forward、殘差連接",
    icon: "◇", color: "#EC4899", accent: "#F472B6",
    goal: "將 Attention 與 Feed-Forward 組裝成完整的 Transformer Block，加入 LayerNorm 與殘差連接。",
    concepts: ["Pre-Norm vs Post-Norm", "GELU 激活函數", "Feed-Forward Network", "殘差連接"],
    readings: ["GPT-2 Paper — Model Architecture", "On Layer Normalization in the Transformer Architecture"],
    deliverable: {
      name: "Transformer Block",
      desc: "可堆疊的 Transformer Block，包含 Attention + FFN + LayerNorm + Residual",
      acceptance: ["輸入輸出 shape 一致", "殘差路徑保持梯度流通", "可堆疊 N 層"]
    }
  },
  {
    id: 4, week: "Phase 4", duration: "2–3 週", title: "預訓練",
    subtitle: "Cross-Entropy Loss、AdamW、LR Schedule、梯度裁剪",
    icon: "◎", color: "#F59E0B", accent: "#FBBF24",
    goal: "實作完整的 GPT 預訓練迴圈：損失函數、優化器、學習率排程與檢查點儲存。",
    concepts: ["Cross-Entropy Loss", "AdamW 優化器", "Cosine LR Schedule", "Gradient Clipping", "Checkpointing"],
    readings: ["GPT-2 Paper — Training Details", "Decoupled Weight Decay Regularization (Loshchilov & Hutter)"],
    deliverable: {
      name: "Pre-training Pipeline",
      desc: "在小型語料上預訓練 GPT，並觀察 loss 下降",
      acceptance: ["training loss 持續下降", "checkpoint 可正確恢復", "gradient norm 保持穩定"]
    }
  },
  {
    id: 5, week: "Phase 5", duration: "1–2 週", title: "文本生成",
    subtitle: "Greedy、Temperature、Top-k、Top-p / Nucleus",
    icon: "△", color: "#10B981", accent: "#34D399",
    goal: "實作多種文本生成策略，從貪心搜索到 Nucleus Sampling，並加入 KV-Cache 加速推論。",
    concepts: ["Greedy Decoding", "Temperature Scaling", "Top-k Sampling", "Top-p / Nucleus Sampling", "KV-Cache"],
    readings: ["The Curious Case of Neural Text Degeneration (Holtzman et al.)", "GPT-2 原始程式碼 — generation"],
    deliverable: {
      name: "Text Generation Module",
      desc: "支援多種採樣策略的文本生成器",
      acceptance: ["greedy 結果確定性", "temperature=0 等同 greedy", "top-k/top-p 結果合理多樣"]
    }
  },
  {
    id: 6, week: "Phase 6", duration: "1–2 週", title: "分類微調",
    subtitle: "分類頭、特徵擷取、完整微調",
    icon: "□", color: "#06B6D4", accent: "#22D3EE",
    goal: "在預訓練模型上新增分類頭，嘗試凍結骨幹的特徵擷取與完整微調兩種策略。",
    concepts: ["Classification Head", "Feature Extraction（凍結骨幹）", "Full Fine-Tuning"],
    readings: ["ULMFiT (Howard & Ruder 2018)", "BERT Fine-Tuning Paper"],
    deliverable: {
      name: "Text Classifier",
      desc: "基於預訓練 GPT 的文本分類器",
      acceptance: ["accuracy > 80% on test set", "feature extraction vs full FT 有可比較的結果", "overfitting 控制得當"]
    }
  },
  {
    id: 7, week: "Phase 7", duration: "1–2 週", title: "LoRA 高效微調",
    subtitle: "LoRA 層、低秩分解、權重合併",
    icon: "⬢", color: "#F97316", accent: "#FB923C",
    goal: "實作 LoRA（Low-Rank Adaptation），理解低秩分解原理，並將訓練後的 LoRA 權重合併回原始模型。",
    concepts: ["Low-Rank Decomposition", "LoRA Layers", "Weight Merging", "Parameter Efficiency"],
    readings: ["LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)", "QLoRA Paper"],
    deliverable: {
      name: "LoRA Fine-Tuning",
      desc: "使用 LoRA 微調 GPT 模型，大幅減少可訓練參數",
      acceptance: ["可訓練參數 < 5% 總參數", "微調後效能接近 full FT", "merge 後推論正確"]
    }
  },
  {
    id: 8, week: "Phase 8", duration: "2–3 週", title: "人類對齊",
    subtitle: "SFT、DPO、Chat Templates、LLM-as-Judge",
    icon: "✦", color: "#EF4444", accent: "#F87171",
    goal: "實作 Supervised Fine-Tuning（SFT）與 Direct Preference Optimization（DPO），將模型對齊到人類偏好。",
    concepts: ["Supervised Fine-Tuning (SFT)", "Direct Preference Optimization (DPO)", "Chat Templates", "LLM-as-Judge"],
    readings: ["Training language models to follow instructions (InstructGPT)", "DPO: Direct Preference Optimization (Rafailov et al.)"],
    deliverable: {
      name: "Aligned Chat Model",
      desc: "經過 SFT + DPO 對齊的對話模型",
      acceptance: ["SFT 後能遵循指令格式", "DPO 後偏好回應品質提升", "chat template 正確格式化"]
    }
  },
  {
    id: 9, week: "Phase 9", duration: "2–3 週", title: "Mixture of Experts",
    subtitle: "專家網路、路由器/門控、Top-k 路由、負載均衡",
    icon: "⟐", color: "#A855F7", accent: "#C084FC",
    goal: "實作 Mixture of Experts（MoE）架構，將 Feed-Forward 層替換為多個專家網路，並實作路由機制與負載均衡。",
    concepts: ["Expert Networks", "Router / Gate", "Top-k Routing", "Load Balancing Loss"],
    readings: ["Switch Transformers (Fedus et al. 2021)", "Mixtral of Experts Technical Report"],
    deliverable: {
      name: "MoE Transformer",
      desc: "將標準 Transformer 升級為 MoE 架構",
      acceptance: ["router 正確選擇 top-k 專家", "load balancing loss 有效", "MoE 模型可正常訓練與推論"]
    }
  }
];

const phasesZhCN: Phase[] = [
  {
    id: 0, week: "Phase 0", duration: "1 周", title: "全景图：Transformer 与 LLM",
    subtitle: "从 RNN 到 GPT——理解大型语言模型的全貌",
    icon: "⊙", color: "#6366F1", accent: "#818CF8",
    goal: "在动手写任何代码之前，先建立对 Transformer 与 LLM 的全局理解。你会了解一段文字如何进入模型、经过哪些处理、最终如何生成新的文字。这个 Phase 没有代码作业，但它是所有后续 Phase 的心智地图。",
    concepts: ["LLM 是什么：文字进、文字出的机器", "完整数据流：Text → Tokens → Embeddings → Transformer Blocks → Logits → Sampling → Text", "历史脉络：RNN → Seq2Seq+Attention → Transformer → GPT", "Encoder-Decoder vs Decoder-Only 架构", "Self-Attention 的直觉（不含数学）", "什么让 LLM「大」：参数、数据、算力的三角关系"],
    readings: ["3Blue1Brown — Visual Intro to Transformers", "Brendan Bycroft — LLM Visualization (互动式 3D)", "Andrej Karpathy — Let's build GPT (2h video)", "Lilian Weng — Attention? Attention!"],
    deliverable: {
      name: "概念笔记 + 架构图",
      desc: "手绘或数字的 Transformer 架构图，标注每个组件的角色与数据流方向",
      acceptance: ["能用自己的话解释 LLM 如何从文字生成文字", "能画出完整的数据流管线", "能说明 Decoder-Only 与 Encoder-Decoder 的差异"]
    }
  },
  {
    id: 1, week: "Phase 1", duration: "1–2 周", title: "文本处理与数据管线",
    subtitle: "BPE Tokenizer、Token/Position Embedding、DataLoader",
    icon: "⬡", color: "#3B82F6", accent: "#60A5FA",
    goal: "从零实现 BPE Tokenizer，建立 Token 与 Position Embedding，并打造 DataLoader 来生成训练用的 (x, y) 对。这是所有后续 Phase 的基石。",
    concepts: ["Byte-Pair Encoding (BPE)", "Token Embedding", "Positional Encoding", "Sliding-Window DataLoader"],
    readings: ["Sennrich et al. 2016 — BPE 原始论文", "GPT-2 Paper §2.2 — Input Representation", "tiktoken 源代码"],
    deliverable: {
      name: "BPE Tokenizer + DataLoader",
      desc: "能对任意文本做 encode / decode，并生成固定长度的训练批次",
      acceptance: ["vocab_size 可设定且合理", "encode → decode 完美 roundtrip", "DataLoader 生成正确的 (x, y) 偏移对"]
    }
  },
  {
    id: 2, week: "Phase 2", duration: "2–3 周", title: "注意力机制",
    subtitle: "Scaled Dot-Product、Causal Mask、Multi-Head Attention",
    icon: "◈", color: "#8B5CF6", accent: "#A78BFA",
    goal: "深入理解 QKV 语义，实现 Scaled Dot-Product Attention 与 Causal Mask，最后组装出 Multi-Head Attention 模块。",
    concepts: ["Query / Key / Value", "Scaled Dot-Product", "Causal Masking", "Multi-Head Attention"],
    readings: ["Attention Is All You Need (Vaswani et al. 2017)", "The Illustrated Transformer", "FlashAttention Paper"],
    deliverable: {
      name: "Multi-Head Causal Self-Attention",
      desc: "完整的注意力模块，支持多头与因果遮罩",
      acceptance: ["attention weights 行和为 1", "causal mask 正确遮蔽未来位置", "多头输出 shape 正确"]
    }
  },
  {
    id: 3, week: "Phase 3", duration: "1–2 周", title: "Transformer Block",
    subtitle: "LayerNorm、GELU、Feed-Forward、残差连接",
    icon: "◇", color: "#EC4899", accent: "#F472B6",
    goal: "将 Attention 与 Feed-Forward 组装成完整的 Transformer Block，加入 LayerNorm 与残差连接。",
    concepts: ["Pre-Norm vs Post-Norm", "GELU 激活函数", "Feed-Forward Network", "残差连接"],
    readings: ["GPT-2 Paper — Model Architecture", "On Layer Normalization in the Transformer Architecture"],
    deliverable: {
      name: "Transformer Block",
      desc: "可堆叠的 Transformer Block，包含 Attention + FFN + LayerNorm + Residual",
      acceptance: ["输入输出 shape 一致", "残差路径保持梯度流通", "可堆叠 N 层"]
    }
  },
  {
    id: 4, week: "Phase 4", duration: "2–3 周", title: "预训练",
    subtitle: "Cross-Entropy Loss、AdamW、LR Schedule、梯度裁剪",
    icon: "◎", color: "#F59E0B", accent: "#FBBF24",
    goal: "实现完整的 GPT 预训练循环：损失函数、优化器、学习率调度与检查点保存。",
    concepts: ["Cross-Entropy Loss", "AdamW 优化器", "Cosine LR Schedule", "Gradient Clipping", "Checkpointing"],
    readings: ["GPT-2 Paper — Training Details", "Decoupled Weight Decay Regularization (Loshchilov & Hutter)"],
    deliverable: {
      name: "Pre-training Pipeline",
      desc: "在小型语料上预训练 GPT，并观察 loss 下降",
      acceptance: ["training loss 持续下降", "checkpoint 可正确恢复", "gradient norm 保持稳定"]
    }
  },
  {
    id: 5, week: "Phase 5", duration: "1–2 周", title: "文本生成",
    subtitle: "Greedy、Temperature、Top-k、Top-p / Nucleus",
    icon: "△", color: "#10B981", accent: "#34D399",
    goal: "实现多种文本生成策略，从贪心搜索到 Nucleus Sampling，并加入 KV-Cache 加速推理。",
    concepts: ["Greedy Decoding", "Temperature Scaling", "Top-k Sampling", "Top-p / Nucleus Sampling", "KV-Cache"],
    readings: ["The Curious Case of Neural Text Degeneration (Holtzman et al.)", "GPT-2 源代码 — generation"],
    deliverable: {
      name: "Text Generation Module",
      desc: "支持多种采样策略的文本生成器",
      acceptance: ["greedy 结果确定性", "temperature=0 等同 greedy", "top-k/top-p 结果合理多样"]
    }
  },
  {
    id: 6, week: "Phase 6", duration: "1–2 周", title: "分类微调",
    subtitle: "分类头、特征提取、完整微调",
    icon: "□", color: "#06B6D4", accent: "#22D3EE",
    goal: "在预训练模型上新增分类头，尝试冻结骨干的特征提取与完整微调两种策略。",
    concepts: ["Classification Head", "Feature Extraction（冻结骨干）", "Full Fine-Tuning"],
    readings: ["ULMFiT (Howard & Ruder 2018)", "BERT Fine-Tuning Paper"],
    deliverable: {
      name: "Text Classifier",
      desc: "基于预训练 GPT 的文本分类器",
      acceptance: ["accuracy > 80% on test set", "feature extraction vs full FT 有可比较的结果", "overfitting 控制得当"]
    }
  },
  {
    id: 7, week: "Phase 7", duration: "1–2 周", title: "LoRA 高效微调",
    subtitle: "LoRA 层、低秩分解、权重合并",
    icon: "⬢", color: "#F97316", accent: "#FB923C",
    goal: "实现 LoRA（Low-Rank Adaptation），理解低秩分解原理，并将训练后的 LoRA 权重合并回原始模型。",
    concepts: ["Low-Rank Decomposition", "LoRA Layers", "Weight Merging", "Parameter Efficiency"],
    readings: ["LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)", "QLoRA Paper"],
    deliverable: {
      name: "LoRA Fine-Tuning",
      desc: "使用 LoRA 微调 GPT 模型，大幅减少可训练参数",
      acceptance: ["可训练参数 < 5% 总参数", "微调后效能接近 full FT", "merge 后推理正确"]
    }
  },
  {
    id: 8, week: "Phase 8", duration: "2–3 周", title: "人类对齐",
    subtitle: "SFT、DPO、Chat Templates、LLM-as-Judge",
    icon: "✦", color: "#EF4444", accent: "#F87171",
    goal: "实现 Supervised Fine-Tuning（SFT）与 Direct Preference Optimization（DPO），将模型对齐到人类偏好。",
    concepts: ["Supervised Fine-Tuning (SFT)", "Direct Preference Optimization (DPO)", "Chat Templates", "LLM-as-Judge"],
    readings: ["Training language models to follow instructions (InstructGPT)", "DPO: Direct Preference Optimization (Rafailov et al.)"],
    deliverable: {
      name: "Aligned Chat Model",
      desc: "经过 SFT + DPO 对齐的对话模型",
      acceptance: ["SFT 后能遵循指令格式", "DPO 后偏好回应质量提升", "chat template 正确格式化"]
    }
  },
  {
    id: 9, week: "Phase 9", duration: "2–3 周", title: "Mixture of Experts",
    subtitle: "专家网络、路由器/门控、Top-k 路由、负载均衡",
    icon: "⟐", color: "#A855F7", accent: "#C084FC",
    goal: "实现 Mixture of Experts（MoE）架构，将 Feed-Forward 层替换为多个专家网络，并实现路由机制与负载均衡。",
    concepts: ["Expert Networks", "Router / Gate", "Top-k Routing", "Load Balancing Loss"],
    readings: ["Switch Transformers (Fedus et al. 2021)", "Mixtral of Experts Technical Report"],
    deliverable: {
      name: "MoE Transformer",
      desc: "将标准 Transformer 升级为 MoE 架构",
      acceptance: ["router 正确选择 top-k 专家", "load balancing loss 有效", "MoE 模型可正常训练与推理"]
    }
  }
];

const phasesEn: Phase[] = [
  {
    id: 0, week: "Phase 0", duration: "1 week", title: "The Big Picture: Transformers & LLMs",
    subtitle: "From RNN to GPT — Understanding the Full Landscape",
    icon: "⊙", color: "#6366F1", accent: "#818CF8",
    goal: "Before writing any code, build a complete mental model of how Transformers and LLMs work. You'll understand how text enters the model, what processing happens at each stage, and how new text is generated. This phase has no coding assignments — it's your mental map for everything that follows.",
    concepts: ["What is an LLM: a text-in, text-out machine", "Complete data flow: Text → Tokens → Embeddings → Transformer Blocks → Logits → Sampling → Text", "Historical context: RNN → Seq2Seq+Attention → Transformer → GPT", "Encoder-Decoder vs Decoder-Only architecture", "Self-Attention intuition (no math yet)", "What makes LLMs 'large': the parameters-data-compute triangle"],
    readings: ["3Blue1Brown — Visual Intro to Transformers", "Brendan Bycroft — LLM Visualization (interactive 3D)", "Andrej Karpathy — Let's build GPT (2h video)", "Lilian Weng — Attention? Attention!"],
    deliverable: {
      name: "Concept Notes + Architecture Diagram",
      desc: "Hand-drawn or digital Transformer architecture diagram labeling each component's role and data flow direction",
      acceptance: ["Can explain in your own words how an LLM generates text from text", "Can draw the complete data flow pipeline", "Can explain the difference between Decoder-Only and Encoder-Decoder"]
    }
  },
  {
    id: 1, week: "Phase 1", duration: "1–2 weeks", title: "Text Processing & Data Pipeline",
    subtitle: "BPE Tokenizer, Token/Position Embedding, DataLoader",
    icon: "⬡", color: "#3B82F6", accent: "#60A5FA",
    goal: "Build a BPE Tokenizer from scratch, create Token and Position Embeddings, and build a DataLoader that produces (x, y) pairs for training. This is the foundation for everything that follows.",
    concepts: ["Byte-Pair Encoding (BPE)", "Token Embedding", "Positional Encoding", "Sliding-Window DataLoader"],
    readings: ["Sennrich et al. 2016 — BPE original paper", "GPT-2 Paper §2.2 — Input Representation", "tiktoken source code"],
    deliverable: {
      name: "BPE Tokenizer + DataLoader",
      desc: "Encode/decode arbitrary text and produce fixed-length training batches",
      acceptance: ["vocab_size is configurable and reasonable", "encode → decode perfect roundtrip", "DataLoader produces correct (x, y) offset pairs"]
    }
  },
  {
    id: 2, week: "Phase 2", duration: "2–3 weeks", title: "Attention Mechanisms",
    subtitle: "Scaled Dot-Product, Causal Mask, Multi-Head Attention",
    icon: "◈", color: "#8B5CF6", accent: "#A78BFA",
    goal: "Deeply understand QKV semantics, implement Scaled Dot-Product Attention with Causal Masking, and assemble Multi-Head Attention.",
    concepts: ["Query / Key / Value", "Scaled Dot-Product", "Causal Masking", "Multi-Head Attention"],
    readings: ["Attention Is All You Need (Vaswani et al. 2017)", "The Illustrated Transformer", "FlashAttention Paper"],
    deliverable: {
      name: "Multi-Head Causal Self-Attention",
      desc: "Complete attention module with multi-head and causal masking",
      acceptance: ["attention weights rows sum to 1", "causal mask correctly blocks future positions", "multi-head output shape is correct"]
    }
  },
  {
    id: 3, week: "Phase 3", duration: "1–2 weeks", title: "Transformer Block",
    subtitle: "LayerNorm, GELU, Feed-Forward, Residual Connections",
    icon: "◇", color: "#EC4899", accent: "#F472B6",
    goal: "Assemble Attention and Feed-Forward into a complete Transformer Block with LayerNorm and residual connections.",
    concepts: ["Pre-Norm vs Post-Norm", "GELU Activation", "Feed-Forward Network", "Residual Connections"],
    readings: ["GPT-2 Paper — Model Architecture", "On Layer Normalization in the Transformer Architecture"],
    deliverable: {
      name: "Transformer Block",
      desc: "Stackable Transformer Block with Attention + FFN + LayerNorm + Residual",
      acceptance: ["Input/output shapes match", "Residual paths maintain gradient flow", "Can stack N layers"]
    }
  },
  {
    id: 4, week: "Phase 4", duration: "2–3 weeks", title: "Pre-training",
    subtitle: "Cross-Entropy Loss, AdamW, LR Schedule, Gradient Clipping",
    icon: "◎", color: "#F59E0B", accent: "#FBBF24",
    goal: "Implement the complete GPT pre-training loop: loss function, optimizer, learning rate schedule, and checkpointing.",
    concepts: ["Cross-Entropy Loss", "AdamW Optimizer", "Cosine LR Schedule", "Gradient Clipping", "Checkpointing"],
    readings: ["GPT-2 Paper — Training Details", "Decoupled Weight Decay Regularization (Loshchilov & Hutter)"],
    deliverable: {
      name: "Pre-training Pipeline",
      desc: "Pre-train GPT on a small corpus and observe loss decreasing",
      acceptance: ["training loss consistently decreases", "checkpoint can be correctly restored", "gradient norm stays stable"]
    }
  },
  {
    id: 5, week: "Phase 5", duration: "1–2 weeks", title: "Text Generation",
    subtitle: "Greedy, Temperature, Top-k, Top-p / Nucleus",
    icon: "△", color: "#10B981", accent: "#34D399",
    goal: "Implement multiple text generation strategies from greedy search to Nucleus Sampling, plus KV-Cache for faster inference.",
    concepts: ["Greedy Decoding", "Temperature Scaling", "Top-k Sampling", "Top-p / Nucleus Sampling", "KV-Cache"],
    readings: ["The Curious Case of Neural Text Degeneration (Holtzman et al.)", "GPT-2 source code — generation"],
    deliverable: {
      name: "Text Generation Module",
      desc: "Text generator supporting multiple sampling strategies",
      acceptance: ["greedy results are deterministic", "temperature=0 equals greedy", "top-k/top-p results are reasonably diverse"]
    }
  },
  {
    id: 6, week: "Phase 6", duration: "1–2 weeks", title: "Classification Fine-Tuning",
    subtitle: "Classification Head, Feature Extraction, Full Fine-Tuning",
    icon: "□", color: "#06B6D4", accent: "#22D3EE",
    goal: "Add a classification head on top of the pre-trained model, try both frozen-backbone feature extraction and full fine-tuning.",
    concepts: ["Classification Head", "Feature Extraction (frozen backbone)", "Full Fine-Tuning"],
    readings: ["ULMFiT (Howard & Ruder 2018)", "BERT Fine-Tuning Paper"],
    deliverable: {
      name: "Text Classifier",
      desc: "Text classifier based on pre-trained GPT",
      acceptance: ["accuracy > 80% on test set", "feature extraction vs full FT have comparable results", "overfitting is controlled"]
    }
  },
  {
    id: 7, week: "Phase 7", duration: "1–2 weeks", title: "LoRA Efficient Tuning",
    subtitle: "LoRA Layers, Low-Rank Decomposition, Weight Merging",
    icon: "⬢", color: "#F97316", accent: "#FB923C",
    goal: "Implement LoRA (Low-Rank Adaptation), understand low-rank decomposition, and merge trained LoRA weights back into the original model.",
    concepts: ["Low-Rank Decomposition", "LoRA Layers", "Weight Merging", "Parameter Efficiency"],
    readings: ["LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)", "QLoRA Paper"],
    deliverable: {
      name: "LoRA Fine-Tuning",
      desc: "Fine-tune GPT with LoRA, drastically reducing trainable parameters",
      acceptance: ["trainable params < 5% of total", "fine-tuned performance close to full FT", "merged model infers correctly"]
    }
  },
  {
    id: 8, week: "Phase 8", duration: "2–3 weeks", title: "Human Alignment",
    subtitle: "SFT, DPO, Chat Templates, LLM-as-Judge",
    icon: "✦", color: "#EF4444", accent: "#F87171",
    goal: "Implement Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to align the model with human preferences.",
    concepts: ["Supervised Fine-Tuning (SFT)", "Direct Preference Optimization (DPO)", "Chat Templates", "LLM-as-Judge"],
    readings: ["Training language models to follow instructions (InstructGPT)", "DPO: Direct Preference Optimization (Rafailov et al.)"],
    deliverable: {
      name: "Aligned Chat Model",
      desc: "Chat model aligned with SFT + DPO",
      acceptance: ["SFT model follows instruction format", "DPO improves response quality", "chat template formats correctly"]
    }
  },
  {
    id: 9, week: "Phase 9", duration: "2–3 weeks", title: "Mixture of Experts",
    subtitle: "Expert Networks, Router/Gate, Top-k Routing, Load Balancing",
    icon: "⟐", color: "#A855F7", accent: "#C084FC",
    goal: "Implement Mixture of Experts (MoE) architecture, replacing Feed-Forward layers with multiple expert networks, with routing and load balancing.",
    concepts: ["Expert Networks", "Router / Gate", "Top-k Routing", "Load Balancing Loss"],
    readings: ["Switch Transformers (Fedus et al. 2021)", "Mixtral of Experts Technical Report"],
    deliverable: {
      name: "MoE Transformer",
      desc: "Upgrade the standard Transformer to MoE architecture",
      acceptance: ["router correctly selects top-k experts", "load balancing loss is effective", "MoE model trains and infers normally"]
    }
  }
];

const allPhases: Record<string, Phase[]> = {
  "zh-TW": phasesZhTW,
  "zh-CN": phasesZhCN,
  en: phasesEn,
};

export function getPhases(locale: Locale): Phase[] {
  return allPhases[locale] ?? allPhases["en"];
}

export const architecture: Architecture = {
  layers: [
    { name: "Text Processing", color: "#3B82F6", modules: ["BPE Tokenizer", "Token Embedding", "Position Embedding", "DataLoader"] },
    { name: "Attention", color: "#8B5CF6", modules: ["Scaled Dot-Product", "Causal Mask", "Multi-Head Attention", "QKV Projections"] },
    { name: "Transformer", color: "#EC4899", modules: ["LayerNorm", "GELU", "Feed-Forward", "Residual Connection", "Transformer Block"] },
    { name: "Training", color: "#F59E0B", modules: ["Cross-Entropy Loss", "AdamW", "LR Schedule", "Gradient Clipping", "Checkpointing"] },
    { name: "Generation", color: "#10B981", modules: ["Greedy", "Temperature", "Top-k", "Top-p / Nucleus", "KV-Cache"] },
    { name: "Fine-Tuning", color: "#06B6D4", modules: ["Classification Head", "Feature Extraction", "Full Fine-Tuning"] },
    { name: "Efficient Tuning", color: "#F97316", modules: ["LoRA Layers", "Low-Rank Decomposition", "Weight Merging"] },
    { name: "Alignment", color: "#EF4444", modules: ["SFT", "DPO", "Chat Templates", "LLM-as-Judge"] },
    { name: "Sparse Models", color: "#A855F7", modules: ["Expert Networks", "Router/Gate", "Top-k Routing", "Load Balancing"] },
  ],
};

const principlesZhTW: Principle[] = [
  { num: "01", title: "由下而上", desc: "每個 Phase 只引入一個新抽象層，前一個 Phase 的輸出就是下一個的輸入。你永遠清楚每一行程式碼在做什麼。", color: "#3B82F6" },
  { num: "02", title: "真實可運行", desc: "每個 Lab 都能在筆電上運行，用真實的資料、真實的梯度。不是虛擬碼，而是可以直接 python train.py 的完整程式。", color: "#10B981" },
  { num: "03", title: "測試驅動", desc: "每個模組都有完整的測試套件，包含 shape test、數值穩定性檢查與整合測試。先讀測試，再寫程式碼。", color: "#F59E0B" },
  { num: "04", title: "漸進式複雜度", desc: "從 50 行的 tokenizer 到完整的 MoE 架構，每一步只增加可管理的複雜度。你不會突然面對無法理解的巨型系統。", color: "#EC4899" },
];

const principlesZhCN: Principle[] = [
  { num: "01", title: "由下而上", desc: "每个 Phase 只引入一个新抽象层，前一个 Phase 的输出就是下一个的输入。你永远清楚每一行代码在做什么。", color: "#3B82F6" },
  { num: "02", title: "真实可运行", desc: "每个 Lab 都能在笔记本上运行，用真实的数据、真实的梯度。不是伪代码，而是可以直接 python train.py 的完整程序。", color: "#10B981" },
  { num: "03", title: "测试驱动", desc: "每个模块都有完整的测试套件，包含 shape test、数值稳定性检查与集成测试。先读测试，再写代码。", color: "#F59E0B" },
  { num: "04", title: "渐进式复杂度", desc: "从 50 行的 tokenizer 到完整的 MoE 架构，每一步只增加可管理的复杂度。你不会突然面对无法理解的巨型系统。", color: "#EC4899" },
];

const principlesEn: Principle[] = [
  { num: "01", title: "Bottom-Up", desc: "Each Phase introduces exactly one new abstraction layer. The output of one Phase becomes the input of the next. You always know what every line of code does.", color: "#3B82F6" },
  { num: "02", title: "Real & Runnable", desc: "Every Lab runs on your laptop with real data and real gradients. Not pseudocode — a complete program you can run with python train.py.", color: "#10B981" },
  { num: "03", title: "Test-Driven", desc: "Every module has a complete test suite including shape tests, numerical stability checks, and integration tests. Read the tests first, then write the code.", color: "#F59E0B" },
  { num: "04", title: "Progressive Complexity", desc: "From a 50-line tokenizer to a full MoE architecture, each step adds only manageable complexity. You never face an incomprehensible giant system.", color: "#EC4899" },
];

const allPrinciples: Record<string, Principle[]> = {
  "zh-TW": principlesZhTW,
  "zh-CN": principlesZhCN,
  en: principlesEn,
};

export function getPrinciples(locale: Locale): Principle[] {
  return allPrinciples[locale] ?? allPrinciples["en"];
}

export const dataFlowSteps = [
  "Text", "Tokenizer", "Embedding", "Attention", "FFN",
  "Transformer Block", "Training", "Generation", "Classification",
  "LoRA", "SFT+DPO", "MoE"
];
