import type { PhaseContent } from "@/data/types";
import type { Locale } from "@/i18n";

// ═══════════════════════════════════════════════════════════════════
// Phase 3 & 4 — zh-TW
// ═══════════════════════════════════════════════════════════════════

const phase3ContentZhTW: PhaseContent = {
  phaseId: 3,
  color: "#EC4899",
  accent: "#F472B6",
  lessons: [
    {
      phaseId: 3, lessonId: 1,
      title: "Transformer Components",
      subtitle: "LayerNorm、GELU、FeedForward——組裝 Transformer Block 的核心零件",
      type: "concept",
      duration: "60 min",
      objectives: [
        "從零實作 Layer Normalization，理解其穩定訓練的原理",
        "實作 GELU 激活函數，比較其與 ReLU 的差異",
        "建構 FeedForward 網路，理解 4x 擴展比例的設計考量",
        "組裝完整的 Pre-Norm Transformer Block，包含殘差連接",
      ],
      sections: [
        {
          title: "Layer Normalization",
          blocks: [
            { type: "paragraph", text: "今天我們要造三個配件：LayerNorm（會議前的暖身）、GELU（思考時的開關）、FeedForward（獨立消化的過程），然後把它們和 Attention 組裝成一個完整的 Transformer Block。就像搭樂高——每個零件都不複雜，但組合起來就是一個強大的引擎。" },
            { type: "paragraph", text: "在深度神經網路中，每一層的輸入分布會隨訓練不斷變化（internal covariate shift）。Layer Normalization 透過對每個樣本的特徵維度進行正規化，穩定訓練過程、加速收斂。" },
            { type: "code", language: "python", code: "# LayerNorm 的數學定義\n# y = gamma * (x - mean) / sqrt(var + eps) + beta\n#\n# mean = x.mean(dim=-1, keepdim=True)\n# var  = x.var(dim=-1, keepdim=True, unbiased=False)\n#\n# gamma: 可學習的縮放參數，初始化為 1\n# beta:  可學習的偏移參數，初始化為 0\n# eps:   防止除零的小常數（通常 1e-5）" },
            { type: "heading", level: 3, text: "Pre-Norm vs Post-Norm" },
            { type: "table", headers: ["特性", "Post-Norm (原始 Transformer)", "Pre-Norm (GPT-2)"], rows: [
              ["LayerNorm 位置", "子層之後", "子層之前"],
              ["訓練穩定性", "較差，需要 warmup", "較好，梯度流更順暢"],
              ["代表模型", "BERT, 原始 Transformer", "GPT-2, GPT-3"],
            ]},
            { type: "callout", variant: "quote", text: "💡 講師心得：Pre-Norm vs Post-Norm 不只是學術爭論。GPT-2 之後幾乎所有主流模型（GPT-3、LLaMA、Mistral）都用 Pre-Norm，原因是它讓殘差流（residual stream）保持「乾淨」——資訊可以暢通無阻地流過整個網路。" },
          ],
        },
        {
          title: "GELU Activation",
          blocks: [
            { type: "paragraph", text: "GELU（Gaussian Error Linear Unit）是 GPT 系列模型使用的激活函數。與 ReLU 直接截斷負值不同，GELU 對輸入施加一個隨輸入值平滑變化的「門控」。" },
            { type: "code", language: "python", code: "# GPT-2 使用的近似形式：\n# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))" },
            { type: "diagram", content: "GELU vs ReLU:\n\n  輸出 ▲\n   3  │            ╱ ReLU\n      │           ╱\n   2  │          ╱    ·· GELU\n      │         ╱  ··\n   1  │        ╱··\n      │      ·╱\n   0  │···──╱─────────── 輸入\n      │  · ╱\n  -0.2│·· ╱\n\nReLU: max(0, x)         — 硬截斷\nGELU: 0.5x(1+tanh(...)) — 平滑門控" },
          ],
        },
        {
          title: "FeedForward Network",
          blocks: [
            { type: "paragraph", text: "Transformer 中的 FeedForward Network（FFN）是一個簡單的兩層 MLP，應用在每個 token 的表示上（position-wise）。標準設計是：第一層將維度從 d_model 擴展到 4 * d_model，經過 GELU 激活，第二層再壓縮回 d_model。" },
            { type: "code", language: "python", code: "# FFN 結構：\n# Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model) → Dropout" },
            { type: "callout", variant: "tip", text: "💡 講師心得：如果說 Attention 是 token 之間的「溝通」，FFN 就是每個 token 的「思考」。最新研究發現 FFN 層實際上充當了一個巨大的鍵值記憶體——第一層矩陣是 key（匹配模式），第二層是 value（存儲知識）。" },
          ],
        },
        {
          title: "TransformerBlock 組裝",
          blocks: [
            { type: "diagram", content: "Pre-Norm Transformer Block:\n\n  輸入 x ─────────────────────────────┐\n    │                                  │ (殘差連接 1)\n    ▼                                  │\n  LayerNorm (ln1)                      │\n    ▼                                  │\n  Multi-Head Attention                  │\n    ▼                                  │\n  Dropout                             │\n    ▼                                  │\n  ╔══ x = x + out ══╗◄────────────────┘\n    │ ─────────────────────────────┐\n    ▼                               │ (殘差連接 2)\n  LayerNorm (ln2)                   │\n    ▼                               │\n  FeedForward                       │\n    ▼                               │\n  Dropout                          │\n    ▼                               │\n  ╔══ x = x + out ══╗◄─────────────┘\n    ▼\n  輸出 x" },
            { type: "code", language: "python", code: "# Pre-Norm Transformer Block 的 forward 邏輯：\n# def forward(self, x):\n#     x = x + self.dropout(self.attn(self.ln1(x)))\n#     x = x + self.dropout(self.ffn(self.ln2(x)))\n#     return x" },
            { type: "callout", variant: "warning", text: "殘差連接的加法對象是原始的 x，不是經過 LayerNorm 的 x。Pre-Norm 的精髓在於：LN 只影響子層的輸入，不影響殘差路徑。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們的 Transformer Block 用 Pre-Norm。但如果你仔細看最後的 output head 前面，還有一個額外的 LayerNorm——為什麼最後還要加一層？如果去掉它會怎樣？" },
          ],
        },
      ],
      exercises: [
        { id: "layernorm", title: "TODO 1: LayerNorm", description: "從零實作 Layer Normalization。在 __init__ 中建立可學習的 gamma（ones）和 beta（zeros）參數。在 forward 中計算 mean、variance，正規化後套用 affine transform。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["gamma 使用 nn.Parameter(torch.ones(d_model))", "mean 和 var 都在 dim=-1 上計算，keepdim=True", "使用 unbiased=False 計算 variance"], pseudocode: "__init__:\n  self.gamma = nn.Parameter(ones(d_model))\n  self.beta = nn.Parameter(zeros(d_model))\n\nforward(x):\n  mean = x.mean(dim=-1, keepdim=True)\n  var = x.var(dim=-1, keepdim=True, unbiased=False)\n  x_norm = (x - mean) / sqrt(var + eps)\n  return gamma * x_norm + beta" },
        { id: "gelu", title: "TODO 2: GELU Activation", description: "實作 GELU 的近似公式：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["math.sqrt(2.0 / math.pi) ≈ 0.7978845608", "一行就能寫完"] },
        { id: "feedforward", title: "TODO 3: FeedForward Network", description: "建構兩層 MLP：Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout。d_ff 預設為 4 * d_model。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["如果 d_ff 為 None，設為 4 * d_model", "可以用 nn.Sequential 串接所有層"] },
        { id: "transformer_block", title: "TODO 4: TransformerBlock", description: "組裝 Pre-Norm Transformer Block：兩個子層各有 LayerNorm + 殘差連接。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["需要兩個 LayerNorm 實例（ln1, ln2）", "殘差連接：加法的左邊是原始 x"] },
      ],
      acceptanceCriteria: [
        "LayerNorm 輸出的 mean ≈ 0、variance ≈ 1（在 feature dimension 上）",
        "GELU(0) = 0，GELU(x) > 0 for large positive x",
        "FeedForward 輸入輸出 shape 相同 (batch, seq_len, d_model)",
        "TransformerBlock 輸出 shape == 輸入 shape",
      ],
      references: [
        { title: "Layer Normalization", description: "Ba et al. 2016 — LayerNorm 的原始論文", url: "https://arxiv.org/abs/1607.06450" },
        { title: "Gaussian Error Linear Units (GELUs)", description: "Hendrycks & Gimpel 2016 — GELU 的原始論文", url: "https://arxiv.org/abs/1606.08415" },
        { title: "On Layer Normalization in the Transformer Architecture", description: "Xiong et al. 2020 — 深入比較 Pre-Norm 和 Post-Norm", url: "https://arxiv.org/abs/2002.04745" },
      ],
    },
    {
      phaseId: 3, lessonId: 2,
      title: "GPT Model Assembly",
      subtitle: "將所有元件組裝成完整的 GPT 語言模型",
      type: "concept",
      duration: "45 min",
      objectives: [
        "定義 GPTConfig dataclass，管理所有模型超參數",
        "組裝 token embedding + position embedding + N 層 Transformer Block + lm_head",
        "理解並實作 weight tying 技術",
        "實作完整的 forward pass，包含可選的 loss 計算",
      ],
      sections: [
        {
          title: "GPTConfig 設計",
          blocks: [
            { type: "paragraph", text: "前三課的零件已經全部造好了：Tokenizer 把文字變成數字，Embedding 把數字變成向量，Attention 讓向量互相交流，FFN 讓每個向量獨立思考。現在，就像組裝一台電腦——把它們接到主機板上，變成一台能開機的完整機器。" },
            { type: "code", language: "python", code: "@dataclass\nclass GPTConfig:\n    vocab_size: int = 50257    # GPT-2 BPE 詞彙表大小\n    block_size: int = 1024     # 最大序列長度\n    d_model:    int = 768      # embedding 維度\n    n_heads:    int = 12       # attention head 數量\n    n_layers:   int = 12       # Transformer block 層數\n    dropout:  float = 0.1      # dropout 機率" },
            { type: "table", headers: ["模型", "d_model", "n_heads", "n_layers", "參數量"], rows: [
              ["GPT-2 Small", "768", "12", "12", "124M"],
              ["GPT-2 Medium", "1024", "16", "24", "355M"],
              ["GPT-2 Large", "1280", "20", "36", "774M"],
              ["GPT-2 XL", "1600", "25", "48", "1558M"],
            ]},
          ],
        },
        {
          title: "GPT 完整架構",
          blocks: [
            { type: "diagram", content: "GPT 完整架構：\n\n  Token IDs → Token Embedding + Position Embedding\n      │\n      ▼ Dropout\n  ┌──────────────────────────────┐\n  │  Transformer Block × N       │\n  │  (LN → Attention + LN → FFN)│\n  └──────────────────────────────┘\n      │\n      ▼ Final LayerNorm\n      ▼ lm_head: Linear(d_model → vocab)\n  logits: (B, T, vocab_size)" },
            { type: "callout", variant: "quote", text: "💡 講師心得：GPT 模型的結構驚人地簡單——只是把 Phase 1-3 的組件堆疊起來。工程上的一個重要原則：能用簡單結構解決的問題，不要用複雜結構。GPT 的成功證明了，真正重要的不是花哨的架構，而是足夠大的規模和足夠多的資料。" },
          ],
        },
        {
          title: "Weight Tying",
          blocks: [
            { type: "paragraph", text: "Weight tying（權重共享）是 GPT 中一個重要的技巧：token embedding 矩陣和 lm_head 的線性投影矩陣共用同一組權重。" },
            { type: "code", language: "python", code: "# Weight tying:\n# self.lm_head.weight = self.token_embedding.weight\n#\n# 效果：減少 vocab_size * d_model 個參數\n# (GPT-2: 50257 * 768 ≈ 38.6M 參數節省)" },
            { type: "callout", variant: "tip", text: "💡 講師心得：embedding 層學到的是「token → 語義向量」的映射，而 output head 做的是「語義向量 → token 概率」的反向映射。這兩個映射本質上是同一件事的正反面，所以共享權重不僅省參數，還能提升效果。" },
          ],
        },
        {
          title: "Forward Pass 與 Loss 計算",
          blocks: [
            { type: "code", language: "python", code: "def forward(self, idx, targets=None):\n    B, T = idx.shape\n    tok_emb = self.token_embedding(idx)           # (B, T, d_model)\n    pos_emb = self.pos_embedding(torch.arange(T)) # (T, d_model)\n    x = self.dropout(tok_emb + pos_emb)\n    for block in self.blocks:\n        x = block(x)\n    x = self.ln_f(x)\n    logits = self.lm_head(x)  # (B, T, vocab_size)\n    loss = None\n    if targets is not None:\n        loss = F.cross_entropy(\n            logits.view(-1, logits.size(-1)),\n            targets.view(-1)\n        )\n    return logits, loss" },
            { type: "callout", variant: "quote", text: "🤔 思考題：GPT-2 Small 有 124M 參數。如果把層數從 12 增加到 24（其他不變），參數量大概翻倍。但如果把 d_model 從 768 增加到 1536（其他不變），參數量會變成多少？哪種擴展方式更有效率？" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_config", title: "TODO 1: GPTConfig", description: "定義 GPTConfig dataclass，包含 vocab_size、block_size、d_model、n_heads、n_layers、dropout 六個欄位及其預設值。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["使用 @dataclass 裝飾器", "GPT-2 預設值：vocab_size=50257, d_model=768, n_heads=12, n_layers=12"] },
        { id: "gpt_init", title: "TODO 2: GPT.__init__", description: "初始化 GPT 模型的所有組件：token/position embedding、dropout、N 個 TransformerBlock、final LayerNorm、lm_head，並實作 weight tying。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["nn.ModuleList 讓 PyTorch 能正確追蹤所有子模組的參數", "weight tying: self.lm_head.weight = self.token_embedding.weight", "lm_head 的 bias=False"] },
        { id: "gpt_forward", title: "TODO 3: GPT.forward", description: "實作 forward pass：embeddings → dropout → TransformerBlock × N → Final LN → lm_head → logits。如果提供 targets，計算 cross-entropy loss。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["assert T <= self.config.block_size", "position indices: torch.arange(T, device=idx.device)", "loss 用 cross_entropy(logits.view(-1, vocab_size), targets.view(-1))"] },
        { id: "count_parameters", title: "TODO 4: count_parameters", description: "計算模型中所有可訓練參數的總數。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["p.numel() 返回 tensor 中元素的數量", "只計算 requires_grad=True 的參數"], pseudocode: "return sum(p.numel() for p in self.parameters() if p.requires_grad)" },
      ],
      acceptanceCriteria: [
        "GPT forward 輸出 logits shape 為 (B, T, vocab_size)",
        "不提供 targets 時 loss 為 None",
        "提供 targets 時 loss 為正的 scalar",
        "count_parameters 返回正整數",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "Radford et al. 2019 — GPT-2 論文，完整架構描述", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
        { title: "nanoGPT", description: "Andrej Karpathy 的極簡 GPT 實作", url: "https://github.com/karpathy/nanoGPT" },
      ],
    },
  ],
};

const phase4ContentZhTW: PhaseContent = {
  phaseId: 4,
  color: "#F59E0B",
  accent: "#FBBF24",
  lessons: [
    {
      phaseId: 4, lessonId: 1,
      title: "Training Loop",
      subtitle: "從 Loss 函數到完整訓練迴圈——讓 GPT 真正開始學習",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 cross-entropy loss 在語言模型中的意義",
        "實作 cosine learning rate schedule with linear warmup",
        "實作 loss 估計函數與 checkpoint 儲存/載入",
        "建構完整的訓練迴圈，整合 AdamW、gradient clipping、evaluation",
      ],
      sections: [
        {
          title: "Cross-Entropy Loss",
          blocks: [
            { type: "paragraph", text: "模型組裝完成了，但現在它什麼都不會——所有參數都是隨機初始化的。Pre-training 就是讓這個「嬰兒」讀遍整個網際網路，從中學會語言的規律。今天我們要搭建完整的訓練管線。" },
            { type: "paragraph", text: "語言模型的訓練目標是最大化正確 token 的預測機率。Cross-entropy loss 衡量模型輸出的機率分佈與真實 token 的 one-hot 分佈之間的差異。對隨機初始化的模型來說，初始 loss ≈ ln(vocab_size)。" },
            { type: "callout", variant: "info", text: "Perplexity = exp(loss) 是語言模型常用的評估指標。Perplexity 可以直覺地理解為：模型在每個位置平均「猶豫」於多少個選項。" },
          ],
        },
        {
          title: "AdamW Optimizer",
          blocks: [
            { type: "table", headers: ["特性", "SGD", "Adam", "AdamW"], rows: [
              ["自適應學習率", "否", "是", "是"],
              ["Weight Decay", "等同 L2", "與 LR 耦合", "解耦"],
              ["Transformer 適用性", "差", "好", "最佳"],
            ]},
            { type: "code", language: "python", code: "optimizer = torch.optim.AdamW(\n    model.parameters(),\n    lr=3e-4,\n    betas=(0.9, 0.999),\n    eps=1e-8,\n    weight_decay=0.01,\n)" },
            { type: "callout", variant: "tip", text: "💡 講師心得：AdamW 的「W」（decoupled weight decay）是一個常被忽略但很重要的細節。在原始 Adam 中，L2 正則化的效果會被 Adam 的自適應學習率縮放掉。Loshchilov & Hutter 發現，直接在權重上施加 decay 效果好得多。" },
          ],
        },
        {
          title: "Cosine Learning Rate Schedule",
          blocks: [
            { type: "code", language: "python", code: "def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):\n    # 階段 1: Linear Warmup\n    #   lr = max_lr * (step / warmup_steps)\n    # 階段 2: Cosine Decay\n    #   progress = (step - warmup_steps) / (max_steps - warmup_steps)\n    #   lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))\n    # 階段 3: 超過 max_steps → lr = min_lr" },
            { type: "diagram", content: "Cosine LR Schedule with Warmup:\n\n  lr ▲\nmax ─│        ╭──╮\n     │       ╱    ╲\n     │      ╱      ╲\n     │     ╱        ╲\nmin ─│╱                ╲──────────\n     └────┬──────────────┬──────────► step\n       warmup         max_steps" },
          ],
        },
        {
          title: "Loss Estimation 與 Checkpointing",
          blocks: [
            { type: "code", language: "python", code: "@torch.no_grad()\ndef estimate_loss(model, dataloader, eval_steps, device='cpu'):\n    model.eval()\n    losses = []\n    for i, (x, y) in enumerate(dataloader):\n        if i >= eval_steps: break\n        x, y = x.to(device), y.to(device)\n        _, loss = model(x, y)\n        losses.append(loss.item())\n    model.train()\n    return sum(losses) / len(losses)" },
            { type: "code", language: "python", code: "def save_checkpoint(model, optimizer, step, loss, path):\n    Path(path).parent.mkdir(parents=True, exist_ok=True)\n    torch.save({\n        'model_state_dict': model.state_dict(),\n        'optimizer_state_dict': optimizer.state_dict(),\n        'step': step, 'loss': loss,\n    }, path)" },
          ],
        },
        {
          title: "完整訓練迴圈",
          blocks: [
            { type: "code", language: "python", code: "@dataclass\nclass TrainConfig:\n    max_steps: int = 1000\n    batch_size: int = 4\n    learning_rate: float = 3e-4\n    min_lr: float = 3e-5\n    warmup_steps: int = 100\n    eval_interval: int = 100\n    eval_steps: int = 10\n    checkpoint_dir: str = 'checkpoints'\n    checkpoint_interval: int = 500\n    max_grad_norm: float = 1.0\n    device: str = 'cpu'" },
            { type: "callout", variant: "warning", text: "訓練迴圈中最常見的 bug：忘記 zero_grad()（梯度累積）、eval 後忘記切回 train mode、LR schedule 的 step 計數錯誤。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們的 loss 函數是 cross-entropy，它衡量的是模型對「下一個 token」的預測能力。在訓練初期，序列開頭的 token 的 loss 會比結尾的高還是低？為什麼？" },
          ],
        },
      ],
      exercises: [
        { id: "get_lr", title: "TODO 1: get_lr (cosine schedule)", description: "實作 cosine learning rate schedule with linear warmup。三個階段：linear warmup → cosine decay → constant min_lr。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["step >= max_steps 時直接返回 min_lr", "warmup 階段：lr = max_lr * (step / warmup_steps)", "cosine 公式：min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))"] },
        { id: "estimate_loss", title: "TODO 2: estimate_loss", description: "在 eval mode 下計算平均 loss。設定 model.eval()，遍歷至多 eval_steps 個 batch，計算平均 loss，最後恢復 model.train()。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["@torch.no_grad() 裝飾器", "記得在結束時恢復 model.train()", "loss.item() 將 tensor 轉為 Python float"] },
        { id: "checkpoints", title: "TODO 3: save_checkpoint / load_checkpoint", description: "實作 checkpoint 的儲存與載入。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["save: Path(path).parent.mkdir(parents=True, exist_ok=True)", "load: torch.load(path, map_location='cpu', weights_only=True)"] },
        { id: "train_loop", title: "TODO 4: TrainConfig + train()", description: "定義 TrainConfig dataclass 和完整的 train() 函數。", labFile: "labs/phase4_pretraining/phase_4/train.py", hints: ["用 itertools.cycle(train_loader) 處理 DataLoader 循環", "LR 更新：optimizer.param_groups[0]['lr'] = get_lr(...)", "順序：get_lr → forward → backward → clip_grad_norm_ → step → zero_grad"] },
      ],
      acceptanceCriteria: [
        "get_lr 在 step=0 時返回 0，warmup 結束時返回 max_lr",
        "estimate_loss 結束後模型恢復 train mode",
        "checkpoint roundtrip 後模型權重完全一致",
        "train() 完成後模型權重確實改變",
      ],
      references: [
        { title: "Decoupled Weight Decay Regularization", description: "Loshchilov & Hutter 2019 — AdamW 論文", url: "https://arxiv.org/abs/1711.05101" },
        { title: "Chinchilla", description: "Hoffmann et al. 2022 — 探討最佳的 training tokens 數量與模型大小的比例", url: "https://arxiv.org/abs/2203.15556" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// zh-CN variants
// ═══════════════════════════════════════════════════════════════════

const phase3ContentZhCN: PhaseContent = {
  phaseId: 3,
  color: "#EC4899",
  accent: "#F472B6",
  lessons: [
    {
      phaseId: 3, lessonId: 1,
      title: "Transformer Components",
      subtitle: "LayerNorm、GELU、FeedForward——组装 Transformer Block 的核心零件",
      type: "concept",
      duration: "60 min",
      objectives: [
        "从零实现 Layer Normalization，理解其稳定训练的原理",
        "实现 GELU 激活函数，比较其与 ReLU 的差异",
        "构建 FeedForward 网络，理解 4x 扩展比例的设计考量",
        "组装完整的 Pre-Norm Transformer Block，包含残差连接",
      ],
      sections: [
        {
          title: "Layer Normalization",
          blocks: [
            { type: "paragraph", text: "今天我们要造三个配件：LayerNorm（会议前的热身）、GELU（思考时的开关）、FeedForward（独立消化的过程），然后把它们和 Attention 组装成一个完整的 Transformer Block。就像搭乐高——每个零件都不复杂，但组合起来就是一个强大的引擎。" },
            { type: "paragraph", text: "在深度神经网络中，每一层的输入分布会随训练不断变化（internal covariate shift）。Layer Normalization 通过对每个样本的特征维度进行归一化，稳定训练过程、加速收敛。" },
            { type: "code", language: "python", code: "# LayerNorm 的数学定义\n# y = gamma * (x - mean) / sqrt(var + eps) + beta\n#\n# mean = x.mean(dim=-1, keepdim=True)\n# var  = x.var(dim=-1, keepdim=True, unbiased=False)\n#\n# gamma: 可学习的缩放参数，初始化为 1\n# beta:  可学习的偏移参数，初始化为 0\n# eps:   防止除零的小常数（通常 1e-5）" },
            { type: "heading", level: 3, text: "Pre-Norm vs Post-Norm" },
            { type: "table", headers: ["特性", "Post-Norm (原始 Transformer)", "Pre-Norm (GPT-2)"], rows: [
              ["LayerNorm 位置", "子层之后", "子层之前"],
              ["训练稳定性", "较差，需要 warmup", "较好，梯度流更顺畅"],
              ["代表模型", "BERT, 原始 Transformer", "GPT-2, GPT-3"],
            ]},
            { type: "callout", variant: "quote", text: "💡 讲师心得：Pre-Norm vs Post-Norm 不只是学术争论。GPT-2 之后几乎所有主流模型（GPT-3、LLaMA、Mistral）都用 Pre-Norm，原因是它让残差流（residual stream）保持「干净」——信息可以畅通无阻地流过整个网络。" },
          ],
        },
        {
          title: "GELU Activation",
          blocks: [
            { type: "paragraph", text: "GELU（Gaussian Error Linear Unit）是 GPT 系列模型使用的激活函数。与 ReLU 直接截断负值不同，GELU 对输入施加一个随输入值平滑变化的「门控」。" },
            { type: "code", language: "python", code: "# GPT-2 使用的近似形式：\n# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))" },
            { type: "diagram", content: "GELU vs ReLU:\n\n  输出 ▲\n   3  │            ╱ ReLU\n      │           ╱\n   2  │          ╱    ·· GELU\n      │         ╱  ··\n   1  │        ╱··\n      │      ·╱\n   0  │···──╱─────────── 输入\n      │  · ╱\n  -0.2│·· ╱\n\nReLU: max(0, x)         — 硬截断\nGELU: 0.5x(1+tanh(...)) — 平滑门控" },
          ],
        },
        {
          title: "FeedForward Network",
          blocks: [
            { type: "paragraph", text: "Transformer 中的 FeedForward Network（FFN）是一个简单的两层 MLP，应用在每个 token 的表示上（position-wise）。标准设计是：第一层将维度从 d_model 扩展到 4 * d_model，经过 GELU 激活，第二层再压缩回 d_model。" },
            { type: "code", language: "python", code: "# FFN 结构：\n# Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model) → Dropout" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：如果说 Attention 是 token 之间的「沟通」，FFN 就是每个 token 的「思考」。最新研究发现 FFN 层实际上充当了一个巨大的键值内存——第一层矩阵是 key（匹配模式），第二层是 value（存储知识）。" },
          ],
        },
        {
          title: "TransformerBlock 组装",
          blocks: [
            { type: "diagram", content: "Pre-Norm Transformer Block:\n\n  输入 x ─────────────────────────────┐\n    │                                  │ (残差连接 1)\n    ▼                                  │\n  LayerNorm (ln1)                      │\n    ▼                                  │\n  Multi-Head Attention                  │\n    ▼                                  │\n  Dropout                             │\n    ▼                                  │\n  ╔══ x = x + out ══╗◄────────────────┘\n    │ ─────────────────────────────┐\n    ▼                               │ (残差连接 2)\n  LayerNorm (ln2)                   │\n    ▼                               │\n  FeedForward                       │\n    ▼                               │\n  Dropout                          │\n    ▼                               │\n  ╔══ x = x + out ══╗◄─────────────┘\n    ▼\n  输出 x" },
            { type: "code", language: "python", code: "# Pre-Norm Transformer Block 的 forward 逻辑：\n# def forward(self, x):\n#     x = x + self.dropout(self.attn(self.ln1(x)))\n#     x = x + self.dropout(self.ffn(self.ln2(x)))\n#     return x" },
            { type: "callout", variant: "warning", text: "残差连接的加法对象是原始的 x，不是经过 LayerNorm 的 x。Pre-Norm 的精髓在于：LN 只影响子层的输入，不影响残差路径。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：我们的 Transformer Block 用 Pre-Norm。但如果你仔细看最后的 output head 前面，还有一个额外的 LayerNorm——为什么最后还要加一层？如果去掉它会怎样？" },
          ],
        },
      ],
      exercises: [
        { id: "layernorm", title: "TODO 1: LayerNorm", description: "从零实现 Layer Normalization。在 __init__ 中建立可学习的 gamma（ones）和 beta（zeros）参数。在 forward 中计算 mean、variance，归一化后套用 affine transform。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["gamma 使用 nn.Parameter(torch.ones(d_model))", "mean 和 var 都在 dim=-1 上计算，keepdim=True", "使用 unbiased=False 计算 variance"], pseudocode: "__init__:\n  self.gamma = nn.Parameter(ones(d_model))\n  self.beta = nn.Parameter(zeros(d_model))\n\nforward(x):\n  mean = x.mean(dim=-1, keepdim=True)\n  var = x.var(dim=-1, keepdim=True, unbiased=False)\n  x_norm = (x - mean) / sqrt(var + eps)\n  return gamma * x_norm + beta" },
        { id: "gelu", title: "TODO 2: GELU Activation", description: "实现 GELU 的近似公式：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["math.sqrt(2.0 / math.pi) ≈ 0.7978845608", "一行就能写完"] },
        { id: "feedforward", title: "TODO 3: FeedForward Network", description: "构建两层 MLP：Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout。d_ff 默认为 4 * d_model。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["如果 d_ff 为 None，设为 4 * d_model", "可以用 nn.Sequential 串接所有层"] },
        { id: "transformer_block", title: "TODO 4: TransformerBlock", description: "组装 Pre-Norm Transformer Block：两个子层各有 LayerNorm + 残差连接。", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["需要两个 LayerNorm 实例（ln1, ln2）", "残差连接：加法的左边是原始 x"] },
      ],
      acceptanceCriteria: [
        "LayerNorm 输出的 mean ≈ 0、variance ≈ 1（在 feature dimension 上）",
        "GELU(0) = 0，GELU(x) > 0 for large positive x",
        "FeedForward 输入输出 shape 相同 (batch, seq_len, d_model)",
        "TransformerBlock 输出 shape == 输入 shape",
      ],
      references: [
        { title: "Layer Normalization", description: "Ba et al. 2016 — LayerNorm 的原始论文", url: "https://arxiv.org/abs/1607.06450" },
        { title: "Gaussian Error Linear Units (GELUs)", description: "Hendrycks & Gimpel 2016 — GELU 的原始论文", url: "https://arxiv.org/abs/1606.08415" },
        { title: "On Layer Normalization in the Transformer Architecture", description: "Xiong et al. 2020 — 深入比较 Pre-Norm 和 Post-Norm", url: "https://arxiv.org/abs/2002.04745" },
      ],
    },
    {
      phaseId: 3, lessonId: 2,
      title: "GPT Model Assembly",
      subtitle: "将所有组件组装成完整的 GPT 语言模型",
      type: "concept",
      duration: "45 min",
      objectives: [
        "定义 GPTConfig dataclass，管理所有模型超参数",
        "组装 token embedding + position embedding + N 层 Transformer Block + lm_head",
        "理解并实现 weight tying 技术",
        "实现完整的 forward pass，包含可选的 loss 计算",
      ],
      sections: [
        {
          title: "GPTConfig 设计",
          blocks: [
            { type: "paragraph", text: "前三课的零件已经全部造好了：Tokenizer 把文字变成数字，Embedding 把数字变成向量，Attention 让向量互相交流，FFN 让每个向量独立思考。现在，就像组装一台电脑——把它们接到主板上，变成一台能开机的完整机器。" },
            { type: "code", language: "python", code: "@dataclass\nclass GPTConfig:\n    vocab_size: int = 50257    # GPT-2 BPE 词汇表大小\n    block_size: int = 1024     # 最大序列长度\n    d_model:    int = 768      # embedding 维度\n    n_heads:    int = 12       # attention head 数量\n    n_layers:   int = 12       # Transformer block 层数\n    dropout:  float = 0.1      # dropout 概率" },
            { type: "table", headers: ["模型", "d_model", "n_heads", "n_layers", "参数量"], rows: [
              ["GPT-2 Small", "768", "12", "12", "124M"],
              ["GPT-2 Medium", "1024", "16", "24", "355M"],
              ["GPT-2 Large", "1280", "20", "36", "774M"],
              ["GPT-2 XL", "1600", "25", "48", "1558M"],
            ]},
          ],
        },
        {
          title: "GPT 完整架构",
          blocks: [
            { type: "diagram", content: "GPT 完整架构：\n\n  Token IDs → Token Embedding + Position Embedding\n      │\n      ▼ Dropout\n  ┌──────────────────────────────┐\n  │  Transformer Block × N       │\n  │  (LN → Attention + LN → FFN)│\n  └──────────────────────────────┘\n      │\n      ▼ Final LayerNorm\n      ▼ lm_head: Linear(d_model → vocab)\n  logits: (B, T, vocab_size)" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：GPT 模型的结构惊人地简单——只是把 Phase 1-3 的组件堆叠起来。工程上的一个重要原则：能用简单结构解决的问题，不要用复杂结构。GPT 的成功证明了，真正重要的不是花哨的架构，而是足够大的规模和足够多的数据。" },
          ],
        },
        {
          title: "Weight Tying",
          blocks: [
            { type: "paragraph", text: "Weight tying（权重共享）是 GPT 中一个重要的技巧：token embedding 矩阵和 lm_head 的线性投影矩阵共用同一组权重。" },
            { type: "code", language: "python", code: "# Weight tying:\n# self.lm_head.weight = self.token_embedding.weight\n#\n# 效果：减少 vocab_size * d_model 个参数\n# (GPT-2: 50257 * 768 ≈ 38.6M 参数节省)" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：embedding 层学到的是「token → 语义向量」的映射，而 output head 做的是「语义向量 → token 概率」的反向映射。这两个映射本质上是同一件事的正反面，所以共享权重不仅省参数，还能提升效果。" },
          ],
        },
        {
          title: "Forward Pass 与 Loss 计算",
          blocks: [
            { type: "code", language: "python", code: "def forward(self, idx, targets=None):\n    B, T = idx.shape\n    tok_emb = self.token_embedding(idx)           # (B, T, d_model)\n    pos_emb = self.pos_embedding(torch.arange(T)) # (T, d_model)\n    x = self.dropout(tok_emb + pos_emb)\n    for block in self.blocks:\n        x = block(x)\n    x = self.ln_f(x)\n    logits = self.lm_head(x)  # (B, T, vocab_size)\n    loss = None\n    if targets is not None:\n        loss = F.cross_entropy(\n            logits.view(-1, logits.size(-1)),\n            targets.view(-1)\n        )\n    return logits, loss" },
            { type: "callout", variant: "quote", text: "🤔 思考题：GPT-2 Small 有 124M 参数。如果把层数从 12 增加到 24（其他不变），参数量大概翻倍。但如果把 d_model 从 768 增加到 1536（其他不变），参数量会变成多少？哪种扩展方式更有效率？" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_config", title: "TODO 1: GPTConfig", description: "定义 GPTConfig dataclass，包含 vocab_size、block_size、d_model、n_heads、n_layers、dropout 六个字段及其默认值。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["使用 @dataclass 装饰器", "GPT-2 默认值：vocab_size=50257, d_model=768, n_heads=12, n_layers=12"] },
        { id: "gpt_init", title: "TODO 2: GPT.__init__", description: "初始化 GPT 模型的所有组件，并实现 weight tying。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["nn.ModuleList 让 PyTorch 能正确追踪所有子模块的参数", "weight tying: self.lm_head.weight = self.token_embedding.weight"] },
        { id: "gpt_forward", title: "TODO 3: GPT.forward", description: "实现 forward pass：embeddings → dropout → TransformerBlock × N → Final LN → lm_head → logits。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["assert T <= self.config.block_size", "loss 用 cross_entropy(logits.view(-1, vocab_size), targets.view(-1))"] },
        { id: "count_parameters", title: "TODO 4: count_parameters", description: "计算模型中所有可训练参数的总数。", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["p.numel() 返回 tensor 中元素的数量", "只计算 requires_grad=True 的参数"], pseudocode: "return sum(p.numel() for p in self.parameters() if p.requires_grad)" },
      ],
      acceptanceCriteria: [
        "GPT forward 输出 logits shape 为 (B, T, vocab_size)",
        "不提供 targets 时 loss 为 None",
        "提供 targets 时 loss 为正的 scalar",
        "count_parameters 返回正整数",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "Radford et al. 2019 — GPT-2 论文", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
        { title: "nanoGPT", description: "Andrej Karpathy 的极简 GPT 实现", url: "https://github.com/karpathy/nanoGPT" },
      ],
    },
  ],
};

const phase4ContentZhCN: PhaseContent = {
  phaseId: 4,
  color: "#F59E0B",
  accent: "#FBBF24",
  lessons: [
    {
      phaseId: 4, lessonId: 1,
      title: "Training Loop",
      subtitle: "从 Loss 函数到完整训练循环——让 GPT 真正开始学习",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 cross-entropy loss 在语言模型中的意义",
        "实现 cosine learning rate schedule with linear warmup",
        "实现 loss 估计函数与 checkpoint 存储/加载",
        "构建完整的训练循环，整合 AdamW、gradient clipping、evaluation",
      ],
      sections: [
        {
          title: "Cross-Entropy Loss",
          blocks: [
            { type: "paragraph", text: "模型组装完成了，但现在它什么都不会——所有参数都是随机初始化的。Pre-training 就是让这个「婴儿」读遍整个互联网，从中学会语言的规律。今天我们要搭建完整的训练管道。" },
            { type: "paragraph", text: "语言模型的训练目标是最大化正确 token 的预测概率。Cross-entropy loss 衡量模型输出的概率分布与真实 token 的 one-hot 分布之间的差异。对随机初始化的模型来说，初始 loss ≈ ln(vocab_size)。" },
            { type: "callout", variant: "info", text: "Perplexity = exp(loss) 是语言模型常用的评估指标。Perplexity 可以直观地理解为：模型在每个位置平均「犹豫」于多少个选项。" },
          ],
        },
        {
          title: "AdamW Optimizer",
          blocks: [
            { type: "table", headers: ["特性", "SGD", "Adam", "AdamW"], rows: [
              ["自适应学习率", "否", "是", "是"],
              ["Weight Decay", "等同 L2", "与 LR 耦合", "解耦"],
              ["Transformer 适用性", "差", "好", "最佳"],
            ]},
            { type: "code", language: "python", code: "optimizer = torch.optim.AdamW(\n    model.parameters(),\n    lr=3e-4,\n    betas=(0.9, 0.999),\n    eps=1e-8,\n    weight_decay=0.01,\n)" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：AdamW 的「W」（decoupled weight decay）是一个常被忽略但很重要的细节。在原始 Adam 中，L2 正则化的效果会被 Adam 的自适应学习率缩放掉。Loshchilov & Hutter 发现，直接在权重上施加 decay 效果好得多。" },
          ],
        },
        {
          title: "Cosine Learning Rate Schedule",
          blocks: [
            { type: "code", language: "python", code: "def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):\n    # 阶段 1: Linear Warmup\n    #   lr = max_lr * (step / warmup_steps)\n    # 阶段 2: Cosine Decay\n    #   progress = (step - warmup_steps) / (max_steps - warmup_steps)\n    #   lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))\n    # 阶段 3: 超过 max_steps → lr = min_lr" },
            { type: "diagram", content: "Cosine LR Schedule with Warmup:\n\n  lr ▲\nmax ─│        ╭──╮\n     │       ╱    ╲\n     │      ╱      ╲\n     │     ╱        ╲\nmin ─│╱                ╲──────────\n     └────┬──────────────┬──────────► step\n       warmup         max_steps" },
          ],
        },
        {
          title: "Loss Estimation 与 Checkpointing",
          blocks: [
            { type: "code", language: "python", code: "@torch.no_grad()\ndef estimate_loss(model, dataloader, eval_steps, device='cpu'):\n    model.eval()\n    losses = []\n    for i, (x, y) in enumerate(dataloader):\n        if i >= eval_steps: break\n        x, y = x.to(device), y.to(device)\n        _, loss = model(x, y)\n        losses.append(loss.item())\n    model.train()\n    return sum(losses) / len(losses)" },
            { type: "code", language: "python", code: "def save_checkpoint(model, optimizer, step, loss, path):\n    Path(path).parent.mkdir(parents=True, exist_ok=True)\n    torch.save({\n        'model_state_dict': model.state_dict(),\n        'optimizer_state_dict': optimizer.state_dict(),\n        'step': step, 'loss': loss,\n    }, path)" },
          ],
        },
        {
          title: "完整训练循环",
          blocks: [
            { type: "code", language: "python", code: "@dataclass\nclass TrainConfig:\n    max_steps: int = 1000\n    batch_size: int = 4\n    learning_rate: float = 3e-4\n    min_lr: float = 3e-5\n    warmup_steps: int = 100\n    eval_interval: int = 100\n    eval_steps: int = 10\n    checkpoint_dir: str = 'checkpoints'\n    checkpoint_interval: int = 500\n    max_grad_norm: float = 1.0\n    device: str = 'cpu'" },
            { type: "callout", variant: "warning", text: "训练循环中最常见的 bug：忘记 zero_grad()（梯度累积）、eval 后忘记切回 train mode、LR schedule 的 step 计数错误。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：我们的 loss 函数是 cross-entropy，它衡量的是模型对「下一个 token」的预测能力。在训练初期，序列开头的 token 的 loss 会比结尾的高还是低？为什么？" },
          ],
        },
      ],
      exercises: [
        { id: "get_lr", title: "TODO 1: get_lr (cosine schedule)", description: "实现 cosine learning rate schedule with linear warmup。三个阶段：linear warmup → cosine decay → constant min_lr。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["step >= max_steps 时直接返回 min_lr", "warmup 阶段：lr = max_lr * (step / warmup_steps)", "cosine 公式：min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))"] },
        { id: "estimate_loss", title: "TODO 2: estimate_loss", description: "在 eval mode 下计算平均 loss。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["@torch.no_grad() 装饰器", "记得在结束时恢复 model.train()", "loss.item() 将 tensor 转为 Python float"] },
        { id: "checkpoints", title: "TODO 3: save_checkpoint / load_checkpoint", description: "实现 checkpoint 的存储与加载。", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["save: Path(path).parent.mkdir(parents=True, exist_ok=True)", "load: torch.load(path, map_location='cpu', weights_only=True)"] },
        { id: "train_loop", title: "TODO 4: TrainConfig + train()", description: "定义 TrainConfig dataclass 和完整的 train() 函数。", labFile: "labs/phase4_pretraining/phase_4/train.py", hints: ["用 itertools.cycle(train_loader) 处理 DataLoader 循环", "LR 更新：optimizer.param_groups[0]['lr'] = get_lr(...)"] },
      ],
      acceptanceCriteria: [
        "get_lr 在 step=0 时返回 0，warmup 结束时返回 max_lr",
        "estimate_loss 结束后模型恢复 train mode",
        "checkpoint roundtrip 后模型权重完全一致",
        "train() 完成后模型权重确实改变",
      ],
      references: [
        { title: "Decoupled Weight Decay Regularization", description: "Loshchilov & Hutter 2019 — AdamW 论文", url: "https://arxiv.org/abs/1711.05101" },
        { title: "Chinchilla", description: "Hoffmann et al. 2022 — 探讨最优的 training tokens 数量与模型大小的比例", url: "https://arxiv.org/abs/2203.15556" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// English variants
// ═══════════════════════════════════════════════════════════════════

const phase3ContentEn: PhaseContent = {
  phaseId: 3,
  color: "#EC4899",
  accent: "#F472B6",
  lessons: [
    {
      phaseId: 3, lessonId: 1,
      title: "Transformer Components",
      subtitle: "LayerNorm, GELU, FeedForward — Assembling the Core Parts of a Transformer Block",
      type: "concept",
      duration: "60 min",
      objectives: [
        "Implement Layer Normalization from scratch and understand why it stabilizes training",
        "Implement the GELU activation function and compare it with ReLU",
        "Build a FeedForward network and understand the 4x expansion design",
        "Assemble a complete Pre-Norm Transformer Block with residual connections",
      ],
      sections: [
        {
          title: "Layer Normalization",
          blocks: [
            { type: "paragraph", text: "Today we're building three components: LayerNorm (the pre-meeting warm-up), GELU (the thinking switch), and FeedForward (the individual digestion process) — then assembling them with Attention into a complete Transformer Block. Like LEGO bricks: each piece is simple, but together they form a powerful engine." },
            { type: "paragraph", text: "In deep neural networks, the input distribution of each layer shifts continuously during training (internal covariate shift). Layer Normalization stabilizes the training process and accelerates convergence by normalizing across the feature dimension for each sample." },
            { type: "code", language: "python", code: "# LayerNorm mathematical definition\n# y = gamma * (x - mean) / sqrt(var + eps) + beta\n#\n# mean = x.mean(dim=-1, keepdim=True)\n# var  = x.var(dim=-1, keepdim=True, unbiased=False)\n#\n# gamma: learnable scale parameter, initialized to 1\n# beta:  learnable shift parameter, initialized to 0\n# eps:   small constant to prevent division by zero (typically 1e-5)" },
            { type: "heading", level: 3, text: "Pre-Norm vs Post-Norm" },
            { type: "table", headers: ["Property", "Post-Norm (original Transformer)", "Pre-Norm (GPT-2)"], rows: [
              ["LayerNorm position", "After sublayer", "Before sublayer"],
              ["Training stability", "Poorer, needs warmup", "Better, smoother gradient flow"],
              ["Representative models", "BERT, original Transformer", "GPT-2, GPT-3"],
            ]},
            { type: "callout", variant: "quote", text: "Instructor's Note: Pre-Norm vs Post-Norm isn't just an academic debate. Almost every major model after GPT-2 (GPT-3, LLaMA, Mistral) uses Pre-Norm because it keeps the residual stream \"clean\" — information can flow unobstructed through the entire network." },
          ],
        },
        {
          title: "GELU Activation",
          blocks: [
            { type: "paragraph", text: "GELU (Gaussian Error Linear Unit) is the activation function used by GPT models. Unlike ReLU which hard-clips negative values, GELU applies a smooth gate that varies with the input value." },
            { type: "code", language: "python", code: "# The approximation used by GPT-2:\n# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))" },
            { type: "diagram", content: "GELU vs ReLU:\n\n  Output ▲\n   3  │            ╱ ReLU\n      │           ╱\n   2  │          ╱    ·· GELU\n      │         ╱  ··\n   1  │        ╱··\n      │      ·╱\n   0  │···──╱─────────── Input\n      │  · ╱\n  -0.2│·· ╱\n\nReLU: max(0, x)         — hard clip\nGELU: 0.5x(1+tanh(...)) — smooth gate" },
          ],
        },
        {
          title: "FeedForward Network",
          blocks: [
            { type: "paragraph", text: "The FeedForward Network (FFN) in Transformers is a simple two-layer MLP applied position-wise (independently to each token). The standard design: expand from d_model to 4*d_model, apply GELU, then project back to d_model." },
            { type: "code", language: "python", code: "# FFN structure:\n# Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model) → Dropout" },
            { type: "callout", variant: "tip", text: "Instructor's Note: If Attention is \"communication\" between tokens, FFN is each token's \"thinking.\" Recent research (Geva et al. 2021) found that FFN layers actually act as a giant key-value memory: the first layer is keys (pattern matching), the second is values (knowledge storage). That's where the model stores facts like \"Paris is the capital of France.\"" },
          ],
        },
        {
          title: "Assembling the TransformerBlock",
          blocks: [
            { type: "diagram", content: "Pre-Norm Transformer Block:\n\n  Input x ─────────────────────────────┐\n    │                                   │ (residual 1)\n    ▼                                   │\n  LayerNorm (ln1)                       │\n    ▼                                   │\n  Multi-Head Attention                   │\n    ▼                                   │\n  Dropout                              │\n    ▼                                   │\n  ╔══ x = x + out ══╗◄─────────────────┘\n    │ ──────────────────────────────┐\n    ▼                               │ (residual 2)\n  LayerNorm (ln2)                   │\n    ▼                               │\n  FeedForward                       │\n    ▼                               │\n  Dropout                          │\n    ▼                               │\n  ╔══ x = x + out ══╗◄─────────────┘\n    ▼\n  Output x" },
            { type: "code", language: "python", code: "# Pre-Norm Transformer Block forward logic:\n# def forward(self, x):\n#     x = x + self.dropout(self.attn(self.ln1(x)))\n#     x = x + self.dropout(self.ffn(self.ln2(x)))\n#     return x" },
            { type: "callout", variant: "warning", text: "The residual addition adds to the original x, not the LayerNorm-transformed x. The key insight of Pre-Norm: LN only affects the sublayer input, not the residual path." },
            { type: "callout", variant: "quote", text: "Think About It: Our Transformer Block uses Pre-Norm. But if you look closely at the final output head, there's an extra LayerNorm before it — why do we need one more? What happens if you remove it? (Hint: think about how values in the residual stream grow with depth.)" },
          ],
        },
      ],
      exercises: [
        { id: "layernorm", title: "TODO 1: LayerNorm", description: "Implement Layer Normalization from scratch. Create learnable gamma (ones) and beta (zeros) in __init__. In forward, compute mean and variance, normalize, then apply the affine transform.", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["gamma: nn.Parameter(torch.ones(d_model))", "Compute mean and var on dim=-1 with keepdim=True", "Use unbiased=False for variance"], pseudocode: "__init__:\n  self.gamma = nn.Parameter(ones(d_model))\n  self.beta = nn.Parameter(zeros(d_model))\n\nforward(x):\n  mean = x.mean(dim=-1, keepdim=True)\n  var = x.var(dim=-1, keepdim=True, unbiased=False)\n  x_norm = (x - mean) / sqrt(var + eps)\n  return gamma * x_norm + beta" },
        { id: "gelu", title: "TODO 2: GELU Activation", description: "Implement the GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["math.sqrt(2.0 / math.pi) ≈ 0.7978845608", "Can be written in a single line"] },
        { id: "feedforward", title: "TODO 3: FeedForward Network", description: "Build a two-layer MLP: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model) → Dropout. Default d_ff = 4 * d_model.", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["If d_ff is None, set to 4 * d_model", "Can use nn.Sequential to chain all layers"] },
        { id: "transformer_block", title: "TODO 4: TransformerBlock", description: "Assemble the Pre-Norm Transformer Block: two sublayers each with LayerNorm and residual connection.", labFile: "labs/phase3_transformer/phase_3/transformer.py", hints: ["Need two LayerNorm instances (ln1, ln2)", "Residual: add to the original x, not the LN-transformed x"] },
      ],
      acceptanceCriteria: [
        "LayerNorm output has mean ≈ 0, variance ≈ 1 on the feature dimension",
        "GELU(0) = 0; GELU(x) > 0 for large positive x",
        "FeedForward input and output shapes are identical (batch, seq_len, d_model)",
        "TransformerBlock output shape equals input shape",
      ],
      references: [
        { title: "Layer Normalization", description: "Ba et al. 2016 — The original LayerNorm paper", url: "https://arxiv.org/abs/1607.06450" },
        { title: "Gaussian Error Linear Units (GELUs)", description: "Hendrycks & Gimpel 2016 — The original GELU paper", url: "https://arxiv.org/abs/1606.08415" },
        { title: "On Layer Normalization in the Transformer Architecture", description: "Xiong et al. 2020 — Deep comparison of Pre-Norm and Post-Norm", url: "https://arxiv.org/abs/2002.04745" },
      ],
    },
    {
      phaseId: 3, lessonId: 2,
      title: "GPT Model Assembly",
      subtitle: "Assembling All Components into a Complete GPT Language Model",
      type: "concept",
      duration: "45 min",
      objectives: [
        "Define a GPTConfig dataclass to manage all model hyperparameters",
        "Assemble token embedding + position embedding + N Transformer Blocks + lm_head",
        "Understand and implement weight tying",
        "Implement a complete forward pass with optional loss computation",
      ],
      sections: [
        {
          title: "GPTConfig Design",
          blocks: [
            { type: "paragraph", text: "All the pieces from the first three lessons are ready: the Tokenizer turns text into numbers, Embeddings turn numbers into vectors, Attention lets vectors talk to each other, and FFN lets each vector think independently. Now it's time to plug them all into the motherboard and boot up a complete machine." },
            { type: "code", language: "python", code: "@dataclass\nclass GPTConfig:\n    vocab_size: int = 50257    # GPT-2 BPE vocabulary size\n    block_size: int = 1024     # Maximum sequence length\n    d_model:    int = 768      # Embedding dimension\n    n_heads:    int = 12       # Number of attention heads\n    n_layers:   int = 12       # Number of Transformer blocks\n    dropout:  float = 0.1      # Dropout probability" },
            { type: "table", headers: ["Model", "d_model", "n_heads", "n_layers", "Params"], rows: [
              ["GPT-2 Small", "768", "12", "12", "124M"],
              ["GPT-2 Medium", "1024", "16", "24", "355M"],
              ["GPT-2 Large", "1280", "20", "36", "774M"],
              ["GPT-2 XL", "1600", "25", "48", "1558M"],
            ]},
          ],
        },
        {
          title: "Full GPT Architecture",
          blocks: [
            { type: "diagram", content: "Full GPT Architecture:\n\n  Token IDs → Token Embedding + Position Embedding\n      │\n      ▼ Dropout\n  ┌──────────────────────────────┐\n  │  Transformer Block × N       │\n  │  (LN → Attention + LN → FFN)│\n  └──────────────────────────────┘\n      │\n      ▼ Final LayerNorm\n      ▼ lm_head: Linear(d_model → vocab)\n  logits: (B, T, vocab_size)" },
            { type: "callout", variant: "quote", text: "Instructor's Note: GPT's architecture is surprisingly simple — just stack the components from phases 1–3. An important engineering principle: don't use a complex structure when a simple one works. GPT's success proved that what truly matters isn't a fancy architecture, but sufficient scale and data." },
          ],
        },
        {
          title: "Weight Tying",
          blocks: [
            { type: "paragraph", text: "Weight tying is an important trick in GPT: the token embedding matrix and the lm_head's linear projection share the same weights." },
            { type: "code", language: "python", code: "# Weight tying:\n# self.lm_head.weight = self.token_embedding.weight\n#\n# Effect: saves vocab_size * d_model parameters\n# (GPT-2: 50257 * 768 ≈ 38.6M parameters saved)" },
            { type: "callout", variant: "tip", text: "Instructor's Note: The embedding layer learns \"token → semantic vector\" mapping, while the output head does the reverse: \"semantic vector → token probability.\" These two operations are fundamentally two sides of the same coin, so sharing weights not only saves parameters but also improves performance." },
          ],
        },
        {
          title: "Forward Pass and Loss Computation",
          blocks: [
            { type: "code", language: "python", code: "def forward(self, idx, targets=None):\n    B, T = idx.shape\n    tok_emb = self.token_embedding(idx)                     # (B, T, d_model)\n    pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))  # (T, d_model)\n    x = self.dropout(tok_emb + pos_emb)\n    for block in self.blocks:\n        x = block(x)\n    x = self.ln_f(x)\n    logits = self.lm_head(x)  # (B, T, vocab_size)\n    loss = None\n    if targets is not None:\n        loss = F.cross_entropy(\n            logits.view(-1, logits.size(-1)),\n            targets.view(-1)\n        )\n    return logits, loss" },
            { type: "callout", variant: "quote", text: "Think About It: GPT-2 Small has 124M parameters. Doubling the layers from 12 to 24 roughly doubles the parameter count. But if we double d_model from 768 to 1536 (everything else fixed), how much does the parameter count grow? Which scaling strategy is more efficient?" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_config", title: "TODO 1: GPTConfig", description: "Define a GPTConfig dataclass with six fields (vocab_size, block_size, d_model, n_heads, n_layers, dropout) and their default values.", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["Use the @dataclass decorator", "GPT-2 defaults: vocab_size=50257, d_model=768, n_heads=12, n_layers=12"] },
        { id: "gpt_init", title: "TODO 2: GPT.__init__", description: "Initialize all GPT model components: token/position embeddings, dropout, N TransformerBlocks, final LayerNorm, lm_head — and implement weight tying.", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["nn.ModuleList so PyTorch can track all submodule parameters", "Weight tying: self.lm_head.weight = self.token_embedding.weight", "lm_head bias=False"] },
        { id: "gpt_forward", title: "TODO 3: GPT.forward", description: "Implement the forward pass: embeddings → dropout → TransformerBlock × N → Final LN → lm_head → logits. Compute cross-entropy loss if targets are provided.", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["assert T <= self.config.block_size", "Loss: cross_entropy(logits.view(-1, vocab_size), targets.view(-1))"] },
        { id: "count_parameters", title: "TODO 4: count_parameters", description: "Count the total number of trainable parameters in the model.", labFile: "labs/phase3_transformer/phase_3/gpt_model.py", hints: ["p.numel() returns the number of elements in a tensor", "Only count parameters with requires_grad=True"], pseudocode: "return sum(p.numel() for p in self.parameters() if p.requires_grad)" },
      ],
      acceptanceCriteria: [
        "GPT forward output logits shape is (B, T, vocab_size)",
        "loss is None when targets not provided",
        "loss is a positive scalar when targets are provided",
        "count_parameters returns a positive integer",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "Radford et al. 2019 — GPT-2 paper with complete architecture description", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
        { title: "nanoGPT", description: "Andrej Karpathy's minimal GPT implementation", url: "https://github.com/karpathy/nanoGPT" },
      ],
    },
  ],
};

const phase4ContentEn: PhaseContent = {
  phaseId: 4,
  color: "#F59E0B",
  accent: "#FBBF24",
  lessons: [
    {
      phaseId: 4, lessonId: 1,
      title: "Training Loop",
      subtitle: "From Loss Function to Complete Training Loop — Making GPT Actually Learn",
      type: "concept",
      duration: "75 min",
      objectives: [
        "Understand what cross-entropy loss means for a language model",
        "Implement a cosine learning rate schedule with linear warmup",
        "Implement loss estimation and checkpoint save/load",
        "Build a complete training loop integrating AdamW, gradient clipping, and evaluation",
      ],
      sections: [
        {
          title: "Cross-Entropy Loss",
          blocks: [
            { type: "paragraph", text: "The model is assembled, but right now it's useless — all parameters are randomly initialized. Pre-training is like sending this newborn to read the entire internet, learning the patterns of language from scratch. Today we build the complete training pipeline." },
            { type: "paragraph", text: "The language model training objective is to maximize the predicted probability of the correct next token. Cross-entropy loss measures the difference between the model's output distribution and the true one-hot distribution. For a randomly initialized model, initial loss ≈ ln(vocab_size)." },
            { type: "callout", variant: "info", text: "Perplexity = exp(loss) is a common evaluation metric for language models. Intuitively: perplexity tells you how many options the model is \"hesitating between\" at each position." },
          ],
        },
        {
          title: "AdamW Optimizer",
          blocks: [
            { type: "table", headers: ["Property", "SGD", "Adam", "AdamW"], rows: [
              ["Adaptive learning rate", "No", "Yes", "Yes"],
              ["Weight Decay", "Equivalent to L2", "Coupled with adaptive LR", "Decoupled"],
              ["Transformer suitability", "Poor", "Good", "Best"],
            ]},
            { type: "code", language: "python", code: "optimizer = torch.optim.AdamW(\n    model.parameters(),\n    lr=3e-4,\n    betas=(0.9, 0.999),\n    eps=1e-8,\n    weight_decay=0.01,\n)" },
            { type: "callout", variant: "tip", text: "Instructor's Note: The \"W\" in AdamW (decoupled weight decay) is a commonly overlooked but important detail. In regular Adam, L2 regularization gets scaled by the adaptive learning rate — so parameters with small gradients receive disproportionately large weight decay. AdamW fixes this by applying decay directly to the weights after the gradient step." },
          ],
        },
        {
          title: "Cosine Learning Rate Schedule",
          blocks: [
            { type: "code", language: "python", code: "def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):\n    # Phase 1: Linear Warmup\n    #   lr = max_lr * (step / warmup_steps)\n    # Phase 2: Cosine Decay\n    #   progress = (step - warmup_steps) / (max_steps - warmup_steps)\n    #   lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))\n    # Phase 3: Beyond max_steps → lr = min_lr" },
            { type: "diagram", content: "Cosine LR Schedule with Warmup:\n\n  lr ▲\nmax ─│        ╭──╮\n     │       ╱    ╲\n     │      ╱      ╲\n     │     ╱        ╲\nmin ─│╱                ╲──────────\n     └────┬──────────────┬──────────► step\n       warmup         max_steps" },
          ],
        },
        {
          title: "Loss Estimation and Checkpointing",
          blocks: [
            { type: "code", language: "python", code: "@torch.no_grad()\ndef estimate_loss(model, dataloader, eval_steps, device='cpu'):\n    model.eval()  # disable dropout\n    losses = []\n    for i, (x, y) in enumerate(dataloader):\n        if i >= eval_steps: break\n        x, y = x.to(device), y.to(device)\n        _, loss = model(x, y)\n        losses.append(loss.item())\n    model.train()  # restore training mode\n    return sum(losses) / len(losses)" },
            { type: "code", language: "python", code: "def save_checkpoint(model, optimizer, step, loss, path):\n    Path(path).parent.mkdir(parents=True, exist_ok=True)\n    torch.save({\n        'model_state_dict': model.state_dict(),\n        'optimizer_state_dict': optimizer.state_dict(),\n        'step': step, 'loss': loss,\n    }, path)" },
          ],
        },
        {
          title: "The Complete Training Loop",
          blocks: [
            { type: "code", language: "python", code: "@dataclass\nclass TrainConfig:\n    max_steps: int = 1000\n    batch_size: int = 4\n    learning_rate: float = 3e-4\n    min_lr: float = 3e-5\n    warmup_steps: int = 100\n    eval_interval: int = 100\n    eval_steps: int = 10\n    checkpoint_dir: str = 'checkpoints'\n    checkpoint_interval: int = 500\n    max_grad_norm: float = 1.0\n    device: str = 'cpu'" },
            { type: "callout", variant: "warning", text: "Most common training loop bugs: forgetting zero_grad() (gradient accumulation), forgetting to switch back to train mode after eval, incorrect step counting in the LR schedule." },
            { type: "callout", variant: "quote", text: "Think About It: During the early stages of training, will tokens at the beginning of a sequence have higher or lower loss than tokens at the end? Why? (Hint: think about the causal mask effect — position 0 can only see itself, while position T-1 can see all prior tokens.)" },
          ],
        },
      ],
      exercises: [
        { id: "get_lr", title: "TODO 1: get_lr (cosine schedule)", description: "Implement the cosine learning rate schedule with linear warmup. Three phases: linear warmup → cosine decay → constant min_lr.", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["Return min_lr immediately when step >= max_steps", "Warmup phase: lr = max_lr * (step / warmup_steps)", "Cosine formula: min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))"] },
        { id: "estimate_loss", title: "TODO 2: estimate_loss", description: "Compute average loss in eval mode. Set model.eval(), iterate over at most eval_steps batches, compute average loss, restore model.train().", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["@torch.no_grad() decorator", "Remember to restore model.train() at the end", "loss.item() converts tensor to Python float"] },
        { id: "checkpoints", title: "TODO 3: save_checkpoint / load_checkpoint", description: "Implement checkpoint save and load.", labFile: "labs/phase4_pretraining/phase_4/utils.py", hints: ["save: Path(path).parent.mkdir(parents=True, exist_ok=True)", "load: torch.load(path, map_location='cpu', weights_only=True)"] },
        { id: "train_loop", title: "TODO 4: TrainConfig + train()", description: "Define the TrainConfig dataclass and complete train() function.", labFile: "labs/phase4_pretraining/phase_4/train.py", hints: ["Use itertools.cycle(train_loader) to loop through the DataLoader", "LR update: optimizer.param_groups[0]['lr'] = get_lr(...)", "Order: get_lr → forward → backward → clip_grad_norm_ → step → zero_grad"] },
      ],
      acceptanceCriteria: [
        "get_lr returns 0 at step=0 and max_lr at end of warmup",
        "estimate_loss restores model to train mode after evaluation",
        "checkpoint roundtrip results in identical model weights",
        "train() results in measurably changed model weights",
      ],
      references: [
        { title: "Decoupled Weight Decay Regularization", description: "Loshchilov & Hutter 2019 — The AdamW paper", url: "https://arxiv.org/abs/1711.05101" },
        { title: "Chinchilla: Training Compute-Optimal Large Language Models", description: "Hoffmann et al. 2022 — Optimal training tokens to model size ratio", url: "https://arxiv.org/abs/2203.15556" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// Locale dispatch
// ═══════════════════════════════════════════════════════════════════

const phase3Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase3ContentZhTW,
  "zh-CN": phase3ContentZhCN,
  "en": phase3ContentEn,
};

const phase4Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase4ContentZhTW,
  "zh-CN": phase4ContentZhCN,
  "en": phase4ContentEn,
};

export function getPhase3Content(locale: Locale): PhaseContent { return phase3Map[locale]; }
export function getPhase4Content(locale: Locale): PhaseContent { return phase4Map[locale]; }
