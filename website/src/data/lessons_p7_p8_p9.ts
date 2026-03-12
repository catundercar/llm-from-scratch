import type { PhaseContent } from "./types";

// ─────────────────────────────────────────────────────────────
// Phase 7: LoRA (Low-Rank Adaptation)
// ─────────────────────────────────────────────────────────────

export const phase7Content: PhaseContent = {
  phaseId: 7,
  color: "#F97316",
  accent: "#FB923C",
  lessons: [
    {
      phaseId: 7, lessonId: 1,
      title: "LoRA: Low-Rank Adaptation",
      subtitle: "用極少參數微調大型模型——低秩分解的力量",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 full fine-tuning 的記憶體與計算成本，並分析參數量",
        "掌握低秩分解（low-rank decomposition）的數學直覺",
        "實作 LoRALinear 模組，包含 frozen weight 與 trainable low-rank matrices",
        "將 LoRA 注入現有模型，並學會合併權重以供推論部署",
        "分析 LoRA 的參數效率，比較 Full FT、LoRA 與 QLoRA",
      ],
      sections: [
        {
          title: "Why Full Fine-Tuning is Expensive",
          blocks: [
            { type: "paragraph", text: "假設你有一個 7B 參數的模型，想讓它學會寫法律文書。Full fine-tuning 意味著更新全部 70 億個參數——你需要一張 80GB 的 A100 GPU，訓練好幾天。但如果我告訴你，只更新其中 0.1% 的參數就能達到 95% 的效果呢？這就是 LoRA 的魔法。今天我們來揭開它的祕密——為什麼這麼少的參數就夠了？" },
            { type: "callout", variant: "quote", text: "💡 講師心得：LoRA 是過去幾年最具影響力的效率化技術之一。它的核心洞察驚人地簡單：模型微調時的權重變化（ΔW）其實是低秩的——也就是說，你不需要更新所有參數，只需要在一個低維子空間中調整就夠了。這就像你搬進新房子，不需要重新裝修整棟樓，只需要改幾面牆的顏色。" },
            { type: "paragraph", text: "當我們拿到一個預訓練好的 LLM（例如 LLaMA-7B），想讓它學會新任務時，最直覺的方法是 full fine-tuning：解凍所有參數，用任務資料繼續訓練。但這代價極高。" },
            { type: "heading", level: 3, text: "參數量分析" },
            { type: "paragraph", text: "考慮一個 7B 參數的模型。Full fine-tuning 需要儲存：(1) 模型參數本身（FP16 約 14 GB），(2) 梯度（同樣 14 GB），(3) Adam 優化器狀態（每個參數需要 first moment + second moment，共 28 GB）。光是這三項就需要約 56 GB 的 GPU 記憶體，還不包括 activation memory。" },
            { type: "table", headers: ["項目", "FP16 大小 (7B model)", "說明"], rows: [
              ["模型參數", "14 GB", "7B × 2 bytes"],
              ["梯度", "14 GB", "每個參數一個梯度值"],
              ["Adam m (1st moment)", "14 GB", "動量的指數移動平均"],
              ["Adam v (2nd moment)", "14 GB", "梯度平方的指數移動平均"],
              ["合計", "~56 GB", "還需加上 activations"],
            ]},
            { type: "callout", variant: "warning", text: "這意味著 full fine-tuning 一個 7B 模型至少需要一張 80 GB 的 A100 GPU。對大多數研究者和開發者而言，這是不切實際的。我們需要更聰明的方法。" },
            { type: "paragraph", text: "核心觀察：fine-tuning 時，模型權重的變化量 ΔW 通常是低秩的（low-rank）。Aghajanyan et al. (2021) 的研究發現，預訓練模型具有很低的 intrinsic dimensionality——即使將更新限制在一個很小的子空間中，效果仍然接近 full fine-tuning。這就是 LoRA 的理論基礎。" },
          ],
        },
        {
          title: "Low-Rank Decomposition 直覺",
          blocks: [
            { type: "paragraph", text: "低秩分解的核心思想：一個大矩陣可以被近似為兩個小矩陣的乘積。如果原始權重矩陣 W 是 d×d 維（例如 4096×4096），完整的更新 ΔW 也是 d×d 維，有 d² 個參數。但如果 ΔW 的秩（rank）只有 r（遠小於 d），我們就可以把它分解為 B×A，其中 B 是 d×r、A 是 r×d，參數量從 d² 降為 2dr。" },
            { type: "diagram", content: "Low-Rank Decomposition 視覺化：\n\n原始更新 ΔW:  d×d = d² 參數\n┌─────────────────────┐\n│                     │\n│     d × d 矩陣       │  例: 4096 × 4096 = 16,777,216 參數\n│   (full rank)       │\n│                     │\n└─────────────────────┘\n\n低秩近似 B × A:  2 × d × r 參數\n┌───┐   ┌─────────────────────┐\n│   │   │                     │\n│ B │ × │         A           │  例: r = 8\n│d×r│   │       r × d         │  4096×8 + 8×4096 = 65,536 參數\n│   │   │                     │  壓縮比: 256×\n└───┘   └─────────────────────┘" },
            { type: "callout", variant: "info", text: "rank r 是 LoRA 的核心超參數。r 越大，表達能力越強但參數越多。實務上 r = 4~16 就能達到接近 full fine-tuning 的效果。" },
          ],
        },
        {
          title: "LoRA 公式與 LoRALinear 實作",
          blocks: [
            { type: "paragraph", text: "LoRA 的正式公式非常優雅：" },
            { type: "diagram", content: "W' = W + (α/r) · B @ A\n\n其中：\n  W:  原始預訓練權重 (frozen, requires_grad=False)\n  A:  低秩矩陣, shape (in_features, rank), Kaiming uniform 初始化\n  B:  低秩矩陣, shape (rank, out_features), 初始化為零\n  α:  scaling factor (超參數)\n  r:  rank (超參數)\n\n關鍵設計：B 初始化為零 → 訓練開始時 BA = 0 → 模型行為完全不變" },
            { type: "callout", variant: "tip", text: "💡 講師心得：α/r 這個 scaling factor 容易被忽略但非常重要。α 控制 LoRA 的「學習率倍數」——α 越大，LoRA 的影響越大。實務中通常設 α = r 或 α = 2r。一個常見的錯誤是只調 r 而忘了 α，導致 r 增大時效果反而變差（因為 α/r 變小了，LoRA 的影響被稀釋）。" },
            { type: "heading", level: 3, text: "LoRA Layer 架構圖" },
            { type: "diagram", content: "                    LoRALinear 層\n    ┌──────────────────────────────────────┐\n    │                                      │\n    │  ┌──────────────────────┐             │\n    │  │   W (frozen)         │  ──┐        │\n    │  │  (out × in)          │    │        │\n    │  └──────────────────────┘    │        │\nx ──┤                              ├── + ── ├──→ output\n    │  ┌─────┐    ┌──────────┐    │        │\n    │  │  A  │ →  │    B     │ ───┘        │\n    │  │in×r │    │  r×out   │  × (α/r)    │\n    │  └─────┘    └──────────┘             │\n    │   trainable   trainable              │\n    └──────────────────────────────────────┘\n\n前向傳播：\n  base_output = x @ W^T + bias          (frozen path)\n  lora_output = x @ A @ B^T × (α/r)    (trainable path)\n  output = base_output + lora_output" },
            { type: "heading", level: 3, text: "__init__ 實作要點" },
            { type: "code", language: "python", code: "class LoRALinear(nn.Module):\n    def __init__(self, in_features, out_features, rank=4, alpha=1.0, bias=True):\n        super().__init__()\n        # Step 1: Frozen pretrained weight\n        self.weight = nn.Parameter(\n            torch.empty(out_features, in_features), requires_grad=False\n        )\n        if bias:\n            self.bias = nn.Parameter(\n                torch.zeros(out_features), requires_grad=False\n            )\n        else:\n            self.bias = None\n\n        # Step 2: Trainable low-rank matrices\n        self.lora_A = nn.Parameter(torch.empty(in_features, rank))\n        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n\n        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))\n        # B = 0 → 初始時 LoRA 貢獻為零\n\n        # Step 3: Scaling factor\n        self.scaling = alpha / rank" },
            { type: "heading", level: 3, text: "forward 實作" },
            { type: "code", language: "python", code: "def forward(self, x):\n    # Frozen path: standard linear\n    base_out = F.linear(x, self.weight, self.bias)\n    # LoRA path: low-rank bypass\n    lora_out = (x @ self.lora_A @ self.lora_B.T) * self.scaling\n    return base_out + lora_out" },
            { type: "callout", variant: "tip", text: "注意 F.linear(x, W, bias) 計算的是 x @ W^T + bias，其中 W 的 shape 是 (out, in)，這是 PyTorch 的慣例。而 LoRA path 中 A 是 (in, r)、B 是 (r, out)，所以 x @ A @ B^T 的結果 shape 正確。" },
          ],
        },
        {
          title: "Applying LoRA to Model & Merging Weights",
          blocks: [
            { type: "paragraph", text: "有了 LoRALinear 後，我們需要把它注入到現有的 Transformer 模型中。典型的做法是替換 attention 中的 Q、V 投影層（q_proj、v_proj），因為這些層對任務適應最為關鍵。" },
            { type: "heading", level: 3, text: "apply_lora_to_model" },
            { type: "code", language: "python", code: "def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):\n    if target_modules is None:\n        target_modules = ['q_proj', 'v_proj']\n\n    # Step 1: Freeze ALL existing parameters\n    for param in model.parameters():\n        param.requires_grad = False\n\n    # Step 2: Collect modules to replace\n    replacements = []\n    for name, module in model.named_modules():\n        if isinstance(module, nn.Linear):\n            if any(name.endswith(t) for t in target_modules):\n                replacements.append((name, module))\n\n    # Step 3: Replace with LoRALinear\n    for name, module in replacements:\n        parent_name = '.'.join(name.split('.')[:-1])\n        attr_name = name.split('.')[-1]\n        parent = dict(model.named_modules())[parent_name]\n        setattr(parent, attr_name, LoRALinear.from_linear(module, rank, alpha))\n\n    return model" },
            { type: "heading", level: 3, text: "合併 LoRA 權重" },
            { type: "paragraph", text: "訓練完成後，我們可以把 LoRA 的低秩更新合併回原始權重。合併後的模型與原始模型架構完全相同（普通的 nn.Linear），推論時沒有額外開銷。" },
            { type: "code", language: "python", code: "def merge_lora_weights(model):\n    for name, module in model.named_modules():\n        if isinstance(module, LoRALinear):\n            # W_merged = W + scaling * (B^T @ A^T)\n            # W: (out, in), B^T: (out, r), A^T: (r, in)\n            merged = module.weight + module.scaling * (\n                module.lora_B.T @ module.lora_A.T\n            )\n            new_linear = nn.Linear(\n                module.weight.shape[1], module.weight.shape[0],\n                bias=module.bias is not None\n            )\n            new_linear.weight.data = merged\n            if module.bias is not None:\n                new_linear.bias.data = module.bias.data\n            # Replace in parent\n            # ... (same setattr pattern)\n    return model" },
            { type: "callout", variant: "info", text: "合併的數學：W' = W + (α/r) × B^T × A^T。注意轉置的方向——因為 W 儲存為 (out_features, in_features)，所以 B^T 是 (out, r)，A^T 是 (r, in)，乘積為 (out, in)，shape 一致。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：LoRA 的另一個巧妙之處在於推論時零開銷。訓練時 BA 是單獨的矩陣，但部署前你可以把它合併回 W' = W + (α/r)BA。合併後的模型和原始模型結構完全一樣，推論速度不受影響。這意味著你可以為不同任務訓練多個 LoRA adapter，部署時選擇性載入——一個底座模型，多個任務能力。" },
          ],
        },
        {
          title: "Parameter Efficiency Analysis",
          blocks: [
            { type: "paragraph", text: "LoRA 的參數效率令人驚嘆。讓我們以具體數字來看：" },
            { type: "table", headers: ["方法", "可訓練參數", "記憶體需求", "效果", "推論開銷"], rows: [
              ["Full Fine-Tuning", "100%", "~4× 模型大小", "最佳（但可能 overfit）", "無"],
              ["LoRA (r=8)", "~0.1-1%", "模型大小 + 少量", "接近 Full FT", "合併後無"],
              ["QLoRA", "~0.1-1%", "~1/4 模型大小 + 少量", "接近 LoRA", "需要 dequant"],
            ]},
            { type: "heading", level: 3, text: "計算範例" },
            { type: "paragraph", text: "對一個 d_model = 4096 的 Transformer 層，每個 attention 有 q_proj 和 v_proj 兩個 Linear(4096, 4096)。原始參數量：2 × 4096² = 33,554,432。使用 rank=8 的 LoRA：2 × (4096×8 + 8×4096) = 131,072。壓縮比約 256 倍。" },
            { type: "code", language: "python", code: "def count_trainable_params(model):\n    total = sum(p.numel() for p in model.parameters())\n    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    print(f'Total params:     {total:,}')\n    print(f'Trainable params: {trainable:,} ({100*trainable/total:.2f}%)')\n    print(f'Frozen params:    {total - trainable:,}')\n    return trainable, total" },
            { type: "callout", variant: "tip", text: "QLoRA 更進一步：它將 frozen weights 量化為 4-bit（NF4 格式），大幅降低記憶體需求。搭配分頁優化器，甚至可以在單張 24 GB 的消費級 GPU 上微調 65B 模型。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：LoRA 只對 Linear 層（通常是 attention 的 W_q, W_k, W_v, W_o）注入低秩矩陣。但 FFN 層佔了模型一半以上的參數——為什麼通常不對 FFN 做 LoRA？如果對 FFN 也做 LoRA，效果會更好嗎？這跟 FFN 和 Attention 各自學到的知識類型有什麼關係？" },
          ],
        },
      ],
      exercises: [
        {
          id: "lora_linear_init", title: "TODO 1: LoRALinear.__init__",
          description: "初始化 LoRALinear 模組：建立 frozen weight、trainable lora_A（Kaiming uniform）、trainable lora_B（zero init），並計算 scaling = alpha / rank。",
          labFile: "labs/phase7_lora/phase_7/lora.py",
          hints: [
            "使用 nn.Parameter(..., requires_grad=False) 建立 frozen weight",
            "lora_A shape: (in_features, rank)，用 nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))",
            "lora_B shape: (rank, out_features)，初始化為 torch.zeros",
            "B 初始化為零確保訓練開始時 LoRA 貢獻為零",
          ],
          pseudocode: "self.weight = Parameter(empty(out, in), requires_grad=False)\nself.bias = Parameter(zeros(out), requires_grad=False) if bias else None\nself.lora_A = Parameter(empty(in, rank))\nkaiming_uniform_(self.lora_A, a=sqrt(5))\nself.lora_B = Parameter(zeros(rank, out))\nself.scaling = alpha / rank",
        },
        {
          id: "lora_linear_forward", title: "TODO 2: LoRALinear.forward",
          description: "實作 LoRA 的前向傳播：frozen path (F.linear) + trainable low-rank bypass (x @ A @ B^T * scaling)。",
          labFile: "labs/phase7_lora/phase_7/lora.py",
          hints: [
            "base output: F.linear(x, self.weight, self.bias)",
            "LoRA path: (x @ self.lora_A @ self.lora_B.T) * self.scaling",
            "兩者相加即為最終輸出",
          ],
          pseudocode: "base = F.linear(x, self.weight, self.bias)\nlora = (x @ self.lora_A @ self.lora_B.T) * self.scaling\nreturn base + lora",
        },
        {
          id: "apply_lora_to_model", title: "TODO 3: apply_lora_to_model",
          description: "將模型中指定的 nn.Linear 層替換為 LoRALinear。先凍結所有參數，再替換目標層。",
          labFile: "labs/phase7_lora/phase_7/lora.py",
          hints: [
            "先用 for param in model.parameters(): param.requires_grad = False 凍結所有參數",
            "用 model.named_modules() 遍歷，找到 name 以 target_modules 中字串結尾的 nn.Linear",
            "用 LoRALinear.from_linear() 建立替換模組",
            "用 setattr(parent, attr_name, new_module) 執行替換",
          ],
          pseudocode: "freeze all params\nfor name, module in model.named_modules():\n  if isinstance(module, Linear) and name.endswith(target):\n    collect (name, module)\nfor name, module in collected:\n  parent = get_parent_module(name)\n  setattr(parent, attr, LoRALinear.from_linear(module, rank, alpha))\nreturn model",
        },
        {
          id: "merge_lora_weights", title: "TODO 4: merge_lora_weights",
          description: "將 LoRA 權重合併回原始權重矩陣，產生標準 nn.Linear 供高效推論。",
          labFile: "labs/phase7_lora/phase_7/lora.py",
          hints: [
            "merged_weight = lora.weight + lora.scaling * (lora.lora_B.T @ lora.lora_A.T)",
            "建立新的 nn.Linear 並複製合併後的 weight 和 bias",
            "使用與 apply_lora_to_model 相同的 parent setattr 模式",
          ],
          pseudocode: "for name, module in model.named_modules():\n  if isinstance(module, LoRALinear):\n    W_merged = module.weight + scaling * (B^T @ A^T)\n    new_linear = nn.Linear(in, out, bias=...)\n    new_linear.weight.data = W_merged\n    replace in parent\nreturn model",
        },
        {
          id: "count_trainable_params", title: "TODO 5: count_trainable_params",
          description: "統計模型的可訓練參數量與總參數量，並印出摘要。",
          labFile: "labs/phase7_lora/phase_7/lora.py",
          hints: [
            "total = sum(p.numel() for p in model.parameters())",
            "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)",
            "回傳 (trainable, total)",
          ],
          pseudocode: "total = sum(p.numel() for all params)\ntrainable = sum(p.numel() for params with requires_grad)\nprint summary\nreturn (trainable, total)",
        },
      ],
      acceptanceCriteria: [
        "LoRALinear 初始化後，B 為零矩陣，forward 輸出與原始 nn.Linear 一致",
        "apply_lora_to_model 後，只有 lora_A 和 lora_B 的 requires_grad 為 True",
        "merge_lora_weights 後，模型不含任何 LoRALinear，且輸出與合併前一致",
        "count_trainable_params 正確報告可訓練與凍結參數數量",
        "LoRA rank 和 alpha 可自訂設定",
      ],
      references: [
        { title: "LoRA: Low-Rank Adaptation of Large Language Models", description: "Hu et al. 2021 — LoRA 的原始論文，提出低秩適應方法", url: "https://arxiv.org/abs/2106.09685" },
        { title: "QLoRA: Efficient Finetuning of Quantized LLMs", description: "Dettmers et al. 2023 — 結合 4-bit 量化與 LoRA 的方法", url: "https://arxiv.org/abs/2305.14314" },
        { title: "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning", description: "Aghajanyan et al. 2021 — 預訓練模型低 intrinsic dimension 的理論基礎", url: "https://arxiv.org/abs/2012.13255" },
        { title: "HuggingFace PEFT Library", description: "Parameter-Efficient Fine-Tuning 的官方實作庫，包含 LoRA、QLoRA 等方法", url: "https://github.com/huggingface/peft" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 8: Human Alignment (SFT & DPO)
// ─────────────────────────────────────────────────────────────

export const phase8Content: PhaseContent = {
  phaseId: 8,
  color: "#EF4444",
  accent: "#F87171",
  lessons: [
    {
      phaseId: 8, lessonId: 1,
      title: "SFT & DPO",
      subtitle: "從語言模型到 AI 助理——指令微調與人類偏好對齊",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 Supervised Fine-Tuning (SFT) 的指令格式與 label masking 技術",
        "實作 InstructionDataset，正確處理 Alpaca 模板與 prompt/response 分割",
        "理解 RLHF 與 DPO 的差異，以及 DPO 如何避免訓練 reward model",
        "掌握 DPO loss 的數學推導與實作",
        "實作 PreferenceDataset 與 dpo_loss 函數",
      ],
      sections: [
        {
          title: "From Language Model to Assistant: Why SFT?",
          blocks: [
            { type: "paragraph", text: "你有沒有用過那種「野生」的 base model（比如直接下載 LLaMA-2 base）？你會發現它很奇怪——你問它「台北的天氣如何？」，它不會回答你，而是接著寫「台南的天氣如何？高雄的天氣如何？」——它以為你在寫一個城市列表！這就是 base model 的問題：它學會了語言，但不知道自己該扮演什麼角色。SFT 教它「你是一個助手，要回答問題」，DPO 教它「哪種回答是人類喜歡的」。今天我們來搞定這兩步。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：如果說 Pre-training 教會了模型「語言能力」，SFT 和 DPO 就是教會模型「社會規範」。預訓練後的模型像一個博學但沒有教養的人——它知道很多，但不知道怎麼恰當地回答問題。SFT 教它格式（用 instruction-response 的格式回答），DPO 教它品質（哪種回答更好）。這兩步合起來，就是所謂的「人類對齊」（alignment）。" },
            { type: "paragraph", text: "預訓練的語言模型是強大的「文本補全器」——給它一段開頭，它能續寫出流暢的文本。但它不是一個「助理」。如果你問它「什麼是量子力學？」，它可能會續寫出「是很多物理學家研究的課題...」，而不是直接給你一個清晰的解釋。" },
            { type: "paragraph", text: "Supervised Fine-Tuning (SFT) 的目標是教模型理解「指令→回應」的格式。我們用大量的 (instruction, response) 對來訓練模型，讓它學會：看到指令格式時，生成有幫助的回應。" },
            { type: "callout", variant: "info", text: "InstructGPT (Ouyang et al., 2022) 的三階段訓練流程：(1) SFT — 用人工撰寫的示範來微調，(2) Reward Model — 訓練一個偏好模型，(3) PPO — 用 RL 進一步對齊。DPO 將後兩步合併為一步。" },
          ],
        },
        {
          title: "Instruction Format & Label Masking",
          blocks: [
            { type: "paragraph", text: "SFT 使用結構化的 prompt 模板。Alpaca 格式是最常見的之一：" },
            { type: "code", language: "text", code: "### Instruction:\n{instruction}\n\n### Input:\n{input}          ← 如果沒有 input，此段省略\n\n### Response:\n{output}" },
            { type: "heading", level: 3, text: "Label Masking: 只在 Response 上計算 Loss" },
            { type: "paragraph", text: "關鍵技巧：我們不想讓模型「學會生成指令」，只想讓它「學會回應指令」。因此，我們對 prompt 部分的 token 設定 label = -100（PyTorch 的 cross_entropy 會自動忽略 -100），只在 response 部分計算 loss。" },
            { type: "diagram", content: "Label Masking 示意圖：\n\n Token:  [### Inst: 什麼是AI？ ### Response: AI是人工智慧...]\n          ├──── prompt tokens ────┤├── response tokens ──┤\n Labels: [-100 -100 -100 ... -100  AI  是  人工  智慧  ...]\n          ├──── masked (no loss) ─┤├── compute loss ─────┤\n\n只有 response tokens 會貢獻到 cross-entropy loss！\n這確保模型專注學習「如何回應」，而非「如何複述指令」。" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Label masking 是 SFT 中最容易出錯的地方。原理很簡單：我們只想讓模型學會「如何回答」，不想讓它學會「如何提問」。所以 loss 只計算 assistant response 部分，instruction 部分的 loss 被 mask 掉。如果忘了做 label masking，模型會花一半的容量去「背」instruction 的措辭，浪費訓練資源且效果變差。" },
            { type: "heading", level: 3, text: "InstructionDataset 實作" },
            { type: "code", language: "python", code: "class InstructionDataset(Dataset):\n    def __init__(self, jsonl_path, tokenizer, max_length=512):\n        self.samples = []\n        with open(jsonl_path) as f:\n            for line in f:\n                data = json.loads(line)\n                instruction = data['instruction']\n                input_text = data.get('input', '')\n                output_text = data['output']\n\n                # Format prompt (without response)\n                if input_text.strip():\n                    prompt = TEMPLATE_WITH_INPUT.format(\n                        instruction=instruction, input=input_text\n                    )\n                else:\n                    prompt = TEMPLATE_NO_INPUT.format(\n                        instruction=instruction\n                    )\n\n                full_text = prompt + output_text\n\n                # Tokenize\n                prompt_ids = tokenizer.encode(prompt)\n                full_ids = tokenizer.encode(full_text)\n\n                # Create labels: -100 for prompt, real ids for response\n                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]\n\n                # Pad/truncate to max_length\n                # ... (padding with pad_token_id, labels padded with -100)\n                self.samples.append({...})" },
            { type: "callout", variant: "warning", text: "Tokenization 的陷阱：tokenizer.encode(prompt + response) 的結果可能不等於 tokenizer.encode(prompt) + tokenizer.encode(response)，因為 BPE merge 可能跨越邊界。最安全的做法是先 tokenize 整個文本，再用 prompt-only 的 token 長度來決定 mask 邊界。" },
          ],
        },
        {
          title: "SFT Training Loop",
          blocks: [
            { type: "paragraph", text: "SFT 的訓練迴圈與標準語言模型訓練幾乎相同，唯一的差別是 label masking。Loss 函數使用 cross_entropy 並設定 ignore_index=-100。" },
            { type: "diagram", content: "SFT 資料流：\n\n   JSONL 檔案 ──→ InstructionDataset ──→ DataLoader\n   (instruction,    (tokenize, mask       (batch)\n    input, output)   labels, pad)\n                         │\n                         ▼\n   ┌──────────────────────────────────────┐\n   │  Training Loop:                      │\n   │  1. input_ids → model → logits      │\n   │  2. Shift logits & labels by 1      │\n   │  3. CE loss (ignore_index=-100)     │\n   │  4. loss.backward() → optimizer     │\n   └──────────────────────────────────────┘" },
            { type: "code", language: "python", code: "def train_sft(model, train_loader, val_loader, config):\n    model.to(config.device)\n    model.train()\n    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)\n    epoch_losses = []\n\n    for epoch in range(config.n_epochs):\n        total_loss = 0\n        for step, batch in enumerate(train_loader):\n            input_ids = batch['input_ids'].to(config.device)\n            labels = batch['labels'].to(config.device)\n\n            logits = model(input_ids)  # (B, T, vocab_size)\n\n            # Shift for next-token prediction\n            logits = logits[:, :-1, :].contiguous()\n            labels = labels[:, 1:].contiguous()\n\n            loss = F.cross_entropy(\n                logits.view(-1, logits.size(-1)),\n                labels.view(-1),\n                ignore_index=-100\n            )\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()\n            total_loss += loss.item()\n\n        epoch_losses.append(total_loss / len(train_loader))\n    return epoch_losses" },
            { type: "callout", variant: "tip", text: "注意 shift 的方向：logits[:, :-1] 預測 labels[:, 1:]。這是因為位置 i 的 logits 預測的是位置 i+1 的 token。忘記 shift 是最常見的 bug 之一。" },
          ],
        },
        {
          title: "Direct Preference Optimization (DPO)",
          blocks: [
            { type: "paragraph", text: "SFT 教模型「如何回答」，但不教它「哪種回答更好」。人類偏好對齊（alignment）的目標是讓模型的回答更安全、更有幫助、更誠實。傳統做法是 RLHF：先訓練 reward model，再用 PPO 做 reinforcement learning。DPO 提供了一條更簡單的路。" },
            { type: "heading", level: 3, text: "RLHF vs DPO" },
            { type: "table", headers: ["特性", "RLHF (PPO)", "DPO"], rows: [
              ["訓練階段", "SFT → RM → PPO（三階段）", "SFT → DPO（兩階段）"],
              ["需要 Reward Model?", "是，需單獨訓練", "否，隱式學習"],
              ["需要 RL 訓練?", "是 (PPO, 不穩定)", "否，純 supervised loss"],
              ["實作複雜度", "高（PPO + value head + KL penalty）", "低（一個 loss 函數）"],
              ["記憶體需求", "高（policy + RM + value + ref）", "中（policy + ref）"],
              ["訓練穩定性", "低（RL 固有問題）", "高（supervised learning）"],
              ["效果", "優秀", "comparable 甚至更好"],
            ]},
            { type: "heading", level: 3, text: "DPO 的核心洞見" },
            { type: "paragraph", text: "DPO 的核心發現是：在 RLHF 的 Bradley-Terry 偏好模型下，最優的 policy 與 reward function 之間有一個封閉形式的關係。這意味著我們可以直接用偏好數據來訓練 policy，跳過 reward model 的訓練。" },
            { type: "diagram", content: "RLHF Pipeline:\n  ┌────┐   ┌──────────┐   ┌─────┐\n  │SFT │ → │Reward    │ → │PPO  │ → aligned model\n  │    │   │Model     │   │(RL) │\n  └────┘   └──────────┘   └─────┘\n   示範資料  偏好 pair 資料  RL 訓練\n\nDPO Pipeline:\n  ┌────┐   ┌──────────────────┐\n  │SFT │ → │DPO Loss          │ → aligned model\n  │    │   │(直接用偏好資料)    │\n  └────┘   └──────────────────┘\n   示範資料  偏好 pair 資料\n\n更簡單，更穩定，效果 comparable！" },
          ],
        },
        {
          title: "DPO Loss 公式與實作",
          blocks: [
            { type: "paragraph", text: "DPO 的 loss 函數看起來有些複雜，但拆解後每一步都很直觀：" },
            { type: "diagram", content: "DPO Loss:\n\n  L_DPO = -log σ( β × [ log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ] )\n\n展開：\n  log_ratio_chosen  = log π_policy(y_w|x) - log π_ref(y_w|x)\n  log_ratio_rejected = log π_policy(y_l|x) - log π_ref(y_l|x)\n\n  L = -log σ( β × (log_ratio_chosen - log_ratio_rejected) )\n\n其中：\n  y_w: chosen (preferred) response\n  y_l: rejected (dispreferred) response\n  x:   prompt\n  π:   policy model (正在訓練)\n  π_ref: reference model (frozen SFT model)\n  β:   temperature，控制偏離 reference 的程度\n  σ:   sigmoid function" },
            { type: "paragraph", text: "直覺解讀：DPO loss 鼓勵 policy model（相對於 reference model）增加 chosen response 的概率、降低 rejected response 的概率。β 控制偏離 reference 的程度：β 越大，懲罰偏離越強。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：DPO 最大的貢獻是證明了你不需要一個單獨的 reward model。傳統的 RLHF 流程是：收集人類偏好 → 訓練 reward model → 用 PPO 優化 policy。DPO 把這三步壓縮成一步——直接從偏好數據中優化模型。數學上它是等價的（可以證明 DPO 的最優解就是 RLHF 的最優解），但實作上簡單得多、穩定得多。這就是為什麼 DPO 正在取代 RLHF 成為主流。" },
            { type: "heading", level: 3, text: "get_log_probs 與 dpo_loss" },
            { type: "code", language: "python", code: "def get_log_probs(model, input_ids, attention_mask, prompt_lengths):\n    \"\"\"計算 response tokens 的 log probability 總和。\"\"\"\n    logits = model(input_ids)  # (B, T, V)\n    # Shift for next-token prediction\n    logits = logits[:, :-1, :]  # (B, T-1, V)\n    labels = input_ids[:, 1:]   # (B, T-1)\n\n    log_probs = logits.log_softmax(dim=-1)  # (B, T-1, V)\n    token_log_probs = log_probs.gather(\n        dim=-1, index=labels.unsqueeze(-1)\n    ).squeeze(-1)  # (B, T-1)\n\n    # Mask: only response tokens (after prompt_length)\n    # After shifting, response starts at prompt_length - 1\n    mask = torch.arange(labels.size(1), device=labels.device)\n    mask = mask.unsqueeze(0) >= (prompt_lengths.unsqueeze(1) - 1)\n    mask = mask & attention_mask[:, 1:].bool()\n\n    return (token_log_probs * mask).sum(dim=-1)  # (B,)" },
            { type: "code", language: "python", code: "def dpo_loss(policy_chosen_lp, policy_rejected_lp,\n             ref_chosen_lp, ref_rejected_lp, beta=0.1):\n    log_ratio_chosen = policy_chosen_lp - ref_chosen_lp\n    log_ratio_rejected = policy_rejected_lp - ref_rejected_lp\n    loss = -F.logsigmoid(\n        beta * (log_ratio_chosen - log_ratio_rejected)\n    ).mean()\n    return loss" },
            { type: "callout", variant: "tip", text: "使用 F.logsigmoid 而非 log(sigmoid(...)) 可以避免數值不穩定。當 log_ratio 的差值很大（正或負）時，sigmoid 可能飽和到 0 或 1，取 log 會產生 -inf 或 0 的梯度問題。" },
          ],
        },
        {
          title: "DPO Training Pipeline",
          blocks: [
            { type: "paragraph", text: "DPO 的訓練需要兩個模型：policy model（正在訓練）和 reference model（frozen，通常是 SFT 後的 checkpoint）。" },
            { type: "diagram", content: "DPO Training Pipeline:\n\n  偏好資料: (prompt, chosen_response, rejected_response)\n                    │\n                    ▼\n  ┌─────────────────────────────────────────────┐\n  │                                             │\n  │  policy model ──→ log π(y_w|x), log π(y_l|x) │  ← trainable\n  │                                             │\n  │  ref model ────→ log π_ref(y_w|x), log π_ref(y_l|x) │  ← frozen\n  │                                             │\n  │  DPO Loss = -log σ(β × (Δchosen - Δrejected)) │\n  │                                             │\n  │  loss.backward() → update policy only       │\n  └─────────────────────────────────────────────┘" },
            { type: "code", language: "python", code: "def train_dpo(policy_model, ref_model, train_loader, config):\n    # Freeze reference model\n    ref_model.eval()\n    for p in ref_model.parameters():\n        p.requires_grad = False\n\n    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)\n\n    for epoch in range(config.n_epochs):\n        for batch in train_loader:\n            chosen_ids = batch['chosen_ids'].to(config.device)\n            rejected_ids = batch['rejected_ids'].to(config.device)\n            prompt_lengths = batch['prompt_length'].to(config.device)\n            # ... masks\n\n            # Policy log probs\n            pi_chosen = get_log_probs(policy_model, chosen_ids, ...)\n            pi_rejected = get_log_probs(policy_model, rejected_ids, ...)\n\n            # Reference log probs (no gradient!)\n            with torch.no_grad():\n                ref_chosen = get_log_probs(ref_model, chosen_ids, ...)\n                ref_rejected = get_log_probs(ref_model, rejected_ids, ...)\n\n            loss = dpo_loss(pi_chosen, pi_rejected,\n                           ref_chosen, ref_rejected, config.beta)\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()" },
            { type: "callout", variant: "info", text: "β (beta) 的選擇很重要。β 太小，模型會大幅偏離 reference，可能產生不連貫的文本。β 太大，模型幾乎不會從 reference 移動，學習效果差。常見範圍是 0.1-0.5。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：DPO 需要「偏好對」數據——對同一個 prompt，一個好的回答和一個差的回答。但在實務中，收集這種配對數據很昂貴。如果我們只有「好的回答」（沒有對應的「差的回答」），還能用 DPO 嗎？有沒有什麼技巧可以自動生成「差的回答」？（提示：想想 rejection sampling 和 self-play。）" },
          ],
        },
      ],
      exercises: [
        {
          id: "instruction_dataset_init", title: "TODO 1: InstructionDataset.__init__",
          description: "從 JSONL 載入指令資料，用 Alpaca 模板格式化，tokenize 後建立 label-masked 的訓練樣本。",
          labFile: "labs/phase8_instruction_tuning/phase_8/sft.py",
          hints: [
            "讀取 JSONL 後，根據 input 是否為空選擇不同模板",
            "先 tokenize 純 prompt（不含 output），記錄其長度作為 mask 邊界",
            "labels 中 prompt 部分設為 -100，response 部分為真實 token id",
            "Pad 到 max_length：input_ids 用 pad_token_id，labels 用 -100",
          ],
          pseudocode: "for line in jsonl_file:\n  data = json.loads(line)\n  prompt = format_template(instruction, input)\n  full_text = prompt + output\n  prompt_ids = tokenizer.encode(prompt)\n  full_ids = tokenizer.encode(full_text)\n  labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]\n  pad/truncate to max_length\n  store sample",
        },
        {
          id: "format_prompt", title: "TODO 2: InstructionDataset.__len__ & __getitem__",
          description: "實作 Dataset 的標準方法：__len__ 返回樣本數，__getitem__ 返回包含 input_ids、attention_mask、labels 的 dict。",
          labFile: "labs/phase8_instruction_tuning/phase_8/sft.py",
          hints: [
            "__len__ 直接返回 self.samples 的長度",
            "__getitem__ 返回 dict，包含 torch.Tensor 格式的 input_ids、attention_mask、labels",
            "attention_mask: 1 表示真實 token，0 表示 padding",
          ],
          pseudocode: "def __len__: return len(self.samples)\ndef __getitem__(idx):\n  sample = self.samples[idx]\n  return {\n    'input_ids': tensor(sample.input_ids),\n    'attention_mask': tensor(sample.attention_mask),\n    'labels': tensor(sample.labels)\n  }",
        },
        {
          id: "train_sft", title: "TODO 3: train_sft",
          description: "實作 SFT 訓練迴圈：forward pass、shift logits/labels、masked cross-entropy loss、backward、optimizer step。",
          labFile: "labs/phase8_instruction_tuning/phase_8/sft.py",
          hints: [
            "記得 shift：logits[:, :-1] 對應 labels[:, 1:]",
            "使用 F.cross_entropy(..., ignore_index=-100) 自動忽略 masked positions",
            "每個 epoch 結束後計算 validation loss（如果 val_loader 不為 None）",
          ],
          pseudocode: "model.to(device), model.train()\noptimizer = AdamW(model.parameters(), lr)\nfor epoch:\n  for batch in train_loader:\n    logits = model(input_ids)\n    logits = logits[:, :-1]\n    labels = labels[:, 1:]\n    loss = CE(logits.view(-1, V), labels.view(-1), ignore=-100)\n    loss.backward()\n    optimizer.step(), zero_grad()\n  epoch_losses.append(avg_loss)",
        },
        {
          id: "preference_dataset", title: "TODO 4: PreferenceDataset",
          description: "載入偏好對資料（prompt, chosen, rejected），tokenize 並建立 DPO 訓練所需的資料結構。",
          labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py",
          hints: [
            "每個樣本需要 tokenize 兩個文本：prompt+chosen 和 prompt+rejected",
            "記錄 prompt_length（in tokens）供 get_log_probs 做 masking",
            "__getitem__ 返回 chosen_ids, chosen_mask, rejected_ids, rejected_mask, prompt_length",
          ],
          pseudocode: "for line in jsonl:\n  prompt, chosen, rejected = parse(line)\n  chosen_ids = tokenize(prompt + chosen)\n  rejected_ids = tokenize(prompt + rejected)\n  prompt_len = len(tokenize(prompt))\n  pad/truncate all to max_length\n  store sample",
        },
        {
          id: "dpo_loss", title: "TODO 5: dpo_loss",
          description: "實作 DPO loss：計算 log ratio chosen/rejected，套用 β scaling，取 -logsigmoid 的 mean。",
          labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py",
          hints: [
            "log_ratio_chosen = policy_chosen_lp - ref_chosen_lp",
            "log_ratio_rejected = policy_rejected_lp - ref_rejected_lp",
            "用 F.logsigmoid 代替 log(sigmoid(...)) 以確保數值穩定",
          ],
          pseudocode: "log_ratio_w = policy_chosen - ref_chosen\nlog_ratio_l = policy_rejected - ref_rejected\nloss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()\nreturn loss",
        },
      ],
      acceptanceCriteria: [
        "InstructionDataset 正確載入 JSONL 並套用 Alpaca 模板",
        "Label masking 確保 loss 只在 response tokens 上計算（prompt 部分為 -100）",
        "train_sft 正確 shift logits 和 labels，使用 ignore_index=-100",
        "PreferenceDataset 正確處理 chosen 和 rejected 對",
        "dpo_loss 的梯度方向正確：增加 chosen 的 log ratio，減少 rejected 的 log ratio",
        "Reference model 在 DPO 訓練中始終保持 frozen",
      ],
      references: [
        { title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", description: "Rafailov et al. 2023 — DPO 的原始論文，證明可以跳過 reward model 直接優化偏好", url: "https://arxiv.org/abs/2305.18290" },
        { title: "Training language models to follow instructions with human feedback", description: "Ouyang et al. 2022 — InstructGPT 論文，提出 RLHF 三階段流程", url: "https://arxiv.org/abs/2203.02155" },
        { title: "Stanford Alpaca", description: "Stanford 的指令微調研究，使用 GPT-4 生成的指令資料微調 LLaMA", url: "https://github.com/tatsu-lab/stanford_alpaca" },
        { title: "Self-Instruct: Aligning LMs with Self-Generated Instructions", description: "Wang et al. 2023 — 自動生成指令資料的方法", url: "https://arxiv.org/abs/2212.10560" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 9: Mixture of Experts (MoE)
// ─────────────────────────────────────────────────────────────

export const phase9Content: PhaseContent = {
  phaseId: 9,
  color: "#A855F7",
  accent: "#C084FC",
  lessons: [
    {
      phaseId: 9, lessonId: 1,
      title: "MoE Architecture",
      subtitle: "用更多參數但不增加計算量——混合專家模型的設計哲學",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 MoE 的核心動機：擴展參數量而不等比擴展計算量",
        "實作 Expert（標準 FFN）、Router（gating network）和 MoELayer",
        "理解 Top-k routing 與 load balancing loss 的必要性",
        "組裝 MoETransformerBlock 並建構完整的 MoEGPT 模型",
      ],
      sections: [
        {
          title: "Why Mixture of Experts?",
          blocks: [
            { type: "paragraph", text: "到目前為止，我們的模型有一個「浪費」的問題：不管你問的是數學題還是寫詩，所有 70 億個參數都會被啟動。但人類不是這樣工作的——做數學題時你用的腦區跟寫詩時不同。Mixture of Experts（MoE）就是借鑒了這個思路：模型有 8 個「專家」，但每個 token 只啟動其中 2 個。結果是：模型可以有 560 億參數的「知識容量」，但每個 token 的計算量只相當於 70 億參數。聽起來像是白嫖？今天我們來看看這個魔法是怎麼實現的。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：MoE 解決了深度學習中最根本的困境之一：模型容量和推論成本的矛盾。傳統的 dense model（如 GPT-3 175B）每個 token 都要經過所有 175B 個參數——這很浪費，因為不是每個 token 都需要那麼多計算。MoE 的思路是：有 8 個「專家」，但每個 token 只啟動其中 2 個。這樣模型總參數可以很大（容量充足），但單個 token 的計算量保持不變。Mixtral 8x7B 的效果接近 LLaMA-70B，但推論速度和 LLaMA-13B 差不多。" },
            { type: "paragraph", text: "Dense Transformer 有一個根本限制：每個 token 都會經過模型的所有參數。參數量增加意味著計算量等比增加。但我們觀察到，並非所有知識都與每個 token 相關——處理程式碼的神經元不需要在處理詩歌時也被激活。" },
            { type: "paragraph", text: "Mixture of Experts (MoE) 的核心思想是：用一組「專家」（Expert）替代單一的大型 FFN，每個 token 只被路由到其中少數幾個專家。這樣，模型的總參數量可以很大（所有專家的參數總和），但每個 token 實際使用的計算量（active parameters）只是其中一小部分。" },
            { type: "table", headers: ["特性", "Dense Transformer", "MoE Transformer"], rows: [
              ["FFN 結構", "1 個大型 FFN", "N 個小型 Expert FFN"],
              ["每 token 激活參數", "100%", "Top-k/N（例如 2/8 = 25%）"],
              ["總參數量", "P", "~N×P（可達數倍）"],
              ["FLOPs per token", "與參數量成正比", "與 active 參數量成正比"],
              ["典型應用", "GPT-4（部分）、LLaMA", "Mixtral 8x7B、GPT-4（推測）"],
              ["訓練難度", "簡單", "需要 load balancing"],
            ]},
            { type: "callout", variant: "info", text: "Mixtral 8x7B 有 8 個專家，每個 token 路由到 2 個專家。總參數約 47B，但 active 參數只有約 13B——與 LLaMA-13B 的推論速度相近，但效果接近 LLaMA-34B。" },
          ],
        },
        {
          title: "Expert Network",
          blocks: [
            { type: "paragraph", text: "每個 Expert 就是一個標準的 Transformer FFN：Linear → GELU → Linear。唯一的差別是我們有多個這樣的 FFN，而不是一個。" },
            { type: "code", language: "python", code: "class Expert(nn.Module):\n    \"\"\"單個專家網路 = 標準 FFN。\"\"\"\n    def __init__(self, d_model: int, d_ff: int):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(d_model, d_ff),   # 擴展: d_model → 4*d_model\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),   # 壓縮: 4*d_model → d_model\n        )\n\n    def forward(self, x):\n        return self.net(x)" },
            { type: "callout", variant: "tip", text: "Expert 的架構刻意與 dense FFN 相同，這樣 MoE 可以視為 dense Transformer 的「drop-in replacement」。每個 Expert 學習不同的特徵轉換。" },
          ],
        },
        {
          title: "Router / Gate Network",
          blocks: [
            { type: "paragraph", text: "Router（也稱 Gate）是 MoE 的「大腦」——它決定每個 token 應該被哪些 Expert 處理。Router 是一個簡單的線性層，將 token 的表示映射到 N 維空間（N = 專家數），再用 softmax 轉換為概率分布。" },
            { type: "diagram", content: "Router 機制：\n\n  token embedding (d_model)\n         │\n         ▼\n  ┌──────────────────┐\n  │ Linear(d_model, N)│   N = 專家數 (e.g., 8)\n  └──────────────────┘\n         │\n         ▼\n    gate logits (N)\n         │\n         ▼\n     softmax(N)\n         │\n         ▼\n  router probs: [0.05, 0.35, 0.02, 0.40, 0.03, 0.05, 0.08, 0.02]\n         │\n         ▼\n      top-k = 2\n         │\n         ▼\n  selected: Expert 3 (0.40), Expert 1 (0.35)\n  weights (renormalized): [0.533, 0.467]" },
            { type: "code", language: "python", code: "class Router(nn.Module):\n    def __init__(self, d_model, n_experts, top_k=2):\n        super().__init__()\n        self.gate = nn.Linear(d_model, n_experts)\n        self.top_k = top_k\n        self.n_experts = n_experts\n\n    def forward(self, x):  # x: (B, T, d_model)\n        B, T, D = x.shape\n        x_flat = x.view(B * T, D)  # (B*T, d_model)\n\n        logits = self.gate(x_flat)  # (B*T, n_experts)\n        probs = F.softmax(logits, dim=-1)  # routing probabilities\n\n        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)\n        # Renormalize so selected weights sum to 1\n        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)\n\n        return probs, top_k_weights, top_k_indices" },
            { type: "callout", variant: "warning", text: "Top-k 後必須 renormalize weights！如果兩個專家的原始 probability 是 0.4 和 0.3，renormalize 後變為 0.571 和 0.429（和為 1）。這確保了 MoE 的輸出 scale 一致。" },
          ],
        },
        {
          title: "MoE Layer: 完整組裝",
          blocks: [
            { type: "paragraph", text: "MoELayer 將 Router 和 Experts 組合在一起。對每個 token，Router 選出 top-k 個 Expert，每個 Expert 獨立處理該 token，最後用路由權重加權合併輸出。" },
            { type: "diagram", content: "MoE Layer 完整架構：\n\n  input: (B, T, d_model)\n         │\n    ┌─────┴─────────────────────────┐\n    │           Router              │\n    │  決定 top-k experts per token  │\n    └─────┬─────────────────────────┘\n          │  routing weights & indices\n          ▼\n    ┌─────────────────────────────────────┐\n    │  Expert 0  Expert 1  ...  Expert N  │\n    │    FFN       FFN           FFN      │\n    │     │         │             │       │\n    │     ▼         ▼             ▼       │\n    │  只處理被路由到的 token               │\n    └─────────────────────────────────────┘\n          │\n          ▼\n    weighted sum of selected expert outputs\n          │\n          ▼\n  output: (B, T, d_model)" },
            { type: "code", language: "python", code: "class MoELayer(nn.Module):\n    def __init__(self, d_model, d_ff, n_experts, top_k=2):\n        super().__init__()\n        self.experts = nn.ModuleList(\n            [Expert(d_model, d_ff) for _ in range(n_experts)]\n        )\n        self.router = Router(d_model, n_experts, top_k)\n        self.n_experts = n_experts\n        self.top_k = top_k\n\n    def forward(self, x):  # x: (B, T, D)\n        B, T, D = x.shape\n        router_probs, top_k_weights, top_k_indices = self.router(x)\n\n        x_flat = x.view(B * T, D)\n        output = torch.zeros_like(x_flat)\n\n        for expert_idx in range(self.n_experts):\n            # 找出哪些 token 被路由到此 expert\n            mask = (top_k_indices == expert_idx).any(dim=-1)\n            if not mask.any():\n                continue\n\n            expert_input = x_flat[mask]  # 只取被路由的 token\n            expert_output = self.experts[expert_idx](expert_input)\n\n            # 計算此 expert 對這些 token 的權重\n            weight_mask = (top_k_indices[mask] == expert_idx)\n            weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)\n\n            output[mask] += weights.unsqueeze(-1) * expert_output\n\n        return output.view(B, T, D), router_probs" },
          ],
        },
        {
          title: "Load Balancing Loss",
          blocks: [
            { type: "paragraph", text: "MoE 有一個嚴重的訓練問題：「專家崩塌」(expert collapse)。Router 傾向於將大部分 token 都路由到少數幾個表現好的 Expert，其他 Expert 得不到訓練，表現越來越差，形成惡性循環。最終可能只有 1-2 個 Expert 在實際工作。" },
            { type: "heading", level: 3, text: "Auxiliary Load Balancing Loss" },
            { type: "paragraph", text: "解決方案是添加一個輔助 loss，鼓勵路由分布均勻：" },
            { type: "diagram", content: "Load Balancing Loss:\n\n  L_aux = N × Σᵢ (fᵢ × pᵢ)\n\n  fᵢ = 被路由到 Expert i 的 token 比例 (dispatch fraction)\n  pᵢ = 所有 token 對 Expert i 的平均路由概率 (routing probability)\n  N  = 專家數\n\n理想情況（完全均勻）：\n  fᵢ = 1/N for all i  (等比例分配)\n  pᵢ = 1/N for all i  (等概率路由)\n  L_aux = N × N × (1/N × 1/N) = 1  (最小值)\n\n最差情況（完全崩塌到一個 Expert）：\n  f₀ = 1, pᵢ = 1, 其餘為 0\n  L_aux = N × 1 × 1 = N  (最大值)" },
            { type: "code", language: "python", code: "def load_balancing_loss(router_probs, top_k_indices, n_experts):\n    # fᵢ: fraction of tokens dispatched to expert i\n    expert_mask = F.one_hot(top_k_indices, n_experts)  # (n_tokens, top_k, n_experts)\n    expert_mask = expert_mask.sum(dim=1)  # (n_tokens, n_experts)\n    f = expert_mask.float().mean(dim=0)   # (n_experts,)\n\n    # pᵢ: mean routing probability for expert i\n    p = router_probs.mean(dim=0)  # (n_experts,)\n\n    # L_aux = N * sum(fᵢ * pᵢ)\n    return n_experts * (f * p).sum()" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Load balancing loss 是 MoE 中最 tricky 的部分。沒有它，Router 會傾向把所有 token 都送到同一個「最好的」專家——這就是「專家坍縮」（expert collapse）。一旦坍縮，MoE 就退化成了普通的 dense model，多出來的專家完全浪費。Auxiliary loss 強迫 router 把 token 均勻分配到所有專家。它是一個 hack，但是一個有效的 hack。Switch Transformer 論文中花了大量篇幅討論這個問題。" },
            { type: "callout", variant: "info", text: "最終的總 loss = CE loss + λ × L_aux，其中 λ (aux_loss_weight) 通常設為 0.01-0.1。太大會犧牲語言建模能力，太小則無法防止崩塌。" },
          ],
        },
        {
          title: "MoE Transformer Block & MoEGPT",
          blocks: [
            { type: "callout", variant: "quote", text: "💡 講師心得：實務中不會把每一層都換成 MoE——通常是每隔一層。例如 Mixtral 是「dense attention + MoE FFN」交替。原因是 attention 層已經很高效了（它是共享的），而 FFN 佔了模型大部分參數。只把 FFN 換成 MoE 就能獲得大部分的容量提升，同時保持架構的穩定性。" },
            { type: "paragraph", text: "MoETransformerBlock 的結構與標準 Transformer Block 完全相同，唯一差別是用 MoELayer 替代 dense FFN。在完整的 MoEGPT 模型中，我們交替使用 Dense 和 MoE block（類似 Mixtral 的設計）。" },
            { type: "code", language: "python", code: "class MoETransformerBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(config.n_embd)\n        self.attn = SampleAttention(config)\n        self.ln2 = nn.LayerNorm(config.n_embd)\n        self.moe = MoELayer(\n            config.n_embd, 4 * config.n_embd,\n            config.n_experts, config.top_k\n        )\n\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        moe_out, router_probs = self.moe(self.ln2(x))\n        x = x + moe_out\n        return x, router_probs" },
            { type: "heading", level: 3, text: "MoEGPT: 交錯 Dense/MoE Blocks" },
            { type: "diagram", content: "MoEGPT 架構 (6 layers, moe_every_n_layers=2):\n\n  Token IDs → Token Embedding + Position Embedding\n                              │\n                              ▼\n             ┌──────────────────────────────┐\n  Layer 0:   │  Dense Transformer Block     │  (standard FFN)\n             └──────────────────────────────┘\n                              │\n             ┌──────────────────────────────┐\n  Layer 1:   │  MoE Transformer Block       │  ← (1+1) % 2 == 0\n             │  (Router → 8 Experts, top-2) │\n             └──────────────────────────────┘\n                              │\n             ┌──────────────────────────────┐\n  Layer 2:   │  Dense Transformer Block     │\n             └──────────────────────────────┘\n                              │\n             ┌──────────────────────────────┐\n  Layer 3:   │  MoE Transformer Block       │  ← (3+1) % 2 == 0\n             └──────────────────────────────┘\n                              │\n             ┌──────────────────────────────┐\n  Layer 4:   │  Dense Transformer Block     │\n             └──────────────────────────────┘\n                              │\n             ┌──────────────────────────────┐\n  Layer 5:   │  MoE Transformer Block       │  ← (5+1) % 2 == 0\n             └──────────────────────────────┘\n                              │\n                              ▼\n             LayerNorm → Linear (vocab_size) → logits" },
            { type: "code", language: "python", code: "class MoEGPT(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)\n        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)\n\n        self.blocks = nn.ModuleList()\n        for i in range(config.n_layer):\n            if (i + 1) % config.moe_every_n_layers == 0:\n                self.blocks.append(MoETransformerBlock(config))\n            else:\n                self.blocks.append(DenseTransformerBlock(config))\n\n        self.ln_f = nn.LayerNorm(config.n_embd)\n        self.head = nn.Linear(config.n_embd, config.vocab_size)\n        self.config = config\n\n    def forward(self, idx, targets=None):\n        B, T = idx.shape\n        tok = self.tok_emb(idx)\n        pos = self.pos_emb(torch.arange(T, device=idx.device))\n        x = tok + pos\n\n        all_router_probs = []\n        for block in self.blocks:\n            x, router_probs = block(x)\n            if router_probs is not None:\n                all_router_probs.append(router_probs)\n\n        logits = self.head(self.ln_f(x))\n\n        loss = None\n        if targets is not None:\n            ce_loss = F.cross_entropy(\n                logits.view(-1, logits.size(-1)), targets.view(-1)\n            )\n            # Auxiliary load balancing loss\n            aux_loss = torch.stack([\n                load_balancing_loss(rp, ...)\n                for rp in all_router_probs\n            ]).mean()\n            loss = ce_loss + config.aux_loss_weight * aux_loss\n\n        return logits, loss" },
            { type: "callout", variant: "tip", text: "Mixtral 的設計選擇：每隔一層使用 MoE（moe_every_n_layers=2）。這是效率與效果的平衡——全部使用 MoE 會增加 routing overhead 和 load balancing 的難度，而交錯設計讓 dense layers 提供穩定的「共享知識」基礎。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：MoE 的 router 是一個簡單的 Linear 層 + Softmax。但這意味著不同的 token 可能被路由到不同的專家——同一個 batch 裡，有的專家可能收到很多 token，有的幾乎沒有。這會造成什麼工程問題？（提示：想想 GPU 並行計算需要什麼條件。Mixtral 是怎麼處理這個問題的？）" },
          ],
        },
      ],
      exercises: [
        {
          id: "expert_init_forward", title: "TODO 1: Expert",
          description: "實作單個 Expert 模組：標準 FFN (Linear → GELU → Linear)。",
          labFile: "labs/phase9_moe/phase_9/moe.py",
          hints: [
            "使用 nn.Sequential 組合 Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model)",
            "forward 直接 return self.net(x)",
          ],
          pseudocode: "def __init__(d_model, d_ff):\n  self.net = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))\ndef forward(x): return self.net(x)",
        },
        {
          id: "router_init_forward", title: "TODO 2: Router",
          description: "實作路由網路：Linear gate → softmax → top-k selection → renormalize weights。",
          labFile: "labs/phase9_moe/phase_9/moe.py",
          hints: [
            "gate 是 nn.Linear(d_model, n_experts)",
            "先 reshape x 為 (B*T, D)，再通過 gate 得到 logits",
            "softmax 得到 probs，torch.topk 選 top-k",
            "renormalize: top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)",
          ],
          pseudocode: "def __init__: self.gate = Linear(d_model, n_experts)\ndef forward(x):\n  x_flat = x.view(B*T, D)\n  logits = self.gate(x_flat)\n  probs = softmax(logits, dim=-1)\n  top_k_w, top_k_idx = topk(probs, self.top_k)\n  top_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)\n  return probs, top_k_w, top_k_idx",
        },
        {
          id: "moe_layer", title: "TODO 3: MoELayer",
          description: "組裝 MoE 層：建立 N 個 Expert + 1 個 Router，forward 時路由每個 token 到 top-k experts 並加權合併輸出。",
          labFile: "labs/phase9_moe/phase_9/moe.py",
          hints: [
            "用 nn.ModuleList 建立 experts 列表",
            "forward 中遍歷每個 expert，找出被路由到它的 token（mask），處理後用 weight 加權累加到 output",
            "weight_mask = (top_k_indices[mask] == expert_idx)，weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)",
          ],
          pseudocode: "def forward(x):\n  probs, topk_w, topk_idx = self.router(x)\n  x_flat = x.view(B*T, D)\n  output = zeros_like(x_flat)\n  for i in range(n_experts):\n    mask = (topk_idx == i).any(dim=-1)\n    if mask.any():\n      out_i = self.experts[i](x_flat[mask])\n      w = get_weight_for_expert(topk_w, topk_idx, mask, i)\n      output[mask] += w.unsqueeze(-1) * out_i\n  return output.view(B,T,D), probs",
        },
        {
          id: "load_balancing_loss", title: "TODO 4: load_balancing_loss",
          description: "實作輔助 load balancing loss：L_aux = N * sum(fi * pi)，防止 expert collapse。",
          labFile: "labs/phase9_moe/phase_9/moe.py",
          hints: [
            "f_i: 用 F.one_hot(top_k_indices, n_experts).sum(dim=1) 得到每個 token 選了哪些 expert，再取 mean 得到比例",
            "p_i: router_probs.mean(dim=0)",
            "return n_experts * (f * p).sum()",
          ],
          pseudocode: "expert_mask = one_hot(top_k_indices, N).sum(dim=1)\nf = expert_mask.float().mean(dim=0)\np = router_probs.mean(dim=0)\nreturn N * (f * p).sum()",
        },
        {
          id: "moe_transformer_block", title: "TODO 5: MoETransformerBlock",
          description: "實作 MoE Transformer Block：LayerNorm → Attention → residual → LayerNorm → MoELayer → residual。",
          labFile: "labs/phase9_moe/phase_9/moe_transformer.py",
          hints: [
            "結構與 DenseTransformerBlock 相同，只是 FFN 替換為 MoELayer",
            "MoELayer 返回 (output, router_probs)，需要一起返回",
            "MoELayer 參數：d_model=n_embd, d_ff=4*n_embd, n_experts, top_k",
          ],
          pseudocode: "def __init__: ln1, attn, ln2, moe = MoELayer(...)\ndef forward(x):\n  x = x + attn(ln1(x))\n  moe_out, router_probs = self.moe(ln2(x))\n  x = x + moe_out\n  return x, router_probs",
        },
        {
          id: "moe_gpt", title: "TODO 6: MoEGPT",
          description: "建構完整 MoE GPT 模型：token/position embeddings、交錯 Dense/MoE blocks、final LayerNorm + head，forward 包含 CE loss + auxiliary loss。",
          labFile: "labs/phase9_moe/phase_9/moe_transformer.py",
          hints: [
            "Block i 是 MoE 如果 (i+1) % moe_every_n_layers == 0",
            "收集每個 MoE block 的 router_probs 計算 auxiliary loss",
            "total loss = CE loss + aux_loss_weight * mean(aux_losses)",
          ],
          pseudocode: "def __init__:\n  tok_emb, pos_emb\n  blocks = [MoEBlock if (i+1)%N==0 else DenseBlock for i]\n  ln_f, head\ndef forward(idx, targets):\n  x = tok_emb(idx) + pos_emb(arange(T))\n  router_probs_list = []\n  for block in blocks:\n    x, rp = block(x)\n    if rp is not None: collect rp\n  logits = head(ln_f(x))\n  if targets:\n    ce = cross_entropy(logits, targets)\n    aux = mean([load_balancing_loss(rp) for rp])\n    loss = ce + weight * aux\n  return logits, loss",
        },
      ],
      acceptanceCriteria: [
        "Expert 的輸入輸出 shape 一致：(batch, seq, d_model) → (batch, seq, d_model)",
        "Router 輸出的 top_k_weights 每行和為 1（renormalized）",
        "MoELayer 的輸出 shape 與輸入相同，且所有 token 都被處理",
        "load_balancing_loss 在均勻路由時為最小值，崩塌時為最大值",
        "MoETransformerBlock 正確返回 (output, router_probs)",
        "MoEGPT 的 loss 包含 CE loss 和 auxiliary load balancing loss",
        "交錯 Dense/MoE 的模式正確：(i+1) % moe_every_n_layers == 0",
      ],
      references: [
        { title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", description: "Shazeer et al. 2017 — MoE 在深度學習中的經典論文，引入 top-k gating 和 load balancing", url: "https://arxiv.org/abs/1701.06538" },
        { title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", description: "Fedus et al. 2022 — 提出 top-1 routing 的簡化設計", url: "https://arxiv.org/abs/2101.03961" },
        { title: "Mixtral of Experts", description: "Jiang et al. 2024 — Mistral AI 的開源 MoE 模型，展示了交錯 Dense/MoE 設計的實際效果", url: "https://arxiv.org/abs/2401.04088" },
        { title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", description: "Lepikhin et al. 2021 — Google 的大規模 MoE 訓練框架", url: "https://arxiv.org/abs/2006.16668" },
      ],
    },
  ],
};
