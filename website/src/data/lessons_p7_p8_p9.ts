import type { PhaseContent } from "@/data/types";
import type { Locale } from "@/i18n";

// ─────────────────────────────────────────────────────────────
// Phase 7: LoRA — zh-TW
// ─────────────────────────────────────────────────────────────

const phase7ContentZhTW: PhaseContent = {
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
            { type: "diagram", content: "                    LoRALinear 層\n    ┌──────────────────────────────────────┐\n    │                                      │\n    │  ┌──────────────────────┐             │\n    │  │   W (frozen)         │  ──┐        │\n    │  │  (out × in)          │    │        │\n    │  └──────────────────────┘    │        │\nx ──┤                              ├── + ── ├──→ output\n    │  ┌─────┐    ┌──────────┐    │        │\n    │  │  A  │ →  │    B     │ ───┘        │\n    │  │in×r │    │  r×out   │  × (α/r)    │\n    │  └─────┘    └──────────┘             │\n    │   trainable   trainable              │\n    └──────────────────────────────────────┘" },
            { type: "heading", level: 3, text: "__init__ 實作要點" },
            { type: "code", language: "python", code: "class LoRALinear(nn.Module):\n    def __init__(self, in_features, out_features, rank=4, alpha=1.0, bias=True):\n        super().__init__()\n        self.weight = nn.Parameter(\n            torch.empty(out_features, in_features), requires_grad=False\n        )\n        if bias:\n            self.bias = nn.Parameter(\n                torch.zeros(out_features), requires_grad=False\n            )\n        else:\n            self.bias = None\n        self.lora_A = nn.Parameter(torch.empty(in_features, rank))\n        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))\n        self.scaling = alpha / rank" },
            { type: "heading", level: 3, text: "forward 實作" },
            { type: "code", language: "python", code: "def forward(self, x):\n    base_out = F.linear(x, self.weight, self.bias)\n    lora_out = (x @ self.lora_A @ self.lora_B.T) * self.scaling\n    return base_out + lora_out" },
            { type: "callout", variant: "tip", text: "注意 F.linear(x, W, bias) 計算的是 x @ W^T + bias，其中 W 的 shape 是 (out, in)，這是 PyTorch 的慣例。而 LoRA path 中 A 是 (in, r)、B 是 (r, out)，所以 x @ A @ B^T 的結果 shape 正確。" },
          ],
        },
        {
          title: "Applying LoRA to Model & Merging Weights",
          blocks: [
            { type: "paragraph", text: "有了 LoRALinear 後，我們需要把它注入到現有的 Transformer 模型中。典型的做法是替換 attention 中的 Q、V 投影層（q_proj、v_proj），因為這些層對任務適應最為關鍵。" },
            { type: "heading", level: 3, text: "apply_lora_to_model" },
            { type: "code", language: "python", code: "def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):\n    if target_modules is None:\n        target_modules = ['q_proj', 'v_proj']\n    for param in model.parameters():\n        param.requires_grad = False\n    replacements = []\n    for name, module in model.named_modules():\n        if isinstance(module, nn.Linear):\n            if any(name.endswith(t) for t in target_modules):\n                replacements.append((name, module))\n    for name, module in replacements:\n        parent_name = '.'.join(name.split('.')[:-1])\n        attr_name = name.split('.')[-1]\n        parent = dict(model.named_modules())[parent_name]\n        setattr(parent, attr_name, LoRALinear.from_linear(module, rank, alpha))\n    return model" },
            { type: "heading", level: 3, text: "合併 LoRA 權重" },
            { type: "paragraph", text: "訓練完成後，我們可以把 LoRA 的低秩更新合併回原始權重。合併後的模型與原始模型架構完全相同（普通的 nn.Linear），推論時沒有額外開銷。" },
            { type: "code", language: "python", code: "def merge_lora_weights(model):\n    for name, module in model.named_modules():\n        if isinstance(module, LoRALinear):\n            merged = module.weight + module.scaling * (\n                module.lora_B.T @ module.lora_A.T\n            )\n            new_linear = nn.Linear(\n                module.weight.shape[1], module.weight.shape[0],\n                bias=module.bias is not None\n            )\n            new_linear.weight.data = merged\n            if module.bias is not None:\n                new_linear.bias.data = module.bias.data\n    return model" },
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
            { type: "code", language: "python", code: "def count_trainable_params(model):\n    total = sum(p.numel() for p in model.parameters())\n    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    print(f'Total params:     {total:,}')\n    print(f'Trainable params: {trainable:,} ({100*trainable/total:.2f}%)')\n    return trainable, total" },
            { type: "callout", variant: "tip", text: "QLoRA 更進一步：它將 frozen weights 量化為 4-bit（NF4 格式），大幅降低記憶體需求。搭配分頁優化器，甚至可以在單張 24 GB 的消費級 GPU 上微調 65B 模型。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：LoRA 只對 Linear 層（通常是 attention 的 W_q, W_k, W_v, W_o）注入低秩矩陣。但 FFN 層佔了模型一半以上的參數——為什麼通常不對 FFN 做 LoRA？如果對 FFN 也做 LoRA，效果會更好嗎？這跟 FFN 和 Attention 各自學到的知識類型有什麼關係？" },
          ],
        },
      ],
      exercises: [
        { id: "lora_linear_init", title: "TODO 1: LoRALinear.__init__", description: "初始化 LoRALinear 模組：建立 frozen weight、trainable lora_A（Kaiming uniform）、trainable lora_B（zero init），並計算 scaling = alpha / rank。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["使用 nn.Parameter(..., requires_grad=False) 建立 frozen weight", "lora_A shape: (in_features, rank)，用 nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))", "lora_B shape: (rank, out_features)，初始化為 torch.zeros", "B 初始化為零確保訓練開始時 LoRA 貢獻為零"], pseudocode: "self.weight = Parameter(empty(out, in), requires_grad=False)\nself.lora_A = Parameter(empty(in, rank))\nkaiming_uniform_(self.lora_A)\nself.lora_B = Parameter(zeros(rank, out))\nself.scaling = alpha / rank" },
        { id: "lora_linear_forward", title: "TODO 2: LoRALinear.forward", description: "實作 LoRA 的前向傳播：frozen path (F.linear) + trainable low-rank bypass (x @ A @ B^T * scaling)。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["base output: F.linear(x, self.weight, self.bias)", "LoRA path: (x @ self.lora_A @ self.lora_B.T) * self.scaling", "兩者相加即為最終輸出"], pseudocode: "base = F.linear(x, self.weight, self.bias)\nlora = (x @ self.lora_A @ self.lora_B.T) * self.scaling\nreturn base + lora" },
        { id: "apply_lora_to_model", title: "TODO 3: apply_lora_to_model", description: "將模型中指定的 nn.Linear 層替換為 LoRALinear。先凍結所有參數，再替換目標層。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["先 freeze 所有參數：for param in model.parameters(): param.requires_grad = False", "用 model.named_modules() 遍歷，找到目標 nn.Linear", "用 setattr(parent, attr_name, new_module) 執行替換"], pseudocode: "freeze all params\nfor name, module in model.named_modules():\n  if Linear and name ends with target:\n    setattr(parent, attr, LoRALinear.from_linear(module, rank, alpha))" },
        { id: "merge_lora_weights", title: "TODO 4: merge_lora_weights", description: "將 LoRA 權重合併回原始權重矩陣，產生標準 nn.Linear 供高效推論。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["merged_weight = lora.weight + lora.scaling * (lora.lora_B.T @ lora.lora_A.T)", "建立新的 nn.Linear 並複製合併後的 weight 和 bias"], pseudocode: "for LoRALinear modules:\n  W_merged = weight + scaling * (B^T @ A^T)\n  new_linear = nn.Linear(...)\n  new_linear.weight.data = W_merged" },
        { id: "count_trainable_params", title: "TODO 5: count_trainable_params", description: "統計模型的可訓練參數量與總參數量，並印出摘要。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["total = sum(p.numel() for p in model.parameters())", "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)"], pseudocode: "total = sum numel for all params\ntrainable = sum numel for requires_grad params\nreturn (trainable, total)" },
      ],
      acceptanceCriteria: [
        "LoRALinear 初始化後，B 為零矩陣，forward 輸出與原始 nn.Linear 一致",
        "apply_lora_to_model 後，只有 lora_A 和 lora_B 的 requires_grad 為 True",
        "merge_lora_weights 後，模型不含任何 LoRALinear，且輸出與合併前一致",
        "count_trainable_params 正確報告可訓練與凍結參數數量",
      ],
      references: [
        { title: "LoRA: Low-Rank Adaptation of Large Language Models", description: "Hu et al. 2021", url: "https://arxiv.org/abs/2106.09685" },
        { title: "QLoRA: Efficient Finetuning of Quantized LLMs", description: "Dettmers et al. 2023", url: "https://arxiv.org/abs/2305.14314" },
        { title: "HuggingFace PEFT Library", description: "LoRA、QLoRA 等方法的官方實作庫", url: "https://github.com/huggingface/peft" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 7: LoRA — zh-CN
// ─────────────────────────────────────────────────────────────

const phase7ContentZhCN: PhaseContent = {
  phaseId: 7,
  color: "#F97316",
  accent: "#FB923C",
  lessons: [
    {
      phaseId: 7, lessonId: 1,
      title: "LoRA: Low-Rank Adaptation",
      subtitle: "用极少参数微调大型模型——低秩分解的力量",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 full fine-tuning 的内存与计算成本，并分析参数量",
        "掌握低秩分解（low-rank decomposition）的数学直觉",
        "实现 LoRALinear 模块，包含 frozen weight 与 trainable low-rank matrices",
        "将 LoRA 注入现有模型，并学会合并权重以供推理部署",
        "分析 LoRA 的参数效率，比较 Full FT、LoRA 与 QLoRA",
      ],
      sections: [
        {
          title: "Why Full Fine-Tuning is Expensive",
          blocks: [
            { type: "paragraph", text: "假设你有一个 7B 参数的模型，想让它学会写法律文书。Full fine-tuning 意味着更新全部 70 亿个参数——你需要一张 80GB 的 A100 GPU，训练好几天。但如果我告诉你，只更新其中 0.1% 的参数就能达到 95% 的效果呢？这就是 LoRA 的魔法。今天我们来揭开它的秘密——为什么这么少的参数就够了？" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：LoRA 是过去几年最具影响力的效率化技术之一。它的核心洞察惊人地简单：模型微调时的权重变化（ΔW）其实是低秩的——也就是说，你不需要更新所有参数，只需要在一个低维子空间中调整就够了。这就像你搬进新房子，不需要重新装修整栋楼，只需要改几面墙的颜色。" },
            { type: "paragraph", text: "当我们拿到一个预训练好的 LLM（例如 LLaMA-7B），想让它学会新任务时，最直觉的方法是 full fine-tuning：解冻所有参数，用任务数据继续训练。但这代价极高。" },
            { type: "heading", level: 3, text: "参数量分析" },
            { type: "paragraph", text: "考虑一个 7B 参数的模型。Full fine-tuning 需要存储：(1) 模型参数本身（FP16 约 14 GB），(2) 梯度（同样 14 GB），(3) Adam 优化器状态（每个参数需要 first moment + second moment，共 28 GB）。光是这三项就需要约 56 GB 的 GPU 内存，还不包括 activation memory。" },
            { type: "table", headers: ["项目", "FP16 大小 (7B model)", "说明"], rows: [
              ["模型参数", "14 GB", "7B × 2 bytes"],
              ["梯度", "14 GB", "每个参数一个梯度值"],
              ["Adam m (1st moment)", "14 GB", "动量的指数移动平均"],
              ["Adam v (2nd moment)", "14 GB", "梯度平方的指数移动平均"],
              ["合计", "~56 GB", "还需加上 activations"],
            ]},
            { type: "callout", variant: "warning", text: "这意味着 full fine-tuning 一个 7B 模型至少需要一张 80 GB 的 A100 GPU。对大多数研究者和开发者而言，这是不切实际的。我们需要更聪明的方法。" },
            { type: "paragraph", text: "核心观察：fine-tuning 时，模型权重的变化量 ΔW 通常是低秩的（low-rank）。Aghajanyan et al. (2021) 的研究发现，预训练模型具有很低的 intrinsic dimensionality——即使将更新限制在一个很小的子空间中，效果仍然接近 full fine-tuning。这就是 LoRA 的理论基础。" },
          ],
        },
        {
          title: "低秩分解的直觉",
          blocks: [
            { type: "paragraph", text: "低秩分解的核心思想：一个大矩阵可以被近似为两个小矩阵的乘积。如果原始权重矩阵 W 是 d×d 维（例如 4096×4096），完整的更新 ΔW 也是 d×d 维，有 d² 个参数。但如果 ΔW 的秩（rank）只有 r（远小于 d），我们就可以把它分解为 B×A，其中 B 是 d×r、A 是 r×d，参数量从 d² 降为 2dr。" },
            { type: "diagram", content: "Low-Rank Decomposition 可视化：\n\n原始更新 ΔW:  d×d = d² 参数\n┌─────────────────────┐\n│                     │\n│     d × d 矩阵       │  例: 4096 × 4096 = 16,777,216 参数\n│   (full rank)       │\n│                     │\n└─────────────────────┘\n\n低秩近似 B × A:  2 × d × r 参数\n┌───┐   ┌─────────────────────┐\n│   │   │                     │\n│ B │ × │         A           │  例: r = 8\n│d×r│   │       r × d         │  4096×8 + 8×4096 = 65,536 参数\n│   │   │                     │  压缩比: 256×\n└───┘   └─────────────────────┘" },
            { type: "callout", variant: "info", text: "rank r 是 LoRA 的核心超参数。r 越大，表达能力越强但参数越多。实践中 r = 4~16 就能达到接近 full fine-tuning 的效果。" },
          ],
        },
        {
          title: "LoRA 公式与 LoRALinear 实现",
          blocks: [
            { type: "paragraph", text: "LoRA 的正式公式非常优雅：" },
            { type: "diagram", content: "W' = W + (α/r) · B @ A\n\n其中：\n  W:  原始预训练权重 (frozen, requires_grad=False)\n  A:  低秩矩阵, shape (in_features, rank), Kaiming uniform 初始化\n  B:  低秩矩阵, shape (rank, out_features), 初始化为零\n  α:  scaling factor (超参数)\n  r:  rank (超参数)\n\n关键设计：B 初始化为零 → 训练开始时 BA = 0 → 模型行为完全不变" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：α/r 这个 scaling factor 容易被忽略但非常重要。α 控制 LoRA 的「学习率倍数」——α 越大，LoRA 的影响越大。实践中通常设 α = r 或 α = 2r。一个常见的错误是只调 r 而忘了 α，导致 r 增大时效果反而变差（因为 α/r 变小了，LoRA 的影响被稀释）。" },
            { type: "heading", level: 3, text: "LoRA Layer 架构图" },
            { type: "diagram", content: "                    LoRALinear 层\n    ┌──────────────────────────────────────┐\n    │                                      │\n    │  ┌──────────────────────┐             │\n    │  │   W (frozen)         │  ──┐        │\n    │  │  (out × in)          │    │        │\n    │  └──────────────────────┘    │        │\nx ──┤                              ├── + ── ├──→ output\n    │  ┌─────┐    ┌──────────┐    │        │\n    │  │  A  │ →  │    B     │ ───┘        │\n    │  │in×r │    │  r×out   │  × (α/r)    │\n    │  └─────┘    └──────────┘             │\n    │   trainable   trainable              │\n    └──────────────────────────────────────┘" },
            { type: "heading", level: 3, text: "__init__ 实现要点" },
            { type: "code", language: "python", code: "class LoRALinear(nn.Module):\n    def __init__(self, in_features, out_features, rank=4, alpha=1.0, bias=True):\n        super().__init__()\n        self.weight = nn.Parameter(\n            torch.empty(out_features, in_features), requires_grad=False\n        )\n        if bias:\n            self.bias = nn.Parameter(\n                torch.zeros(out_features), requires_grad=False\n            )\n        else:\n            self.bias = None\n        self.lora_A = nn.Parameter(torch.empty(in_features, rank))\n        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))\n        # B = 0 → 初始时 LoRA 贡献为零\n        self.scaling = alpha / rank" },
            { type: "heading", level: 3, text: "forward 实现" },
            { type: "code", language: "python", code: "def forward(self, x):\n    base_out = F.linear(x, self.weight, self.bias)\n    lora_out = (x @ self.lora_A @ self.lora_B.T) * self.scaling\n    return base_out + lora_out" },
            { type: "callout", variant: "tip", text: "注意 F.linear(x, W, bias) 计算的是 x @ W^T + bias，其中 W 的 shape 是 (out, in)，这是 PyTorch 的惯例。而 LoRA path 中 A 是 (in, r)、B 是 (r, out)，所以 x @ A @ B^T 的结果 shape 正确。" },
          ],
        },
        {
          title: "将 LoRA 注入模型与合并权重",
          blocks: [
            { type: "paragraph", text: "有了 LoRALinear 后，我们需要把它注入到现有的 Transformer 模型中。典型的做法是替换 attention 中的 Q、V 投影层（q_proj、v_proj），因为这些层对任务适应最为关键。" },
            { type: "heading", level: 3, text: "apply_lora_to_model" },
            { type: "code", language: "python", code: "def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):\n    if target_modules is None:\n        target_modules = ['q_proj', 'v_proj']\n    for param in model.parameters():\n        param.requires_grad = False\n    replacements = []\n    for name, module in model.named_modules():\n        if isinstance(module, nn.Linear):\n            if any(name.endswith(t) for t in target_modules):\n                replacements.append((name, module))\n    for name, module in replacements:\n        parent_name = '.'.join(name.split('.')[:-1])\n        attr_name = name.split('.')[-1]\n        parent = dict(model.named_modules())[parent_name]\n        setattr(parent, attr_name, LoRALinear.from_linear(module, rank, alpha))\n    return model" },
            { type: "heading", level: 3, text: "合并 LoRA 权重" },
            { type: "paragraph", text: "训练完成后，我们可以把 LoRA 的低秩更新合并回原始权重。合并后的模型与原始模型架构完全相同（普通的 nn.Linear），推理时没有额外开销。" },
            { type: "code", language: "python", code: "def merge_lora_weights(model):\n    for name, module in model.named_modules():\n        if isinstance(module, LoRALinear):\n            merged = module.weight + module.scaling * (\n                module.lora_B.T @ module.lora_A.T\n            )\n            new_linear = nn.Linear(\n                module.weight.shape[1], module.weight.shape[0],\n                bias=module.bias is not None\n            )\n            new_linear.weight.data = merged\n            if module.bias is not None:\n                new_linear.bias.data = module.bias.data\n    return model" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：LoRA 的另一个巧妙之处在于推理时零开销。训练时 BA 是单独的矩阵，但部署前你可以把它合并回 W' = W + (α/r)BA。合并后的模型和原始模型结构完全一样，推理速度不受影响。这意味着你可以为不同任务训练多个 LoRA adapter，部署时选择性加载——一个底座模型，多个任务能力。" },
          ],
        },
        {
          title: "参数效率分析",
          blocks: [
            { type: "paragraph", text: "LoRA 的参数效率令人惊叹。让我们以具体数字来看：" },
            { type: "table", headers: ["方法", "可训练参数", "内存需求", "效果", "推理开销"], rows: [
              ["Full Fine-Tuning", "100%", "~4× 模型大小", "最优（但可能 overfit）", "无"],
              ["LoRA (r=8)", "~0.1-1%", "模型大小 + 少量", "接近 Full FT", "合并后无"],
              ["QLoRA", "~0.1-1%", "~1/4 模型大小 + 少量", "接近 LoRA", "需要 dequant"],
            ]},
            { type: "heading", level: 3, text: "计算示例" },
            { type: "paragraph", text: "对一个 d_model = 4096 的 Transformer 层，每个 attention 有 q_proj 和 v_proj 两个 Linear(4096, 4096)。原始参数量：2 × 4096² = 33,554,432。使用 rank=8 的 LoRA：2 × (4096×8 + 8×4096) = 131,072。压缩比约 256 倍。" },
            { type: "code", language: "python", code: "def count_trainable_params(model):\n    total = sum(p.numel() for p in model.parameters())\n    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    print(f'Total params:     {total:,}')\n    print(f'Trainable params: {trainable:,} ({100*trainable/total:.2f}%)')\n    return trainable, total" },
            { type: "callout", variant: "tip", text: "QLoRA 更进一步：它将 frozen weights 量化为 4-bit（NF4 格式），大幅降低内存需求。搭配分页优化器，甚至可以在单张 24 GB 的消费级 GPU 上微调 65B 模型。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：LoRA 只对 Linear 层（通常是 attention 的 W_q, W_k, W_v, W_o）注入低秩矩阵。但 FFN 层占了模型一半以上的参数——为什么通常不对 FFN 做 LoRA？如果对 FFN 也做 LoRA，效果会更好吗？这跟 FFN 和 Attention 各自学到的知识类型有什么关系？" },
          ],
        },
      ],
      exercises: [
        { id: "lora_linear_init", title: "TODO 1: LoRALinear.__init__", description: "初始化 LoRALinear 模块：建立 frozen weight、trainable lora_A（Kaiming uniform）、trainable lora_B（zero init），并计算 scaling = alpha / rank。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["使用 nn.Parameter(..., requires_grad=False) 建立 frozen weight", "lora_A shape: (in_features, rank)，用 nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))", "lora_B shape: (rank, out_features)，初始化为 torch.zeros", "B 初始化为零确保训练开始时 LoRA 贡献为零"], pseudocode: "self.weight = Parameter(empty(out, in), requires_grad=False)\nself.lora_A = Parameter(empty(in, rank))\nkaiming_uniform_(self.lora_A)\nself.lora_B = Parameter(zeros(rank, out))\nself.scaling = alpha / rank" },
        { id: "lora_linear_forward", title: "TODO 2: LoRALinear.forward", description: "实现 LoRA 的前向传播：frozen path (F.linear) + trainable low-rank bypass (x @ A @ B^T * scaling)。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["base output: F.linear(x, self.weight, self.bias)", "LoRA path: (x @ self.lora_A @ self.lora_B.T) * self.scaling", "两者相加即为最终输出"], pseudocode: "base = F.linear(x, self.weight, self.bias)\nlora = (x @ self.lora_A @ self.lora_B.T) * self.scaling\nreturn base + lora" },
        { id: "apply_lora_to_model", title: "TODO 3: apply_lora_to_model", description: "将模型中指定的 nn.Linear 层替换为 LoRALinear。先冻结所有参数，再替换目标层。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["先冻结所有参数：for param in model.parameters(): param.requires_grad = False", "用 model.named_modules() 遍历，找到目标 nn.Linear", "用 setattr(parent, attr_name, new_module) 执行替换"], pseudocode: "freeze all params\nfor name, module in model.named_modules():\n  if Linear and name ends with target:\n    setattr(parent, attr, LoRALinear.from_linear(module, rank, alpha))" },
        { id: "merge_lora_weights", title: "TODO 4: merge_lora_weights", description: "将 LoRA 权重合并回原始权重矩阵，生成标准 nn.Linear 供高效推理。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["merged_weight = lora.weight + lora.scaling * (lora.lora_B.T @ lora.lora_A.T)", "建立新的 nn.Linear 并复制合并后的 weight 和 bias"], pseudocode: "for LoRALinear modules:\n  W_merged = weight + scaling * (B^T @ A^T)\n  replace with nn.Linear" },
        { id: "count_trainable_params", title: "TODO 5: count_trainable_params", description: "统计模型的可训练参数量与总参数量，并打印摘要。", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["total = sum(p.numel() for p in model.parameters())", "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)"], pseudocode: "total = sum numel for all params\ntrainable = sum numel for requires_grad\nreturn (trainable, total)" },
      ],
      acceptanceCriteria: [
        "LoRALinear 初始化后，B 为零矩阵，forward 输出与原始 nn.Linear 一致",
        "apply_lora_to_model 后，只有 lora_A 和 lora_B 的 requires_grad 为 True",
        "merge_lora_weights 后，模型不含任何 LoRALinear，且输出与合并前一致",
        "count_trainable_params 正确报告可训练与冻结参数数量",
      ],
      references: [
        { title: "LoRA: Low-Rank Adaptation of Large Language Models", description: "Hu et al. 2021", url: "https://arxiv.org/abs/2106.09685" },
        { title: "QLoRA: Efficient Finetuning of Quantized LLMs", description: "Dettmers et al. 2023", url: "https://arxiv.org/abs/2305.14314" },
        { title: "HuggingFace PEFT Library", description: "LoRA、QLoRA 等方法的官方实现库", url: "https://github.com/huggingface/peft" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 7: LoRA — en
// ─────────────────────────────────────────────────────────────

const phase7ContentEn: PhaseContent = {
  phaseId: 7,
  color: "#F97316",
  accent: "#FB923C",
  lessons: [
    {
      phaseId: 7, lessonId: 1,
      title: "LoRA: Low-Rank Adaptation",
      subtitle: "Fine-tuning large models with a tiny fraction of parameters",
      type: "concept",
      duration: "60 min",
      objectives: [
        "Understand the memory and compute costs of full fine-tuning with a parameter count breakdown",
        "Grasp the mathematical intuition behind low-rank decomposition",
        "Implement the LoRALinear module with frozen weights and trainable low-rank matrices",
        "Inject LoRA into an existing model and learn to merge weights for inference",
        "Analyze LoRA's parameter efficiency vs. Full FT and QLoRA",
      ],
      sections: [
        {
          title: "Why Full Fine-Tuning is Expensive",
          blocks: [
            { type: "paragraph", text: "Imagine you have a 7B-parameter model and want it to write legal documents. Full fine-tuning means updating all 7 billion parameters — you'd need an 80 GB A100 GPU and several days of training. But what if I told you that updating just 0.1% of those parameters gets you 95% of the performance? That's the magic of LoRA. Today we'll unpack the secret: why are so few parameters enough?" },
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: LoRA is one of the most influential efficiency techniques of recent years. Its core insight is surprisingly simple: the weight changes during fine-tuning (ΔW) are actually low-rank — you don't need to update all parameters, just adjust within a low-dimensional subspace. It's like moving into a new apartment: you don't need to renovate the whole building, just repaint a few walls." },
            { type: "paragraph", text: "When we take a pretrained LLM (e.g., LLaMA-7B) and want it to learn a new task, the most intuitive approach is full fine-tuning: unfreeze all parameters and continue training on task data. But the cost is prohibitive." },
            { type: "heading", level: 3, text: "Parameter Count Breakdown" },
            { type: "paragraph", text: "Consider a 7B-parameter model. Full fine-tuning requires storing: (1) model weights (~14 GB in FP16), (2) gradients (another 14 GB), (3) Adam optimizer state (first and second moments — another 28 GB). That's ~56 GB of GPU memory before accounting for activations." },
            { type: "table", headers: ["Item", "FP16 Size (7B model)", "Notes"], rows: [
              ["Model weights", "14 GB", "7B × 2 bytes"],
              ["Gradients", "14 GB", "one per parameter"],
              ["Adam m (1st moment)", "14 GB", "EMA of gradients"],
              ["Adam v (2nd moment)", "14 GB", "EMA of squared gradients"],
              ["Total", "~56 GB", "plus activation memory"],
            ]},
            { type: "callout", variant: "warning", text: "Full fine-tuning a 7B model requires at least one 80 GB A100 GPU. For most researchers and developers, this simply isn't feasible. We need a smarter approach." },
            { type: "paragraph", text: "Key observation: the weight update ΔW during fine-tuning is typically low-rank. Aghajanyan et al. (2021) found that pretrained models have very low intrinsic dimensionality — restricting updates to a small subspace achieves results close to full fine-tuning. This is the theoretical foundation of LoRA." },
          ],
        },
        {
          title: "The Intuition Behind Low-Rank Decomposition",
          blocks: [
            { type: "paragraph", text: "The core idea: a large matrix can be approximated as the product of two smaller matrices. If the original weight matrix W is d×d (e.g., 4096×4096), the full update ΔW has d² parameters. But if ΔW has rank r (much smaller than d), we can decompose it as B×A, where B is d×r and A is r×d — reducing parameters from d² to 2dr." },
            { type: "diagram", content: "Low-Rank Decomposition:\n\nFull update ΔW:  d×d = d² params\n┌─────────────────────┐\n│                     │\n│     d × d matrix    │  e.g. 4096 × 4096 = 16,777,216 params\n│   (full rank)       │\n│                     │\n└─────────────────────┘\n\nLow-rank approx B × A:  2 × d × r params\n┌───┐   ┌─────────────────────┐\n│   │   │                     │\n│ B │ × │         A           │  r = 8:\n│d×r│   │       r × d         │  4096×8 + 8×4096 = 65,536 params\n│   │   │                     │  256× compression\n└───┘   └─────────────────────┘" },
            { type: "callout", variant: "info", text: "Rank r is LoRA's core hyperparameter. Larger r means more expressive power but more parameters. In practice, r = 4~16 achieves results close to full fine-tuning." },
          ],
        },
        {
          title: "The LoRA Formula and LoRALinear Implementation",
          blocks: [
            { type: "paragraph", text: "LoRA's formula is elegantly simple:" },
            { type: "diagram", content: "W' = W + (α/r) · B @ A\n\nWhere:\n  W:  original pretrained weight (frozen, requires_grad=False)\n  A:  low-rank matrix, shape (in_features, rank), Kaiming uniform init\n  B:  low-rank matrix, shape (rank, out_features), initialized to zero\n  α:  scaling factor (hyperparameter)\n  r:  rank (hyperparameter)\n\nKey design: B init to zero → BA = 0 at start → model behavior unchanged" },
            { type: "callout", variant: "tip", text: "💡 Instructor's Note: The α/r scaling factor is easy to overlook but critically important. α acts as a 'learning rate multiplier' for LoRA — larger α means bigger LoRA influence. In practice, set α = r or α = 2r. A common mistake is tuning only r while forgetting α, causing performance to drop when r increases (because α/r shrinks)." },
            { type: "heading", level: 3, text: "LoRA Layer Architecture" },
            { type: "diagram", content: "                    LoRALinear Layer\n    ┌──────────────────────────────────────┐\n    │                                      │\n    │  ┌──────────────────────┐             │\n    │  │   W (frozen)         │  ──┐        │\n    │  │  (out × in)          │    │        │\n    │  └──────────────────────┘    │        │\nx ──┤                              ├── + ── ├──→ output\n    │  ┌─────┐    ┌──────────┐    │        │\n    │  │  A  │ →  │    B     │ ───┘        │\n    │  │in×r │    │  r×out   │  × (α/r)    │\n    │  └─────┘    └──────────┘             │\n    │   trainable   trainable              │\n    └──────────────────────────────────────┘" },
            { type: "heading", level: 3, text: "__init__ Implementation" },
            { type: "code", language: "python", code: "class LoRALinear(nn.Module):\n    def __init__(self, in_features, out_features, rank=4, alpha=1.0, bias=True):\n        super().__init__()\n        self.weight = nn.Parameter(\n            torch.empty(out_features, in_features), requires_grad=False\n        )\n        if bias:\n            self.bias = nn.Parameter(\n                torch.zeros(out_features), requires_grad=False\n            )\n        else:\n            self.bias = None\n        self.lora_A = nn.Parameter(torch.empty(in_features, rank))\n        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))\n        # B=0 ensures LoRA contributes nothing at training start\n        self.scaling = alpha / rank" },
            { type: "heading", level: 3, text: "forward Implementation" },
            { type: "code", language: "python", code: "def forward(self, x):\n    base_out = F.linear(x, self.weight, self.bias)\n    lora_out = (x @ self.lora_A @ self.lora_B.T) * self.scaling\n    return base_out + lora_out" },
            { type: "callout", variant: "tip", text: "F.linear(x, W, bias) computes x @ W^T + bias, where W has shape (out, in) — the PyTorch convention. In the LoRA path, A is (in, r) and B is (r, out), so x @ A @ B^T produces the correct output shape." },
          ],
        },
        {
          title: "Applying LoRA to a Model & Merging Weights",
          blocks: [
            { type: "paragraph", text: "With LoRALinear ready, we inject it into an existing Transformer. The standard approach is to replace the Q and V projection layers (q_proj, v_proj) in attention, since these are most critical for task adaptation." },
            { type: "heading", level: 3, text: "apply_lora_to_model" },
            { type: "code", language: "python", code: "def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):\n    if target_modules is None:\n        target_modules = ['q_proj', 'v_proj']\n    for param in model.parameters():\n        param.requires_grad = False\n    replacements = []\n    for name, module in model.named_modules():\n        if isinstance(module, nn.Linear):\n            if any(name.endswith(t) for t in target_modules):\n                replacements.append((name, module))\n    for name, module in replacements:\n        parent_name = '.'.join(name.split('.')[:-1])\n        attr_name = name.split('.')[-1]\n        parent = dict(model.named_modules())[parent_name]\n        setattr(parent, attr_name, LoRALinear.from_linear(module, rank, alpha))\n    return model" },
            { type: "heading", level: 3, text: "Merging LoRA Weights" },
            { type: "paragraph", text: "After training, we can merge LoRA's low-rank updates back into the original weights. The merged model is architecturally identical to the original (standard nn.Linear), with zero inference overhead." },
            { type: "code", language: "python", code: "def merge_lora_weights(model):\n    for name, module in model.named_modules():\n        if isinstance(module, LoRALinear):\n            merged = module.weight + module.scaling * (\n                module.lora_B.T @ module.lora_A.T\n            )\n            new_linear = nn.Linear(\n                module.weight.shape[1], module.weight.shape[0],\n                bias=module.bias is not None\n            )\n            new_linear.weight.data = merged\n            if module.bias is not None:\n                new_linear.bias.data = module.bias.data\n    return model" },
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: Another elegant aspect of LoRA is zero inference overhead. During training BA is a separate matrix, but before deployment you merge it back: W' = W + (α/r)BA. The merged model is identical to the original architecture, with no speed penalty. This means you can train multiple LoRA adapters for different tasks and load them selectively — one base model, many task capabilities." },
          ],
        },
        {
          title: "Parameter Efficiency Analysis",
          blocks: [
            { type: "paragraph", text: "LoRA's parameter efficiency is remarkable. Let's look at concrete numbers:" },
            { type: "table", headers: ["Method", "Trainable Params", "Memory", "Performance", "Inference Overhead"], rows: [
              ["Full Fine-Tuning", "100%", "~4× model size", "Best (but may overfit)", "None"],
              ["LoRA (r=8)", "~0.1-1%", "model size + small delta", "Close to Full FT", "None after merge"],
              ["QLoRA", "~0.1-1%", "~1/4 model size + small", "Close to LoRA", "Dequantization needed"],
            ]},
            { type: "heading", level: 3, text: "Worked Example" },
            { type: "paragraph", text: "For a Transformer layer with d_model = 4096, attention has q_proj and v_proj — both Linear(4096, 4096). Original: 2 × 4096² = 33,554,432 params. With rank=8 LoRA: 2 × (4096×8 + 8×4096) = 131,072 params. ~256× compression." },
            { type: "code", language: "python", code: "def count_trainable_params(model):\n    total = sum(p.numel() for p in model.parameters())\n    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    print(f'Total params:     {total:,}')\n    print(f'Trainable params: {trainable:,} ({100*trainable/total:.2f}%)')\n    return trainable, total" },
            { type: "callout", variant: "tip", text: "QLoRA goes further: it quantizes frozen weights to 4-bit (NF4 format), dramatically reducing memory. Combined with paged optimizers, you can fine-tune a 65B model on a single consumer-grade 24 GB GPU." },
            { type: "callout", variant: "quote", text: "🤔 Think About It: LoRA injects low-rank matrices into Linear layers (typically W_q, W_k, W_v, W_o in attention). But FFN layers account for more than half a model's parameters — why don't we typically apply LoRA to FFN layers? Would it help? What does this reveal about what Attention vs. FFN layers learn?" },
          ],
        },
      ],
      exercises: [
        { id: "lora_linear_init", title: "TODO 1: LoRALinear.__init__", description: "Initialize the LoRALinear module: frozen weight, trainable lora_A (Kaiming uniform), trainable lora_B (zero init), and scaling = alpha / rank.", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["Use nn.Parameter(..., requires_grad=False) for the frozen weight", "lora_A shape: (in_features, rank), init with nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))", "lora_B shape: (rank, out_features), init to torch.zeros", "Zero-init B ensures LoRA contributes nothing at the start of training"], pseudocode: "self.weight = Parameter(empty(out, in), requires_grad=False)\nself.lora_A = Parameter(empty(in, rank))\nkaiming_uniform_(self.lora_A)\nself.lora_B = Parameter(zeros(rank, out))\nself.scaling = alpha / rank" },
        { id: "lora_linear_forward", title: "TODO 2: LoRALinear.forward", description: "Implement LoRA's forward pass: frozen path (F.linear) + trainable low-rank bypass (x @ A @ B^T * scaling).", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["base output: F.linear(x, self.weight, self.bias)", "LoRA path: (x @ self.lora_A @ self.lora_B.T) * self.scaling", "Return the sum of both"], pseudocode: "base = F.linear(x, self.weight, self.bias)\nlora = (x @ self.lora_A @ self.lora_B.T) * self.scaling\nreturn base + lora" },
        { id: "apply_lora_to_model", title: "TODO 3: apply_lora_to_model", description: "Replace specified nn.Linear layers in the model with LoRALinear. Freeze all params first, then replace target layers.", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["Freeze all: for param in model.parameters(): param.requires_grad = False", "Use model.named_modules() to find target nn.Linear layers", "Use setattr(parent, attr_name, new_module) to replace"], pseudocode: "freeze all params\nfor name, module in model.named_modules():\n  if Linear and name ends with target:\n    setattr(parent, attr, LoRALinear.from_linear(module, rank, alpha))" },
        { id: "merge_lora_weights", title: "TODO 4: merge_lora_weights", description: "Merge LoRA weights back into the original weight matrix, producing a standard nn.Linear for efficient inference.", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["merged_weight = lora.weight + lora.scaling * (lora.lora_B.T @ lora.lora_A.T)", "Create a new nn.Linear and copy the merged weight and bias"], pseudocode: "for LoRALinear modules:\n  W_merged = weight + scaling * (B^T @ A^T)\n  replace with nn.Linear" },
        { id: "count_trainable_params", title: "TODO 5: count_trainable_params", description: "Count the model's trainable and total parameters and print a summary.", labFile: "labs/phase7_lora/phase_7/lora.py", hints: ["total = sum(p.numel() for p in model.parameters())", "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)"], pseudocode: "total = sum numel for all params\ntrainable = sum numel for requires_grad\nreturn (trainable, total)" },
      ],
      acceptanceCriteria: [
        "After initialization, LoRALinear's B is a zero matrix and forward output matches the original nn.Linear",
        "After apply_lora_to_model, only lora_A and lora_B have requires_grad=True",
        "After merge_lora_weights, no LoRALinear modules remain and outputs match pre-merge",
        "count_trainable_params correctly reports trainable and frozen parameter counts",
      ],
      references: [
        { title: "LoRA: Low-Rank Adaptation of Large Language Models", description: "Hu et al. 2021", url: "https://arxiv.org/abs/2106.09685" },
        { title: "QLoRA: Efficient Finetuning of Quantized LLMs", description: "Dettmers et al. 2023", url: "https://arxiv.org/abs/2305.14314" },
        { title: "HuggingFace PEFT Library", description: "Official implementation of LoRA, QLoRA, and other PEFT methods", url: "https://github.com/huggingface/peft" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 8 & 9 placeholders — to be filled in next
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// Phase 8: Human Alignment — zh-TW
// ─────────────────────────────────────────────────────────────

const phase8ContentZhTW: PhaseContent = {
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
            { type: "paragraph", text: "預訓練的語言模型是強大的「文本補全器」——給它一段開頭，它能續寫出流暢的文本。但它不是一個「助理」。Supervised Fine-Tuning (SFT) 的目標是教模型理解「指令→回應」的格式。" },
            { type: "callout", variant: "info", text: "InstructGPT (Ouyang et al., 2022) 的三階段訓練流程：(1) SFT — 用人工撰寫的示範來微調，(2) Reward Model — 訓練一個偏好模型，(3) PPO — 用 RL 進一步對齊。DPO 將後兩步合併為一步。" },
          ],
        },
        {
          title: "Instruction Format & Label Masking",
          blocks: [
            { type: "paragraph", text: "SFT 使用結構化的 prompt 模板。Alpaca 格式是最常見的之一：" },
            { type: "code", language: "text", code: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" },
            { type: "heading", level: 3, text: "Label Masking: 只在 Response 上計算 Loss" },
            { type: "paragraph", text: "關鍵技巧：我們對 prompt 部分的 token 設定 label = -100（PyTorch 的 cross_entropy 會自動忽略 -100），只在 response 部分計算 loss。" },
            { type: "diagram", content: "Label Masking 示意圖：\n\n Token:  [### Inst: 什麼是AI？ ### Response: AI是人工智慧...]\n          ├──── prompt tokens ────┤├── response tokens ──┤\n Labels: [-100 -100 -100 ... -100  AI  是  人工  智慧  ...]\n\n只有 response tokens 會貢獻到 cross-entropy loss！" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Label masking 是 SFT 中最容易出錯的地方。我們只想讓模型學會「如何回答」，不想讓它學會「如何提問」。如果忘了做 label masking，模型會花一半的容量去「背」instruction 的措辭，浪費訓練資源且效果變差。" },
            { type: "heading", level: 3, text: "InstructionDataset 實作" },
            { type: "code", language: "python", code: "class InstructionDataset(Dataset):\n    def __init__(self, jsonl_path, tokenizer, max_length=512):\n        self.samples = []\n        with open(jsonl_path) as f:\n            for line in f:\n                data = json.loads(line)\n                prompt = format_template(data)\n                full_text = prompt + data['output']\n                prompt_ids = tokenizer.encode(prompt)\n                full_ids = tokenizer.encode(full_text)\n                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]\n                # pad/truncate to max_length\n                self.samples.append({'input_ids': ..., 'labels': ...})" },
            { type: "callout", variant: "warning", text: "Tokenization 的陷阱：tokenizer.encode(prompt + response) 的結果可能不等於 tokenizer.encode(prompt) + tokenizer.encode(response)，因為 BPE merge 可能跨越邊界。最安全的做法是先 tokenize 整個文本，再用 prompt-only 的 token 長度來決定 mask 邊界。" },
          ],
        },
        {
          title: "SFT Training Loop",
          blocks: [
            { type: "paragraph", text: "SFT 的訓練迴圈與標準語言模型訓練幾乎相同，唯一的差別是 label masking。Loss 函數使用 cross_entropy 並設定 ignore_index=-100。" },
            { type: "code", language: "python", code: "def train_sft(model, train_loader, val_loader, config):\n    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)\n    for epoch in range(config.n_epochs):\n        for batch in train_loader:\n            logits = model(batch['input_ids'].to(config.device))\n            logits = logits[:, :-1, :].contiguous()\n            labels = batch['labels'][:, 1:].contiguous().to(config.device)\n            loss = F.cross_entropy(\n                logits.view(-1, logits.size(-1)),\n                labels.view(-1),\n                ignore_index=-100\n            )\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()" },
            { type: "callout", variant: "tip", text: "注意 shift 的方向：logits[:, :-1] 預測 labels[:, 1:]。這是因為位置 i 的 logits 預測的是位置 i+1 的 token。忘記 shift 是最常見的 bug 之一。" },
          ],
        },
        {
          title: "Direct Preference Optimization (DPO)",
          blocks: [
            { type: "paragraph", text: "SFT 教模型「如何回答」，但不教它「哪種回答更好」。DPO 提供了一條比 RLHF 更簡單的路——直接從偏好數據中優化 policy，跳過 reward model 的訓練。" },
            { type: "table", headers: ["特性", "RLHF (PPO)", "DPO"], rows: [
              ["訓練階段", "SFT → RM → PPO（三階段）", "SFT → DPO（兩階段）"],
              ["需要 Reward Model?", "是", "否"],
              ["訓練穩定性", "低（RL 固有問題）", "高（supervised loss）"],
              ["實作複雜度", "高", "低（一個 loss 函數）"],
              ["效果", "優秀", "comparable 甚至更好"],
            ]},
          ],
        },
        {
          title: "DPO Loss 公式與實作",
          blocks: [
            { type: "diagram", content: "DPO Loss:\n\n  L_DPO = -log σ( β × [ log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ] )\n\n  y_w: chosen response  |  y_l: rejected response\n  π:   policy model     |  π_ref: frozen reference model\n  β:   temperature (controls deviation from reference)" },
            { type: "paragraph", text: "直覺解讀：DPO loss 鼓勵 policy model（相對於 reference model）增加 chosen response 的概率、降低 rejected response 的概率。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：DPO 最大的貢獻是證明了你不需要一個單獨的 reward model。傳統的 RLHF 流程是三步；DPO 把它壓縮成一步——直接從偏好數據中優化模型。數學上它是等價的，但實作上簡單得多、穩定得多。這就是為什麼 DPO 正在取代 RLHF 成為主流。" },
            { type: "code", language: "python", code: "def get_log_probs(model, input_ids, attention_mask, prompt_lengths):\n    logits = model(input_ids)\n    logits = logits[:, :-1, :]\n    labels = input_ids[:, 1:]\n    log_probs = logits.log_softmax(dim=-1)\n    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)\n    mask = torch.arange(labels.size(1), device=labels.device)\n    mask = mask.unsqueeze(0) >= (prompt_lengths.unsqueeze(1) - 1)\n    mask = mask & attention_mask[:, 1:].bool()\n    return (token_log_probs * mask).sum(dim=-1)" },
            { type: "code", language: "python", code: "def dpo_loss(policy_chosen_lp, policy_rejected_lp,\n             ref_chosen_lp, ref_rejected_lp, beta=0.1):\n    log_ratio_chosen = policy_chosen_lp - ref_chosen_lp\n    log_ratio_rejected = policy_rejected_lp - ref_rejected_lp\n    return -F.logsigmoid(\n        beta * (log_ratio_chosen - log_ratio_rejected)\n    ).mean()" },
            { type: "callout", variant: "tip", text: "使用 F.logsigmoid 而非 log(sigmoid(...)) 可以避免數值不穩定。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：DPO 需要「偏好對」數據——對同一個 prompt，一個好的回答和一個差的回答。但在實務中，收集這種配對數據很昂貴。如果我們只有「好的回答」（沒有對應的「差的回答」），還能用 DPO 嗎？有沒有什麼技巧可以自動生成「差的回答」？（提示：想想 rejection sampling 和 self-play。）" },
          ],
        },
      ],
      exercises: [
        { id: "instruction_dataset_init", title: "TODO 1: InstructionDataset.__init__", description: "從 JSONL 載入指令資料，用 Alpaca 模板格式化，tokenize 後建立 label-masked 的訓練樣本。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["根據 input 是否為空選擇不同模板", "先 tokenize 純 prompt，記錄其長度作為 mask 邊界", "labels 中 prompt 部分設為 -100"], pseudocode: "for line in jsonl:\n  prompt = format_template(instruction, input)\n  prompt_ids = tokenize(prompt)\n  full_ids = tokenize(prompt + output)\n  labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]" },
        { id: "format_prompt", title: "TODO 2: __len__ & __getitem__", description: "實作 Dataset 標準方法：__len__ 返回樣本數，__getitem__ 返回包含 input_ids、attention_mask、labels 的 dict。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["__len__ 直接返回 self.samples 的長度", "__getitem__ 返回 dict，包含 torch.Tensor 格式的各欄位"], pseudocode: "def __len__: return len(self.samples)\ndef __getitem__(idx): return dict of tensors" },
        { id: "train_sft", title: "TODO 3: train_sft", description: "實作 SFT 訓練迴圈：forward pass、shift logits/labels、masked cross-entropy loss、backward。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["記得 shift：logits[:, :-1] 對應 labels[:, 1:]", "使用 F.cross_entropy(..., ignore_index=-100)"], pseudocode: "for epoch:\n  for batch:\n    logits = model(input_ids)\n    shift and compute CE loss with ignore=-100\n    loss.backward(); step; zero_grad" },
        { id: "preference_dataset", title: "TODO 4: PreferenceDataset", description: "載入偏好對資料（prompt, chosen, rejected），tokenize 並建立 DPO 訓練所需的資料結構。", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["每個樣本需要 tokenize 兩個文本：prompt+chosen 和 prompt+rejected", "記錄 prompt_length 供 get_log_probs 做 masking"], pseudocode: "for line in jsonl:\n  chosen_ids = tokenize(prompt + chosen)\n  rejected_ids = tokenize(prompt + rejected)\n  prompt_len = len(tokenize(prompt))" },
        { id: "dpo_loss", title: "TODO 5: dpo_loss", description: "實作 DPO loss：計算 log ratio chosen/rejected，套用 β scaling，取 -logsigmoid 的 mean。", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["log_ratio_chosen = policy_chosen_lp - ref_chosen_lp", "用 F.logsigmoid 確保數值穩定"], pseudocode: "log_ratio_w = policy_chosen - ref_chosen\nlog_ratio_l = policy_rejected - ref_rejected\nreturn -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()" },
      ],
      acceptanceCriteria: [
        "InstructionDataset 正確載入 JSONL 並套用 Alpaca 模板",
        "Label masking 確保 loss 只在 response tokens 上計算",
        "train_sft 正確 shift logits 和 labels，使用 ignore_index=-100",
        "dpo_loss 的梯度方向正確：增加 chosen 的 log ratio，減少 rejected 的",
        "Reference model 在 DPO 訓練中始終保持 frozen",
      ],
      references: [
        { title: "Direct Preference Optimization", description: "Rafailov et al. 2023 — DPO 原始論文", url: "https://arxiv.org/abs/2305.18290" },
        { title: "Training language models to follow instructions with human feedback", description: "Ouyang et al. 2022 — InstructGPT / RLHF", url: "https://arxiv.org/abs/2203.02155" },
        { title: "Stanford Alpaca", description: "指令微調的經典開源實作", url: "https://github.com/tatsu-lab/stanford_alpaca" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 8: Human Alignment — zh-CN
// ─────────────────────────────────────────────────────────────

const phase8ContentZhCN: PhaseContent = {
  phaseId: 8,
  color: "#EF4444",
  accent: "#F87171",
  lessons: [
    {
      phaseId: 8, lessonId: 1,
      title: "SFT & DPO",
      subtitle: "从语言模型到 AI 助手——指令微调与人类偏好对齐",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 Supervised Fine-Tuning (SFT) 的指令格式与 label masking 技术",
        "实现 InstructionDataset，正确处理 Alpaca 模板与 prompt/response 分割",
        "理解 RLHF 与 DPO 的差异，以及 DPO 如何避免训练 reward model",
        "掌握 DPO loss 的数学推导与实现",
        "实现 PreferenceDataset 与 dpo_loss 函数",
      ],
      sections: [
        {
          title: "从语言模型到助手：为什么需要 SFT？",
          blocks: [
            { type: "paragraph", text: "你有没有用过那种「野生」的 base model（比如直接下载 LLaMA-2 base）？你会发现它很奇怪——你问它「北京的天气如何？」，它不会回答你，而是接着写「上海的天气如何？广州的天气如何？」——它以为你在写一个城市列表！这就是 base model 的问题：它学会了语言，但不知道自己该扮演什么角色。SFT 教它「你是一个助手，要回答问题」，DPO 教它「哪种回答是人类喜欢的」。今天我们来搞定这两步。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：如果说 Pre-training 教会了模型「语言能力」，SFT 和 DPO 就是教会模型「社会规范」。预训练后的模型像一个博学但没有教养的人——它知道很多，但不知道怎么恰当地回答问题。SFT 教它格式（用 instruction-response 的格式回答），DPO 教它质量（哪种回答更好）。这两步合起来，就是所谓的「人类对齐」（alignment）。" },
            { type: "paragraph", text: "预训练的语言模型是强大的「文本补全器」——给它一段开头，它能续写出流畅的文本。但它不是一个「助手」。Supervised Fine-Tuning (SFT) 的目标是教模型理解「指令→回应」的格式。" },
            { type: "callout", variant: "info", text: "InstructGPT (Ouyang et al., 2022) 的三阶段训练流程：(1) SFT — 用人工撰写的示范来微调，(2) Reward Model — 训练一个偏好模型，(3) PPO — 用 RL 进一步对齐。DPO 将后两步合并为一步。" },
          ],
        },
        {
          title: "指令格式与 Label Masking",
          blocks: [
            { type: "paragraph", text: "SFT 使用结构化的 prompt 模板。Alpaca 格式是最常见的之一：" },
            { type: "code", language: "text", code: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" },
            { type: "heading", level: 3, text: "Label Masking：只在 Response 上计算 Loss" },
            { type: "paragraph", text: "关键技巧：我们对 prompt 部分的 token 设定 label = -100（PyTorch 的 cross_entropy 会自动忽略 -100），只在 response 部分计算 loss。" },
            { type: "diagram", content: "Label Masking 示意图：\n\n Token:  [### Inst: 什么是AI？ ### Response: AI是人工智能...]\n          ├──── prompt tokens ────┤├── response tokens ──┤\n Labels: [-100 -100 -100 ... -100  AI  是  人工  智能  ...]\n\n只有 response tokens 会贡献到 cross-entropy loss！" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：Label masking 是 SFT 中最容易出错的地方。我们只想让模型学会「如何回答」，不想让它学会「如何提问」。如果忘了做 label masking，模型会花一半的容量去「背」instruction 的措辞，浪费训练资源且效果变差。" },
            { type: "heading", level: 3, text: "InstructionDataset 实现" },
            { type: "code", language: "python", code: "class InstructionDataset(Dataset):\n    def __init__(self, jsonl_path, tokenizer, max_length=512):\n        self.samples = []\n        with open(jsonl_path) as f:\n            for line in f:\n                data = json.loads(line)\n                prompt = format_template(data)\n                full_text = prompt + data['output']\n                prompt_ids = tokenizer.encode(prompt)\n                full_ids = tokenizer.encode(full_text)\n                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]\n                # pad/truncate to max_length\n                self.samples.append({'input_ids': ..., 'labels': ...})" },
            { type: "callout", variant: "warning", text: "Tokenization 的陷阱：tokenizer.encode(prompt + response) 的结果可能不等于 tokenizer.encode(prompt) + tokenizer.encode(response)，因为 BPE merge 可能跨越边界。最安全的做法是先 tokenize 整个文本，再用 prompt-only 的 token 长度来决定 mask 边界。" },
          ],
        },
        {
          title: "SFT 训练循环",
          blocks: [
            { type: "paragraph", text: "SFT 的训练循环与标准语言模型训练几乎相同，唯一的差别是 label masking。Loss 函数使用 cross_entropy 并设定 ignore_index=-100。" },
            { type: "code", language: "python", code: "def train_sft(model, train_loader, val_loader, config):\n    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)\n    for epoch in range(config.n_epochs):\n        for batch in train_loader:\n            logits = model(batch['input_ids'].to(config.device))\n            logits = logits[:, :-1, :].contiguous()\n            labels = batch['labels'][:, 1:].contiguous().to(config.device)\n            loss = F.cross_entropy(\n                logits.view(-1, logits.size(-1)),\n                labels.view(-1),\n                ignore_index=-100\n            )\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()" },
            { type: "callout", variant: "tip", text: "注意 shift 的方向：logits[:, :-1] 预测 labels[:, 1:]。忘记 shift 是最常见的 bug 之一。" },
          ],
        },
        {
          title: "Direct Preference Optimization (DPO)",
          blocks: [
            { type: "paragraph", text: "SFT 教模型「如何回答」，但不教它「哪种回答更好」。DPO 提供了一条比 RLHF 更简单的路——直接从偏好数据中优化 policy，跳过 reward model 的训练。" },
            { type: "table", headers: ["特性", "RLHF (PPO)", "DPO"], rows: [
              ["训练阶段", "SFT → RM → PPO（三阶段）", "SFT → DPO（两阶段）"],
              ["需要 Reward Model?", "是", "否"],
              ["训练稳定性", "低（RL 固有问题）", "高（supervised loss）"],
              ["实现复杂度", "高", "低（一个 loss 函数）"],
              ["效果", "优秀", "comparable 甚至更好"],
            ]},
          ],
        },
        {
          title: "DPO Loss 公式与实现",
          blocks: [
            { type: "diagram", content: "DPO Loss:\n\n  L_DPO = -log σ( β × [ log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ] )\n\n  y_w: chosen response  |  y_l: rejected response\n  π:   policy model     |  π_ref: frozen reference model\n  β:   temperature (controls deviation from reference)" },
            { type: "paragraph", text: "直觉解读：DPO loss 鼓励 policy model（相对于 reference model）增加 chosen response 的概率、降低 rejected response 的概率。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：DPO 最大的贡献是证明了你不需要一个单独的 reward model。传统的 RLHF 流程是三步；DPO 把它压缩成一步——直接从偏好数据中优化模型。数学上它是等价的，但实现上简单得多、稳定得多。这就是为什么 DPO 正在取代 RLHF 成为主流。" },
            { type: "code", language: "python", code: "def get_log_probs(model, input_ids, attention_mask, prompt_lengths):\n    logits = model(input_ids)\n    logits = logits[:, :-1, :]\n    labels = input_ids[:, 1:]\n    log_probs = logits.log_softmax(dim=-1)\n    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)\n    mask = torch.arange(labels.size(1), device=labels.device)\n    mask = mask.unsqueeze(0) >= (prompt_lengths.unsqueeze(1) - 1)\n    mask = mask & attention_mask[:, 1:].bool()\n    return (token_log_probs * mask).sum(dim=-1)" },
            { type: "code", language: "python", code: "def dpo_loss(policy_chosen_lp, policy_rejected_lp,\n             ref_chosen_lp, ref_rejected_lp, beta=0.1):\n    log_ratio_chosen = policy_chosen_lp - ref_chosen_lp\n    log_ratio_rejected = policy_rejected_lp - ref_rejected_lp\n    return -F.logsigmoid(\n        beta * (log_ratio_chosen - log_ratio_rejected)\n    ).mean()" },
            { type: "callout", variant: "tip", text: "使用 F.logsigmoid 而非 log(sigmoid(...)) 可以避免数值不稳定。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：DPO 需要「偏好对」数据——对同一个 prompt，一个好的回答和一个差的回答。但在实际中，收集这种配对数据很昂贵。如果我们只有「好的回答」，还能用 DPO 吗？有没有什么技巧可以自动生成「差的回答」？（提示：想想 rejection sampling 和 self-play。）" },
          ],
        },
      ],
      exercises: [
        { id: "instruction_dataset_init", title: "TODO 1: InstructionDataset.__init__", description: "从 JSONL 加载指令数据，用 Alpaca 模板格式化，tokenize 后建立 label-masked 的训练样本。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["根据 input 是否为空选择不同模板", "先 tokenize 纯 prompt，记录其长度作为 mask 边界", "labels 中 prompt 部分设为 -100"], pseudocode: "for line in jsonl:\n  prompt = format_template(instruction, input)\n  prompt_ids = tokenize(prompt)\n  full_ids = tokenize(prompt + output)\n  labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]" },
        { id: "format_prompt", title: "TODO 2: __len__ & __getitem__", description: "实现 Dataset 标准方法：__len__ 返回样本数，__getitem__ 返回包含 input_ids、attention_mask、labels 的 dict。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["__len__ 直接返回 self.samples 的长度", "__getitem__ 返回 dict，包含 torch.Tensor 格式的各字段"], pseudocode: "def __len__: return len(self.samples)\ndef __getitem__(idx): return dict of tensors" },
        { id: "train_sft", title: "TODO 3: train_sft", description: "实现 SFT 训练循环：forward pass、shift logits/labels、masked cross-entropy loss、backward。", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["记得 shift：logits[:, :-1] 对应 labels[:, 1:]", "使用 F.cross_entropy(..., ignore_index=-100)"], pseudocode: "for epoch:\n  for batch:\n    logits = model(input_ids)\n    shift and compute CE loss with ignore=-100\n    loss.backward(); step; zero_grad" },
        { id: "preference_dataset", title: "TODO 4: PreferenceDataset", description: "加载偏好对数据（prompt, chosen, rejected），tokenize 并建立 DPO 训练所需的数据结构。", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["每个样本需要 tokenize 两个文本：prompt+chosen 和 prompt+rejected", "记录 prompt_length 供 get_log_probs 做 masking"], pseudocode: "for line in jsonl:\n  chosen_ids = tokenize(prompt + chosen)\n  rejected_ids = tokenize(prompt + rejected)\n  prompt_len = len(tokenize(prompt))" },
        { id: "dpo_loss", title: "TODO 5: dpo_loss", description: "实现 DPO loss：计算 log ratio chosen/rejected，应用 β scaling，取 -logsigmoid 的 mean。", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["log_ratio_chosen = policy_chosen_lp - ref_chosen_lp", "用 F.logsigmoid 确保数值稳定"], pseudocode: "log_ratio_w = policy_chosen - ref_chosen\nlog_ratio_l = policy_rejected - ref_rejected\nreturn -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()" },
      ],
      acceptanceCriteria: [
        "InstructionDataset 正确加载 JSONL 并应用 Alpaca 模板",
        "Label masking 确保 loss 只在 response tokens 上计算",
        "train_sft 正确 shift logits 和 labels，使用 ignore_index=-100",
        "dpo_loss 的梯度方向正确：增加 chosen 的 log ratio，减少 rejected 的",
        "Reference model 在 DPO 训练中始终保持 frozen",
      ],
      references: [
        { title: "Direct Preference Optimization", description: "Rafailov et al. 2023 — DPO 原始论文", url: "https://arxiv.org/abs/2305.18290" },
        { title: "Training language models to follow instructions with human feedback", description: "Ouyang et al. 2022 — InstructGPT / RLHF", url: "https://arxiv.org/abs/2203.02155" },
        { title: "Stanford Alpaca", description: "指令微调的经典开源实现", url: "https://github.com/tatsu-lab/stanford_alpaca" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 8: Human Alignment — en
// ─────────────────────────────────────────────────────────────

const phase8ContentEn: PhaseContent = {
  phaseId: 8,
  color: "#EF4444",
  accent: "#F87171",
  lessons: [
    {
      phaseId: 8, lessonId: 1,
      title: "SFT & DPO",
      subtitle: "From language model to AI assistant — instruction tuning and human alignment",
      type: "concept",
      duration: "75 min",
      objectives: [
        "Understand SFT's instruction format and label masking technique",
        "Implement InstructionDataset with correct Alpaca template handling and prompt/response splitting",
        "Understand the difference between RLHF and DPO, and why DPO skips training a reward model",
        "Master the math behind DPO loss and implement it",
        "Implement PreferenceDataset and dpo_loss",
      ],
      sections: [
        {
          title: "From Language Model to Assistant: Why SFT?",
          blocks: [
            { type: "paragraph", text: "Have you ever tried a 'wild' base model straight out of the box? It behaves strangely — ask it 'What's the weather in New York?' and instead of answering, it continues writing 'What's the weather in Los Angeles? What's the weather in Chicago?' — it thinks you're composing a list! That's the core problem with base models: they've learned language, but don't know what role to play. SFT teaches it 'you're an assistant, answer questions.' DPO teaches it 'here's what humans prefer.' Today we tackle both." },
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: If pre-training gives the model 'language ability,' SFT and DPO give it 'social norms.' A post-pretrained model is like a brilliant but unrefined person — full of knowledge, but uncertain how to respond appropriately. SFT teaches format (respond in instruction-response style). DPO teaches quality (which response is better). Together, these two steps are what we call human alignment." },
            { type: "paragraph", text: "A pretrained language model is a powerful text completer — give it a prefix and it fluently continues. But it's not an assistant. Supervised Fine-Tuning (SFT) teaches the model the instruction→response format by training on large quantities of (instruction, response) pairs." },
            { type: "callout", variant: "info", text: "InstructGPT (Ouyang et al., 2022) three-stage pipeline: (1) SFT — fine-tune with human demonstrations, (2) Reward Model — train a preference model, (3) PPO — further alignment via RL. DPO collapses steps 2 and 3 into one." },
          ],
        },
        {
          title: "Instruction Format & Label Masking",
          blocks: [
            { type: "paragraph", text: "SFT uses structured prompt templates. The Alpaca format is one of the most common:" },
            { type: "code", language: "text", code: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" },
            { type: "heading", level: 3, text: "Label Masking: Compute Loss Only on the Response" },
            { type: "paragraph", text: "Key technique: we set label = -100 for all prompt tokens (PyTorch's cross_entropy automatically ignores -100), computing loss only on response tokens." },
            { type: "diagram", content: "Label Masking:\n\n Tokens:  [### Inst: What is AI? ### Response: AI is artificial intelligence...]\n           ├──── prompt tokens ────┤├── response tokens ──────────────┤\n Labels:  [-100  -100  -100  ...  -100   AI   is   artificial   ...]\n\nOnly response tokens contribute to cross-entropy loss!" },
            { type: "callout", variant: "tip", text: "💡 Instructor's Note: Label masking is the most error-prone part of SFT. We only want the model to learn 'how to answer,' not 'how to ask.' Without masking, the model wastes half its capacity memorizing the phrasing of instructions, degrading performance." },
            { type: "heading", level: 3, text: "InstructionDataset Implementation" },
            { type: "code", language: "python", code: "class InstructionDataset(Dataset):\n    def __init__(self, jsonl_path, tokenizer, max_length=512):\n        self.samples = []\n        with open(jsonl_path) as f:\n            for line in f:\n                data = json.loads(line)\n                prompt = format_template(data)\n                full_text = prompt + data['output']\n                prompt_ids = tokenizer.encode(prompt)\n                full_ids = tokenizer.encode(full_text)\n                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]\n                # pad/truncate to max_length\n                self.samples.append({'input_ids': ..., 'labels': ...})" },
            { type: "callout", variant: "warning", text: "Tokenization gotcha: tokenizer.encode(prompt + response) may not equal tokenizer.encode(prompt) + tokenizer.encode(response), because BPE merges can cross the boundary. The safest approach: tokenize the full text, then use the prompt-only token length to determine the masking boundary." },
          ],
        },
        {
          title: "SFT Training Loop",
          blocks: [
            { type: "paragraph", text: "The SFT training loop is almost identical to standard language model training — the only difference is label masking. The loss function uses cross_entropy with ignore_index=-100." },
            { type: "code", language: "python", code: "def train_sft(model, train_loader, val_loader, config):\n    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)\n    for epoch in range(config.n_epochs):\n        for batch in train_loader:\n            logits = model(batch['input_ids'].to(config.device))\n            logits = logits[:, :-1, :].contiguous()\n            labels = batch['labels'][:, 1:].contiguous().to(config.device)\n            loss = F.cross_entropy(\n                logits.view(-1, logits.size(-1)),\n                labels.view(-1),\n                ignore_index=-100\n            )\n            loss.backward()\n            optimizer.step()\n            optimizer.zero_grad()" },
            { type: "callout", variant: "tip", text: "Note the shift direction: logits[:, :-1] predicts labels[:, 1:]. Forgetting to shift is one of the most common bugs." },
          ],
        },
        {
          title: "Direct Preference Optimization (DPO)",
          blocks: [
            { type: "paragraph", text: "SFT teaches the model 'how to answer,' but not 'which answer is better.' DPO offers a simpler path than RLHF — optimizing the policy directly from preference data, skipping reward model training entirely." },
            { type: "table", headers: ["Property", "RLHF (PPO)", "DPO"], rows: [
              ["Training stages", "SFT → RM → PPO (3 stages)", "SFT → DPO (2 stages)"],
              ["Needs Reward Model?", "Yes", "No"],
              ["Training stability", "Low (RL instability)", "High (supervised loss)"],
              ["Implementation complexity", "High", "Low (one loss function)"],
              ["Performance", "Excellent", "Comparable or better"],
            ]},
          ],
        },
        {
          title: "The DPO Loss Formula and Implementation",
          blocks: [
            { type: "diagram", content: "DPO Loss:\n\n  L_DPO = -log σ( β × [ log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ] )\n\n  y_w: chosen (preferred) response\n  y_l: rejected (dispreferred) response\n  π:   policy model (being trained)\n  π_ref: reference model (frozen SFT checkpoint)\n  β:   temperature — controls how far policy can drift from reference" },
            { type: "paragraph", text: "Intuition: the DPO loss encourages the policy (relative to the reference) to increase the probability of chosen responses and decrease the probability of rejected ones." },
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: DPO's biggest contribution is proving you don't need a separate reward model. The traditional RLHF pipeline has three steps; DPO compresses it to one — optimizing directly from preference data. Mathematically equivalent, but dramatically simpler and more stable. That's why DPO is replacing RLHF as the standard approach." },
            { type: "code", language: "python", code: "def get_log_probs(model, input_ids, attention_mask, prompt_lengths):\n    logits = model(input_ids)\n    logits = logits[:, :-1, :]\n    labels = input_ids[:, 1:]\n    log_probs = logits.log_softmax(dim=-1)\n    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)\n    mask = torch.arange(labels.size(1), device=labels.device)\n    mask = mask.unsqueeze(0) >= (prompt_lengths.unsqueeze(1) - 1)\n    mask = mask & attention_mask[:, 1:].bool()\n    return (token_log_probs * mask).sum(dim=-1)" },
            { type: "code", language: "python", code: "def dpo_loss(policy_chosen_lp, policy_rejected_lp,\n             ref_chosen_lp, ref_rejected_lp, beta=0.1):\n    log_ratio_chosen = policy_chosen_lp - ref_chosen_lp\n    log_ratio_rejected = policy_rejected_lp - ref_rejected_lp\n    return -F.logsigmoid(\n        beta * (log_ratio_chosen - log_ratio_rejected)\n    ).mean()" },
            { type: "callout", variant: "tip", text: "Use F.logsigmoid instead of log(sigmoid(...)) to avoid numerical instability when log ratios are extreme." },
            { type: "callout", variant: "quote", text: "🤔 Think About It: DPO requires 'preference pair' data — for the same prompt, one good response and one bad response. But collecting such paired data is expensive. If you only have good responses (no corresponding bad ones), can you still use DPO? Are there tricks to automatically generate bad responses? (Hint: think rejection sampling and self-play.)" },
          ],
        },
      ],
      exercises: [
        { id: "instruction_dataset_init", title: "TODO 1: InstructionDataset.__init__", description: "Load instruction data from JSONL, format with the Alpaca template, tokenize, and create label-masked training samples.", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["Choose template based on whether 'input' is empty", "Tokenize the prompt alone first to determine the masking boundary", "Set prompt tokens to -100 in labels"], pseudocode: "for line in jsonl:\n  prompt = format_template(instruction, input)\n  prompt_ids = tokenize(prompt)\n  full_ids = tokenize(prompt + output)\n  labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]" },
        { id: "format_prompt", title: "TODO 2: __len__ & __getitem__", description: "Implement standard Dataset methods: __len__ returns sample count, __getitem__ returns a dict with input_ids, attention_mask, and labels as tensors.", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["__len__ just returns len(self.samples)", "__getitem__ returns a dict of torch.Tensor"], pseudocode: "def __len__: return len(self.samples)\ndef __getitem__(idx): return dict of tensors" },
        { id: "train_sft", title: "TODO 3: train_sft", description: "Implement the SFT training loop: forward pass, shift logits/labels, masked cross-entropy loss, backward.", labFile: "labs/phase8_instruction_tuning/phase_8/sft.py", hints: ["Shift: logits[:, :-1] corresponds to labels[:, 1:]", "Use F.cross_entropy(..., ignore_index=-100)"], pseudocode: "for epoch:\n  for batch:\n    logits = model(input_ids)\n    shift and compute CE loss with ignore=-100\n    loss.backward(); step; zero_grad" },
        { id: "preference_dataset", title: "TODO 4: PreferenceDataset", description: "Load preference pair data (prompt, chosen, rejected), tokenize, and build the data structure needed for DPO training.", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["Each sample needs two tokenized texts: prompt+chosen and prompt+rejected", "Record prompt_length for masking in get_log_probs"], pseudocode: "for line in jsonl:\n  chosen_ids = tokenize(prompt + chosen)\n  rejected_ids = tokenize(prompt + rejected)\n  prompt_len = len(tokenize(prompt))" },
        { id: "dpo_loss", title: "TODO 5: dpo_loss", description: "Implement DPO loss: compute log ratios for chosen/rejected, apply β scaling, return mean of -logsigmoid.", labFile: "labs/phase8_instruction_tuning/phase_8/dpo.py", hints: ["log_ratio_chosen = policy_chosen_lp - ref_chosen_lp", "Use F.logsigmoid for numerical stability"], pseudocode: "log_ratio_w = policy_chosen - ref_chosen\nlog_ratio_l = policy_rejected - ref_rejected\nreturn -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()" },
      ],
      acceptanceCriteria: [
        "InstructionDataset correctly loads JSONL and applies the Alpaca template",
        "Label masking ensures loss is computed only on response tokens",
        "train_sft correctly shifts logits and labels and uses ignore_index=-100",
        "dpo_loss gradient direction is correct: increases chosen log ratio, decreases rejected",
        "Reference model stays frozen throughout DPO training",
      ],
      references: [
        { title: "Direct Preference Optimization", description: "Rafailov et al. 2023 — original DPO paper", url: "https://arxiv.org/abs/2305.18290" },
        { title: "Training language models to follow instructions with human feedback", description: "Ouyang et al. 2022 — InstructGPT / RLHF", url: "https://arxiv.org/abs/2203.02155" },
        { title: "Stanford Alpaca", description: "Classic open-source instruction tuning implementation", url: "https://github.com/tatsu-lab/stanford_alpaca" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 9: MoE — zh-TW
// ─────────────────────────────────────────────────────────────

const phase9ContentZhTW: PhaseContent = {
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
            { type: "paragraph", text: "到目前為止，我們的模型有一個「浪費」的問題：不管你問的是數學題還是寫詩，所有 70 億個參數都會被啟動。但人類不是這樣工作的——做數學題時你用的腦區跟寫詩時不同。Mixture of Experts（MoE）就是借鑒了這個思路：模型有 8 個「專家」，但每個 token 只啟動其中 2 個。結果是：模型可以有 560 億參數的「知識容量」，但每個 token 的計算量只相當於 70 億參數。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：MoE 解決了深度學習中最根本的困境之一：模型容量和推論成本的矛盾。Mixtral 8x7B 的效果接近 LLaMA-70B，但推論速度和 LLaMA-13B 差不多。" },
            { type: "table", headers: ["特性", "Dense Transformer", "MoE Transformer"], rows: [
              ["FFN 結構", "1 個大型 FFN", "N 個小型 Expert FFN"],
              ["每 token 激活參數", "100%", "Top-k/N（例如 2/8 = 25%）"],
              ["總參數量", "P", "~N×P"],
              ["FLOPs per token", "與參數量成正比", "與 active 參數量成正比"],
              ["訓練難度", "簡單", "需要 load balancing"],
            ]},
            { type: "callout", variant: "info", text: "Mixtral 8x7B 有 8 個專家，每個 token 路由到 2 個專家。總參數約 47B，但 active 參數只有約 13B——與 LLaMA-13B 的推論速度相近，但效果接近 LLaMA-34B。" },
          ],
        },
        {
          title: "Expert Network",
          blocks: [
            { type: "paragraph", text: "每個 Expert 就是一個標準的 Transformer FFN：Linear → GELU → Linear。唯一的差別是我們有多個這樣的 FFN，而不是一個。" },
            { type: "code", language: "python", code: "class Expert(nn.Module):\n    def __init__(self, d_model: int, d_ff: int):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),\n        )\n\n    def forward(self, x):\n        return self.net(x)" },
          ],
        },
        {
          title: "Router / Gate Network",
          blocks: [
            { type: "paragraph", text: "Router（也稱 Gate）是 MoE 的「大腦」——它決定每個 token 應該被哪些 Expert 處理。Router 是一個簡單的線性層，將 token 的表示映射到 N 維空間（N = 專家數），再用 softmax 轉換為概率分布。" },
            { type: "diagram", content: "Router 機制：\n  token embedding → Linear(d_model, N) → softmax → router probs\n  → top-k selection → renormalized weights\n\n例: probs = [0.05, 0.35, 0.02, 0.40, 0.03, 0.05, 0.08, 0.02]\n    top-2: Expert 3 (0.40), Expert 1 (0.35)\n    renorm: [0.533, 0.467]" },
            { type: "code", language: "python", code: "class Router(nn.Module):\n    def __init__(self, d_model, n_experts, top_k=2):\n        super().__init__()\n        self.gate = nn.Linear(d_model, n_experts)\n        self.top_k = top_k\n\n    def forward(self, x):  # x: (B, T, d_model)\n        B, T, D = x.shape\n        x_flat = x.view(B * T, D)\n        logits = self.gate(x_flat)\n        probs = F.softmax(logits, dim=-1)\n        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)\n        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)\n        return probs, top_k_weights, top_k_indices" },
            { type: "callout", variant: "warning", text: "Top-k 後必須 renormalize weights！確保選中的 weights 和為 1，讓 MoE 輸出的 scale 一致。" },
          ],
        },
        {
          title: "MoE Layer: 完整組裝",
          blocks: [
            { type: "paragraph", text: "MoELayer 將 Router 和 Experts 組合在一起。對每個 token，Router 選出 top-k 個 Expert，每個 Expert 獨立處理該 token，最後用路由權重加權合併輸出。" },
            { type: "code", language: "python", code: "class MoELayer(nn.Module):\n    def __init__(self, d_model, d_ff, n_experts, top_k=2):\n        super().__init__()\n        self.experts = nn.ModuleList(\n            [Expert(d_model, d_ff) for _ in range(n_experts)]\n        )\n        self.router = Router(d_model, n_experts, top_k)\n\n    def forward(self, x):  # x: (B, T, D)\n        B, T, D = x.shape\n        router_probs, top_k_weights, top_k_indices = self.router(x)\n        x_flat = x.view(B * T, D)\n        output = torch.zeros_like(x_flat)\n        for expert_idx in range(len(self.experts)):\n            mask = (top_k_indices == expert_idx).any(dim=-1)\n            if not mask.any():\n                continue\n            expert_out = self.experts[expert_idx](x_flat[mask])\n            weight_mask = (top_k_indices[mask] == expert_idx)\n            weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)\n            output[mask] += weights.unsqueeze(-1) * expert_out\n        return output.view(B, T, D), router_probs" },
          ],
        },
        {
          title: "Load Balancing Loss",
          blocks: [
            { type: "paragraph", text: "MoE 有一個嚴重的訓練問題：「專家崩塌」(expert collapse)。Router 傾向於將大部分 token 都路由到少數幾個表現好的 Expert，其他 Expert 得不到訓練，形成惡性循環。" },
            { type: "diagram", content: "Load Balancing Loss:\n\n  L_aux = N × Σᵢ (fᵢ × pᵢ)\n\n  fᵢ = 被路由到 Expert i 的 token 比例\n  pᵢ = 所有 token 對 Expert i 的平均路由概率\n  N  = 專家數\n\n最終 loss = CE loss + λ × L_aux  (λ ≈ 0.01)" },
            { type: "code", language: "python", code: "def load_balancing_loss(router_probs, top_k_indices, n_experts):\n    expert_mask = F.one_hot(top_k_indices, n_experts).sum(dim=1)\n    f = expert_mask.float().mean(dim=0)\n    p = router_probs.mean(dim=0)\n    return n_experts * (f * p).sum()" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Load balancing loss 是 MoE 中最 tricky 的部分。沒有它，Router 會傾向把所有 token 都送到同一個「最好的」專家——這就是「專家坍縮」（expert collapse）。Auxiliary loss 強迫 router 把 token 均勻分配到所有專家。" },
          ],
        },
        {
          title: "MoETransformerBlock & MoEGPT",
          blocks: [
            { type: "callout", variant: "quote", text: "💡 講師心得：實務中不會把每一層都換成 MoE——通常是每隔一層。例如 Mixtral 是「dense attention + MoE FFN」交替。原因是 attention 層已經很高效了，而 FFN 佔了模型大部分參數。只把 FFN 換成 MoE 就能獲得大部分的容量提升。" },
            { type: "code", language: "python", code: "class MoETransformerBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(config.n_embd)\n        self.attn = SampleAttention(config)\n        self.ln2 = nn.LayerNorm(config.n_embd)\n        self.moe = MoELayer(\n            config.n_embd, 4 * config.n_embd,\n            config.n_experts, config.top_k\n        )\n\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        moe_out, router_probs = self.moe(self.ln2(x))\n        x = x + moe_out\n        return x, router_probs" },
            { type: "diagram", content: "MoEGPT 架構 (moe_every_n_layers=2):\n\n  Layer 0: Dense Transformer Block\n  Layer 1: MoE Transformer Block   ← (1+1) % 2 == 0\n  Layer 2: Dense Transformer Block\n  Layer 3: MoE Transformer Block   ← (3+1) % 2 == 0\n  ...\n  → LayerNorm → Linear (vocab_size)" },
            { type: "callout", variant: "quote", text: "🤔 思考題：MoE 的 router 是一個簡單的 Linear 層 + Softmax。但這意味著不同的 token 可能被路由到不同的專家——同一個 batch 裡，有的專家可能收到很多 token，有的幾乎沒有。這會造成什麼工程問題？（提示：想想 GPU 並行計算需要什麼條件。Mixtral 是怎麼處理這個問題的？）" },
          ],
        },
      ],
      exercises: [
        { id: "expert_init_forward", title: "TODO 1: Expert", description: "實作單個 Expert 模組：標準 FFN (Linear → GELU → Linear)。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["使用 nn.Sequential 組合三層", "forward 直接 return self.net(x)"], pseudocode: "self.net = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))\ndef forward(x): return self.net(x)" },
        { id: "router_init_forward", title: "TODO 2: Router", description: "實作路由網路：Linear gate → softmax → top-k selection → renormalize weights。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["gate 是 nn.Linear(d_model, n_experts)", "softmax 得到 probs，torch.topk 選 top-k", "renormalize: top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)"], pseudocode: "logits = gate(x_flat)\nprobs = softmax(logits)\ntop_k_w, top_k_idx = topk(probs, k)\ntop_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)" },
        { id: "moe_layer", title: "TODO 3: MoELayer", description: "組裝 MoE 層：建立 N 個 Expert + 1 個 Router，forward 時路由每個 token 到 top-k experts 並加權合併輸出。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["用 nn.ModuleList 建立 experts 列表", "遍歷每個 expert，找出被路由到它的 token（mask），用 weight 加權累加"], pseudocode: "for i in range(n_experts):\n  mask = (topk_idx == i).any(dim=-1)\n  out_i = experts[i](x_flat[mask])\n  output[mask] += w_i * out_i" },
        { id: "load_balancing_loss", title: "TODO 4: load_balancing_loss", description: "實作輔助 load balancing loss：L_aux = N * sum(fi * pi)，防止 expert collapse。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["f_i: one_hot(top_k_indices).sum(dim=1).mean(dim=0)", "p_i: router_probs.mean(dim=0)", "return n_experts * (f * p).sum()"], pseudocode: "f = one_hot(topk_indices, N).sum(1).float().mean(0)\np = router_probs.mean(0)\nreturn N * (f * p).sum()" },
        { id: "moe_transformer_block", title: "TODO 5: MoETransformerBlock", description: "實作 MoE Transformer Block：LayerNorm → Attention → residual → LayerNorm → MoELayer → residual。", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["結構與 DenseTransformerBlock 相同，只是 FFN 替換為 MoELayer", "MoELayer 返回 (output, router_probs)，需要一起返回"], pseudocode: "x = x + attn(ln1(x))\nmoe_out, rp = moe(ln2(x))\nx = x + moe_out\nreturn x, rp" },
        { id: "moe_gpt", title: "TODO 6: MoEGPT", description: "建構完整 MoE GPT 模型：交錯 Dense/MoE blocks，forward 包含 CE loss + auxiliary loss。", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["Block i 是 MoE 如果 (i+1) % moe_every_n_layers == 0", "收集每個 MoE block 的 router_probs 計算 auxiliary loss", "total loss = CE loss + aux_loss_weight * mean(aux_losses)"], pseudocode: "blocks = [MoEBlock if (i+1)%N==0 else DenseBlock for i]\nfor block in blocks:\n  x, rp = block(x); collect rp\nce = cross_entropy(logits, targets)\naux = mean(load_balancing_loss for rp)\nloss = ce + weight * aux" },
      ],
      acceptanceCriteria: [
        "Router 輸出的 top_k_weights 每行和為 1（renormalized）",
        "MoELayer 的輸出 shape 與輸入相同，且所有 token 都被處理",
        "load_balancing_loss 在均勻路由時為最小值，崩塌時為最大值",
        "MoEGPT 的 loss 包含 CE loss 和 auxiliary load balancing loss",
        "交錯 Dense/MoE 的模式正確：(i+1) % moe_every_n_layers == 0",
      ],
      references: [
        { title: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer", description: "Shazeer et al. 2017", url: "https://arxiv.org/abs/1701.06538" },
        { title: "Switch Transformers", description: "Fedus et al. 2022 — top-1 routing 的簡化設計", url: "https://arxiv.org/abs/2101.03961" },
        { title: "Mixtral of Experts", description: "Jiang et al. 2024 — 交錯 Dense/MoE 設計的實際效果", url: "https://arxiv.org/abs/2401.04088" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 9: MoE — zh-CN
// ─────────────────────────────────────────────────────────────

const phase9ContentZhCN: PhaseContent = {
  phaseId: 9,
  color: "#A855F7",
  accent: "#C084FC",
  lessons: [
    {
      phaseId: 9, lessonId: 1,
      title: "MoE Architecture",
      subtitle: "用更多参数但不增加计算量——混合专家模型的设计哲学",
      type: "concept",
      duration: "75 min",
      objectives: [
        "理解 MoE 的核心动机：扩展参数量而不等比扩展计算量",
        "实现 Expert（标准 FFN）、Router（gating network）和 MoELayer",
        "理解 Top-k routing 与 load balancing loss 的必要性",
        "组装 MoETransformerBlock 并构建完整的 MoEGPT 模型",
      ],
      sections: [
        {
          title: "为什么需要混合专家？",
          blocks: [
            { type: "paragraph", text: "到目前为止，我们的模型有一个「浪费」的问题：不管你问的是数学题还是写诗，所有 70 亿个参数都会被激活。但人类不是这样工作的——做数学题时你用的脑区跟写诗时不同。Mixture of Experts（MoE）就是借鉴了这个思路：模型有 8 个「专家」，但每个 token 只激活其中 2 个。结果是：模型可以有 560 亿参数的「知识容量」，但每个 token 的计算量只相当于 70 亿参数。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：MoE 解决了深度学习中最根本的困境之一：模型容量和推理成本的矛盾。Mixtral 8x7B 的效果接近 LLaMA-70B，但推理速度和 LLaMA-13B 差不多。" },
            { type: "table", headers: ["特性", "Dense Transformer", "MoE Transformer"], rows: [
              ["FFN 结构", "1 个大型 FFN", "N 个小型 Expert FFN"],
              ["每 token 激活参数", "100%", "Top-k/N（例如 2/8 = 25%）"],
              ["总参数量", "P", "~N×P"],
              ["FLOPs per token", "与参数量成正比", "与 active 参数量成正比"],
              ["训练难度", "简单", "需要 load balancing"],
            ]},
            { type: "callout", variant: "info", text: "Mixtral 8x7B 有 8 个专家，每个 token 路由到 2 个专家。总参数约 47B，但 active 参数只有约 13B——与 LLaMA-13B 的推理速度相近，但效果接近 LLaMA-34B。" },
          ],
        },
        {
          title: "Expert 网络",
          blocks: [
            { type: "paragraph", text: "每个 Expert 就是一个标准的 Transformer FFN：Linear → GELU → Linear。唯一的差别是我们有多个这样的 FFN，而不是一个。" },
            { type: "code", language: "python", code: "class Expert(nn.Module):\n    def __init__(self, d_model: int, d_ff: int):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),\n        )\n\n    def forward(self, x):\n        return self.net(x)" },
          ],
        },
        {
          title: "Router / Gate 网络",
          blocks: [
            { type: "paragraph", text: "Router（也称 Gate）是 MoE 的「大脑」——它决定每个 token 应该被哪些 Expert 处理。Router 是一个简单的线性层，将 token 的表示映射到 N 维空间（N = 专家数），再用 softmax 转换为概率分布。" },
            { type: "diagram", content: "Router 机制：\n  token embedding → Linear(d_model, N) → softmax → router probs\n  → top-k selection → renormalized weights\n\n例: probs = [0.05, 0.35, 0.02, 0.40, 0.03, 0.05, 0.08, 0.02]\n    top-2: Expert 3 (0.40), Expert 1 (0.35)\n    renorm: [0.533, 0.467]" },
            { type: "code", language: "python", code: "class Router(nn.Module):\n    def __init__(self, d_model, n_experts, top_k=2):\n        super().__init__()\n        self.gate = nn.Linear(d_model, n_experts)\n        self.top_k = top_k\n\n    def forward(self, x):\n        B, T, D = x.shape\n        x_flat = x.view(B * T, D)\n        logits = self.gate(x_flat)\n        probs = F.softmax(logits, dim=-1)\n        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)\n        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)\n        return probs, top_k_weights, top_k_indices" },
            { type: "callout", variant: "warning", text: "Top-k 后必须 renormalize weights！确保选中的 weights 和为 1，让 MoE 输出的 scale 一致。" },
          ],
        },
        {
          title: "MoE Layer：完整组装",
          blocks: [
            { type: "paragraph", text: "MoELayer 将 Router 和 Experts 组合在一起。对每个 token，Router 选出 top-k 个 Expert，每个 Expert 独立处理该 token，最后用路由权重加权合并输出。" },
            { type: "code", language: "python", code: "class MoELayer(nn.Module):\n    def __init__(self, d_model, d_ff, n_experts, top_k=2):\n        super().__init__()\n        self.experts = nn.ModuleList(\n            [Expert(d_model, d_ff) for _ in range(n_experts)]\n        )\n        self.router = Router(d_model, n_experts, top_k)\n\n    def forward(self, x):\n        B, T, D = x.shape\n        router_probs, top_k_weights, top_k_indices = self.router(x)\n        x_flat = x.view(B * T, D)\n        output = torch.zeros_like(x_flat)\n        for expert_idx in range(len(self.experts)):\n            mask = (top_k_indices == expert_idx).any(dim=-1)\n            if not mask.any():\n                continue\n            expert_out = self.experts[expert_idx](x_flat[mask])\n            weight_mask = (top_k_indices[mask] == expert_idx)\n            weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)\n            output[mask] += weights.unsqueeze(-1) * expert_out\n        return output.view(B, T, D), router_probs" },
          ],
        },
        {
          title: "负载均衡损失",
          blocks: [
            { type: "paragraph", text: "MoE 有一个严重的训练问题：「专家崩塌」(expert collapse)。Router 倾向于将大部分 token 都路由到少数几个表现好的 Expert，其他 Expert 得不到训练，形成恶性循环。" },
            { type: "diagram", content: "Load Balancing Loss:\n\n  L_aux = N × Σᵢ (fᵢ × pᵢ)\n\n  fᵢ = 被路由到 Expert i 的 token 比例\n  pᵢ = 所有 token 对 Expert i 的平均路由概率\n  N  = 专家数\n\n最终 loss = CE loss + λ × L_aux  (λ ≈ 0.01)" },
            { type: "code", language: "python", code: "def load_balancing_loss(router_probs, top_k_indices, n_experts):\n    expert_mask = F.one_hot(top_k_indices, n_experts).sum(dim=1)\n    f = expert_mask.float().mean(dim=0)\n    p = router_probs.mean(dim=0)\n    return n_experts * (f * p).sum()" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：Load balancing loss 是 MoE 中最 tricky 的部分。没有它，Router 会倾向把所有 token 都送到同一个「最好的」专家——这就是「专家崩塌」（expert collapse）。Auxiliary loss 强迫 router 把 token 均匀分配到所有专家。" },
          ],
        },
        {
          title: "MoETransformerBlock 与 MoEGPT",
          blocks: [
            { type: "callout", variant: "quote", text: "💡 讲师心得：实践中不会把每一层都换成 MoE——通常是每隔一层。例如 Mixtral 是「dense attention + MoE FFN」交替。原因是 attention 层已经很高效了，而 FFN 占了模型大部分参数。只把 FFN 换成 MoE 就能获得大部分的容量提升。" },
            { type: "code", language: "python", code: "class MoETransformerBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(config.n_embd)\n        self.attn = SampleAttention(config)\n        self.ln2 = nn.LayerNorm(config.n_embd)\n        self.moe = MoELayer(\n            config.n_embd, 4 * config.n_embd,\n            config.n_experts, config.top_k\n        )\n\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        moe_out, router_probs = self.moe(self.ln2(x))\n        x = x + moe_out\n        return x, router_probs" },
            { type: "diagram", content: "MoEGPT 架构 (moe_every_n_layers=2):\n\n  Layer 0: Dense Transformer Block\n  Layer 1: MoE Transformer Block   ← (1+1) % 2 == 0\n  Layer 2: Dense Transformer Block\n  Layer 3: MoE Transformer Block   ← (3+1) % 2 == 0\n  ...\n  → LayerNorm → Linear (vocab_size)" },
            { type: "callout", variant: "quote", text: "🤔 思考题：MoE 的 router 是一个简单的 Linear 层 + Softmax。但这意味着不同的 token 可能被路由到不同的专家——同一个 batch 里，有的专家可能收到很多 token，有的几乎没有。这会造成什么工程问题？（提示：想想 GPU 并行计算需要什么条件。Mixtral 是怎么处理这个问题的？）" },
          ],
        },
      ],
      exercises: [
        { id: "expert_init_forward", title: "TODO 1: Expert", description: "实现单个 Expert 模块：标准 FFN (Linear → GELU → Linear)。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["使用 nn.Sequential 组合三层", "forward 直接 return self.net(x)"], pseudocode: "self.net = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))\ndef forward(x): return self.net(x)" },
        { id: "router_init_forward", title: "TODO 2: Router", description: "实现路由网络：Linear gate → softmax → top-k selection → renormalize weights。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["gate 是 nn.Linear(d_model, n_experts)", "softmax 得到 probs，torch.topk 选 top-k", "renormalize: top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)"], pseudocode: "logits = gate(x_flat)\nprobs = softmax(logits)\ntop_k_w, top_k_idx = topk(probs, k)\ntop_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)" },
        { id: "moe_layer", title: "TODO 3: MoELayer", description: "组装 MoE 层：建立 N 个 Expert + 1 个 Router，forward 时路由每个 token 到 top-k experts 并加权合并输出。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["用 nn.ModuleList 建立 experts 列表", "遍历每个 expert，找出被路由到它的 token（mask），用 weight 加权累加"], pseudocode: "for i in range(n_experts):\n  mask = (topk_idx == i).any(dim=-1)\n  out_i = experts[i](x_flat[mask])\n  output[mask] += w_i * out_i" },
        { id: "load_balancing_loss", title: "TODO 4: load_balancing_loss", description: "实现辅助 load balancing loss：L_aux = N * sum(fi * pi)，防止 expert collapse。", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["f_i: one_hot(top_k_indices).sum(dim=1).mean(dim=0)", "p_i: router_probs.mean(dim=0)"], pseudocode: "f = one_hot(topk_indices, N).sum(1).float().mean(0)\np = router_probs.mean(0)\nreturn N * (f * p).sum()" },
        { id: "moe_transformer_block", title: "TODO 5: MoETransformerBlock", description: "实现 MoE Transformer Block：LayerNorm → Attention → residual → LayerNorm → MoELayer → residual。", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["结构与 DenseTransformerBlock 相同，只是 FFN 替换为 MoELayer", "MoELayer 返回 (output, router_probs)，需要一起返回"], pseudocode: "x = x + attn(ln1(x))\nmoe_out, rp = moe(ln2(x))\nx = x + moe_out\nreturn x, rp" },
        { id: "moe_gpt", title: "TODO 6: MoEGPT", description: "构建完整 MoE GPT 模型：交错 Dense/MoE blocks，forward 包含 CE loss + auxiliary loss。", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["Block i 是 MoE 如果 (i+1) % moe_every_n_layers == 0", "收集每个 MoE block 的 router_probs 计算 auxiliary loss"], pseudocode: "blocks = [MoEBlock if (i+1)%N==0 else DenseBlock for i]\nce = cross_entropy(logits, targets)\naux = mean(load_balancing_loss for each MoE block)\nloss = ce + weight * aux" },
      ],
      acceptanceCriteria: [
        "Router 输出的 top_k_weights 每行和为 1（renormalized）",
        "MoELayer 的输出 shape 与输入相同，且所有 token 都被处理",
        "load_balancing_loss 在均匀路由时为最小值，崩塌时为最大值",
        "MoEGPT 的 loss 包含 CE loss 和 auxiliary load balancing loss",
        "交错 Dense/MoE 的模式正确：(i+1) % moe_every_n_layers == 0",
      ],
      references: [
        { title: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer", description: "Shazeer et al. 2017", url: "https://arxiv.org/abs/1701.06538" },
        { title: "Switch Transformers", description: "Fedus et al. 2022 — top-1 routing 的简化设计", url: "https://arxiv.org/abs/2101.03961" },
        { title: "Mixtral of Experts", description: "Jiang et al. 2024 — 交错 Dense/MoE 设计的实际效果", url: "https://arxiv.org/abs/2401.04088" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Phase 9: MoE — en
// ─────────────────────────────────────────────────────────────

const phase9ContentEn: PhaseContent = {
  phaseId: 9,
  color: "#A855F7",
  accent: "#C084FC",
  lessons: [
    {
      phaseId: 9, lessonId: 1,
      title: "MoE Architecture",
      subtitle: "More parameters, same compute — the design philosophy of Mixture of Experts",
      type: "concept",
      duration: "75 min",
      objectives: [
        "Understand MoE's core motivation: scaling parameter count without proportionally scaling compute",
        "Implement Expert (standard FFN), Router (gating network), and MoELayer",
        "Understand why Top-k routing needs a load balancing loss",
        "Assemble MoETransformerBlock and build a complete MoEGPT model",
      ],
      sections: [
        {
          title: "Why Mixture of Experts?",
          blocks: [
            { type: "paragraph", text: "Our models so far have a 'wasteful' problem: whether you're asking a math question or writing poetry, all 7 billion parameters activate for every token. But that's not how humans work — the brain regions you use for math are different from those for poetry. Mixture of Experts (MoE) borrows this idea: the model has 8 'experts,' but each token activates only 2 of them. The result: the model can have the knowledge capacity of 56 billion parameters, but each token's compute cost is equivalent to 7 billion." },
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: MoE addresses one of deep learning's most fundamental tensions: model capacity vs. inference cost. Mixtral 8x7B performs close to LLaMA-70B but runs at roughly LLaMA-13B inference speed." },
            { type: "table", headers: ["Property", "Dense Transformer", "MoE Transformer"], rows: [
              ["FFN structure", "One large FFN", "N small Expert FFNs"],
              ["Active params per token", "100%", "Top-k/N (e.g., 2/8 = 25%)"],
              ["Total parameters", "P", "~N×P"],
              ["FLOPs per token", "Proportional to all params", "Proportional to active params"],
              ["Training difficulty", "Simple", "Requires load balancing"],
            ]},
            { type: "callout", variant: "info", text: "Mixtral 8x7B has 8 experts with top-2 routing per token. Total ~47B params, but only ~13B active — LLaMA-13B inference speed with LLaMA-34B-level performance." },
          ],
        },
        {
          title: "The Expert Network",
          blocks: [
            { type: "paragraph", text: "Each Expert is a standard Transformer FFN: Linear → GELU → Linear. The only difference from a dense model is that we have multiple such FFNs instead of one." },
            { type: "code", language: "python", code: "class Expert(nn.Module):\n    def __init__(self, d_model: int, d_ff: int):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Linear(d_ff, d_model),\n        )\n\n    def forward(self, x):\n        return self.net(x)" },
            { type: "callout", variant: "tip", text: "Expert architecture is intentionally identical to a dense FFN, making MoE a drop-in replacement. Each expert learns different feature transformations." },
          ],
        },
        {
          title: "The Router / Gate Network",
          blocks: [
            { type: "paragraph", text: "The Router (also called Gate) is MoE's 'brain' — it decides which experts process each token. It's a simple linear layer mapping token representations to N-dimensional space (N = number of experts), then softmax to get routing probabilities." },
            { type: "diagram", content: "Router mechanism:\n  token embedding → Linear(d_model, N) → softmax → router probs\n  → top-k selection → renormalized weights\n\ne.g.: probs = [0.05, 0.35, 0.02, 0.40, 0.03, 0.05, 0.08, 0.02]\n      top-2: Expert 3 (0.40), Expert 1 (0.35)\n      renorm: [0.533, 0.467]" },
            { type: "code", language: "python", code: "class Router(nn.Module):\n    def __init__(self, d_model, n_experts, top_k=2):\n        super().__init__()\n        self.gate = nn.Linear(d_model, n_experts)\n        self.top_k = top_k\n\n    def forward(self, x):\n        B, T, D = x.shape\n        x_flat = x.view(B * T, D)\n        logits = self.gate(x_flat)\n        probs = F.softmax(logits, dim=-1)\n        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)\n        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)\n        return probs, top_k_weights, top_k_indices" },
            { type: "callout", variant: "warning", text: "Always renormalize after top-k selection! This ensures the selected weights sum to 1, keeping MoE output scale consistent." },
          ],
        },
        {
          title: "MoE Layer: Full Assembly",
          blocks: [
            { type: "paragraph", text: "MoELayer combines the Router and Experts. For each token, the Router selects top-k Experts; each Expert independently processes its assigned tokens; outputs are then weighted and summed." },
            { type: "code", language: "python", code: "class MoELayer(nn.Module):\n    def __init__(self, d_model, d_ff, n_experts, top_k=2):\n        super().__init__()\n        self.experts = nn.ModuleList(\n            [Expert(d_model, d_ff) for _ in range(n_experts)]\n        )\n        self.router = Router(d_model, n_experts, top_k)\n\n    def forward(self, x):\n        B, T, D = x.shape\n        router_probs, top_k_weights, top_k_indices = self.router(x)\n        x_flat = x.view(B * T, D)\n        output = torch.zeros_like(x_flat)\n        for expert_idx in range(len(self.experts)):\n            mask = (top_k_indices == expert_idx).any(dim=-1)\n            if not mask.any():\n                continue\n            expert_out = self.experts[expert_idx](x_flat[mask])\n            weight_mask = (top_k_indices[mask] == expert_idx)\n            weights = (top_k_weights[mask] * weight_mask.float()).sum(dim=-1)\n            output[mask] += weights.unsqueeze(-1) * expert_out\n        return output.view(B, T, D), router_probs" },
          ],
        },
        {
          title: "Load Balancing Loss",
          blocks: [
            { type: "paragraph", text: "MoE has a critical training failure mode: expert collapse. The router tends to route most tokens to a few high-performing experts, leaving others undertrained — a vicious cycle that can reduce MoE to essentially a dense model." },
            { type: "diagram", content: "Load Balancing Loss:\n\n  L_aux = N × Σᵢ (fᵢ × pᵢ)\n\n  fᵢ = fraction of tokens dispatched to Expert i\n  pᵢ = mean routing probability for Expert i\n  N  = number of experts\n\nFinal loss = CE loss + λ × L_aux  (λ ≈ 0.01)" },
            { type: "code", language: "python", code: "def load_balancing_loss(router_probs, top_k_indices, n_experts):\n    expert_mask = F.one_hot(top_k_indices, n_experts).sum(dim=1)\n    f = expert_mask.float().mean(dim=0)\n    p = router_probs.mean(dim=0)\n    return n_experts * (f * p).sum()" },
            { type: "callout", variant: "tip", text: "💡 Instructor's Note: Load balancing loss is the trickiest part of MoE. Without it, the router collapses to sending everything to one 'best' expert. This auxiliary loss forces uniform token distribution across experts. It's a hack — but an effective one. The Switch Transformer paper devotes considerable space to this problem." },
            { type: "callout", variant: "info", text: "Final loss = CE loss + λ × L_aux, where λ (aux_loss_weight) is typically 0.01–0.1. Too large sacrifices language modeling quality; too small fails to prevent collapse." },
          ],
        },
        {
          title: "MoETransformerBlock & MoEGPT",
          blocks: [
            { type: "callout", variant: "quote", text: "💡 Instructor's Note: In practice, not every layer is replaced with MoE — typically every other layer. Mixtral alternates 'dense attention + MoE FFN.' The attention layer is already efficient (shared), while FFN holds most of the parameters. Replacing only FFN with MoE captures most of the capacity gain while keeping training stable." },
            { type: "code", language: "python", code: "class MoETransformerBlock(nn.Module):\n    def __init__(self, config):\n        super().__init__()\n        self.ln1 = nn.LayerNorm(config.n_embd)\n        self.attn = SampleAttention(config)\n        self.ln2 = nn.LayerNorm(config.n_embd)\n        self.moe = MoELayer(\n            config.n_embd, 4 * config.n_embd,\n            config.n_experts, config.top_k\n        )\n\n    def forward(self, x):\n        x = x + self.attn(self.ln1(x))\n        moe_out, router_probs = self.moe(self.ln2(x))\n        x = x + moe_out\n        return x, router_probs" },
            { type: "diagram", content: "MoEGPT architecture (moe_every_n_layers=2):\n\n  Layer 0: Dense Transformer Block\n  Layer 1: MoE Transformer Block    ← (1+1) % 2 == 0\n  Layer 2: Dense Transformer Block\n  Layer 3: MoE Transformer Block    ← (3+1) % 2 == 0\n  ...\n  → LayerNorm → Linear (vocab_size)" },
            { type: "callout", variant: "quote", text: "🤔 Think About It: The MoE router is a simple Linear + Softmax. But this means different tokens may go to different experts — in the same batch, some experts might receive many tokens while others get almost none. What engineering problems does this create? (Hint: what does GPU parallel computation require? How does Mixtral address this?)" },
          ],
        },
      ],
      exercises: [
        { id: "expert_init_forward", title: "TODO 1: Expert", description: "Implement a single Expert module: standard FFN (Linear → GELU → Linear).", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["Use nn.Sequential to combine the three layers", "forward just returns self.net(x)"], pseudocode: "self.net = Sequential(Linear(d_model, d_ff), GELU(), Linear(d_ff, d_model))\ndef forward(x): return self.net(x)" },
        { id: "router_init_forward", title: "TODO 2: Router", description: "Implement the routing network: Linear gate → softmax → top-k selection → renormalize weights.", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["gate is nn.Linear(d_model, n_experts)", "softmax gives probs, torch.topk selects top-k", "renormalize: top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)"], pseudocode: "logits = gate(x_flat)\nprobs = softmax(logits)\ntop_k_w, top_k_idx = topk(probs, k)\ntop_k_w = top_k_w / top_k_w.sum(dim=-1, keepdim=True)" },
        { id: "moe_layer", title: "TODO 3: MoELayer", description: "Assemble the MoE layer: N Experts + 1 Router. In forward, route each token to top-k experts and combine weighted outputs.", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["Use nn.ModuleList for the experts", "Iterate experts, find routed tokens (mask), accumulate weighted outputs"], pseudocode: "for i in range(n_experts):\n  mask = (topk_idx == i).any(dim=-1)\n  out_i = experts[i](x_flat[mask])\n  output[mask] += w_i * out_i" },
        { id: "load_balancing_loss", title: "TODO 4: load_balancing_loss", description: "Implement the auxiliary load balancing loss: L_aux = N * sum(fi * pi), preventing expert collapse.", labFile: "labs/phase9_moe/phase_9/moe.py", hints: ["f_i: one_hot(top_k_indices).sum(dim=1).mean(dim=0)", "p_i: router_probs.mean(dim=0)", "return n_experts * (f * p).sum()"], pseudocode: "f = one_hot(topk_indices, N).sum(1).float().mean(0)\np = router_probs.mean(0)\nreturn N * (f * p).sum()" },
        { id: "moe_transformer_block", title: "TODO 5: MoETransformerBlock", description: "Implement the MoE Transformer Block: LayerNorm → Attention → residual → LayerNorm → MoELayer → residual.", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["Same structure as DenseTransformerBlock, but FFN replaced with MoELayer", "MoELayer returns (output, router_probs) — return both"], pseudocode: "x = x + attn(ln1(x))\nmoe_out, rp = moe(ln2(x))\nx = x + moe_out\nreturn x, rp" },
        { id: "moe_gpt", title: "TODO 6: MoEGPT", description: "Build the complete MoE GPT model with interleaved Dense/MoE blocks. Forward includes CE loss + auxiliary loss.", labFile: "labs/phase9_moe/phase_9/moe_transformer.py", hints: ["Block i is MoE if (i+1) % moe_every_n_layers == 0", "Collect router_probs from each MoE block for the auxiliary loss", "total loss = CE loss + aux_loss_weight * mean(aux_losses)"], pseudocode: "blocks = [MoEBlock if (i+1)%N==0 else DenseBlock for i]\nce = cross_entropy(logits, targets)\naux = mean(load_balancing_loss for each MoE block)\nloss = ce + weight * aux" },
      ],
      acceptanceCriteria: [
        "Router top_k_weights sum to 1 per row (renormalized)",
        "MoELayer output shape matches input shape; all tokens are processed",
        "load_balancing_loss is minimized with uniform routing and maximized with expert collapse",
        "MoEGPT loss includes both CE loss and auxiliary load balancing loss",
        "Interleaved Dense/MoE pattern is correct: (i+1) % moe_every_n_layers == 0",
      ],
      references: [
        { title: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer", description: "Shazeer et al. 2017", url: "https://arxiv.org/abs/1701.06538" },
        { title: "Switch Transformers", description: "Fedus et al. 2022 — simplified top-1 routing design", url: "https://arxiv.org/abs/2101.03961" },
        { title: "Mixtral of Experts", description: "Jiang et al. 2024 — interleaved Dense/MoE in practice", url: "https://arxiv.org/abs/2401.04088" },
      ],
    },
  ],
};

// ─────────────────────────────────────────────────────────────
// Dispatch maps & exports
// ─────────────────────────────────────────────────────────────

const phase7Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase7ContentZhTW,
  "zh-CN": phase7ContentZhCN,
  "en": phase7ContentEn,
};

const phase8Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase8ContentZhTW,
  "zh-CN": phase8ContentZhCN,
  "en": phase8ContentEn,
};

const phase9Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase9ContentZhTW,
  "zh-CN": phase9ContentZhCN,
  "en": phase9ContentEn,
};

export function getPhase7Content(locale: Locale): PhaseContent { return phase7Map[locale]; }
export function getPhase8Content(locale: Locale): PhaseContent { return phase8Map[locale]; }
export function getPhase9Content(locale: Locale): PhaseContent { return phase9Map[locale]; }
