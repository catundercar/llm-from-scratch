import type { PhaseContent } from "@/data/types";
import type { Locale } from "@/i18n";

// ═══════════════════════════════════════════════════════════════════
// Phase 5: Text Generation
// ═══════════════════════════════════════════════════════════════════

const phase5ContentZhTW: PhaseContent = {
  phaseId: 5,
  color: "#10B981",
  accent: "#34D399",
  lessons: [
    {
      phaseId: 5, lessonId: 1,
      title: "Generation Strategies",
      subtitle: "從 Greedy 到 Nucleus Sampling——掌握文本生成的各種解碼策略",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 autoregressive generation 的核心迴圈",
        "實作 greedy decoding 並理解其局限性",
        "理解 temperature 參數如何重塑機率分布",
        "實作 top-k sampling 與 top-p (nucleus) sampling",
        "建立統一的 generate 介面整合所有策略",
      ],
      sections: [
        {
          title: "Autoregressive Generation Loop",
          blocks: [
            { type: "paragraph", text: "模型訓練好了，現在到了最令人期待的時刻——讓它開口「說話」。但你可能會發現一個奇怪的現象：同一個 prompt，用不同的生成策略，模型可能寫出無聊的重複句子，也可能寫出驚艷的創意文章。差別在哪裡？全在今天要學的這幾種 sampling 策略裡。你在 ChatGPT 裡調的那個 Temperature 滑桿，背後就是這些數學。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：文本生成是你第一次看到模型「活過來」的時刻。前面四個 Phase 都是在訓練——模型是被動的學生。從這個 Phase 開始，模型要主動「創作」了。但生成策略的選擇比你想像的重要得多——同一個模型，用 greedy decoding 可能輸出枯燥重複的文字，換成 nucleus sampling 就能寫出流暢自然的文章。" },
            { type: "paragraph", text: "語言模型的生成過程本質上是一個自回歸（autoregressive）迴圈：模型每次只生成一個 token，然後將這個新 token 加入輸入序列，再用更新後的序列預測下一個 token。這個過程反覆執行，直到達到指定的長度或遇到結束標記。" },
            { type: "paragraph", text: "關鍵觀察：模型的 forward pass 輸出的是整個詞彙表上的 logits（未歸一化的分數），而不是單一的 token。我們只關心最後一個位置的 logits——因為前面位置的預測已經在訓練時使用過了。如何從這些 logits 中選擇下一個 token，就是「解碼策略」要解決的問題。" },
            { type: "diagram", content: "Autoregressive Generation Loop:\n\n  ┌─────────────────────────────────────────────────┐\n  │  Input: [The, cat, sat]                         │\n  │                                                 │\n  │  ┌──────────┐    logits[:, -1, :]    ┌────────┐ │\n  │  │ GPT Model│ ──────────────────────>│Decode  │ │\n  │  │ Forward  │    (vocab_size,)       │Strategy│ │\n  │  └──────────┘                        └───┬────┘ │\n  │                                          │      │\n  │       next_token = selected token        │      │\n  │                                          v      │\n  │  Input: [The, cat, sat, on]  <── append token   │\n  └─────────────────────────────────────────────────┘" },
            { type: "callout", variant: "info", text: "如果模型有 block_size（最大上下文長度）屬性，當序列長度超過 block_size 時需要截斷。只保留最後 block_size 個 token 作為輸入，避免超出位置嵌入的範圍。" },
            { type: "code", language: "python", code: "# Autoregressive generation 的通用骨架\nmodel.eval()\nwith torch.no_grad():\n    for _ in range(max_new_tokens):\n        # 截斷到 block_size（如果需要）\n        idx_cond = idx[:, -block_size:] if hasattr(model, 'block_size') else idx\n        # Forward pass\n        logits, _ = model(idx_cond)\n        # 只取最後一個位置的 logits\n        next_logits = logits[:, -1, :]    # (B, vocab_size)\n        # ── 這裡插入解碼策略 ──\n        next_token = decode_strategy(next_logits)\n        # 將新 token 附加到序列\n        idx = torch.cat([idx, next_token], dim=1)" },
          ],
        },
        {
          title: "Greedy Decoding",
          blocks: [
            { type: "paragraph", text: "最簡單的解碼策略：每一步都選擇機率最高的 token（argmax）。這保證了輸出的確定性——同樣的輸入永遠產生同樣的輸出。" },
            { type: "code", language: "python", code: "# Greedy decoding: 永遠選最大的\nnext_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n# next_token shape: (B, 1)" },
            { type: "heading", level: 3, text: "Greedy Decoding 的問題" },
            { type: "list", ordered: true, items: [
              "重複性高：模型容易陷入重複迴圈，生成 'the the the...' 或重複相同的句子",
              "缺乏創意：永遠走最安全的路線，無法產生驚喜或多樣性",
              "全域次優：逐步的局部最優不保證整個序列的全域最優",
            ]},
            { type: "callout", variant: "tip", text: "💡 講師心得：Greedy decoding 的問題不是它「太確定」，而是它「局部最優但全局最差」。就像下棋只看一步——每一步都選「當前最佳」，結果卻走進死胡同。具體表現就是重複（'the the the the...'）——模型進入了一個高概率的循環，但沒有全局規劃能力跳出來。" },
          ],
        },
        {
          title: "Temperature Scaling",
          blocks: [
            { type: "paragraph", text: "Temperature 是控制生成隨機性的最直觀參數。其原理是在 softmax 之前將 logits 除以一個正數 T（temperature）：" },
            { type: "code", language: "python", code: "# Temperature scaling\nscaled_logits = logits / temperature\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "diagram", content: "Temperature 對機率分布的影響（原始 logits = [2.0, 1.0, 0.5, 0.1]）:\n\n  T=0.1 (極低)  ████████████████████████  token_0: 99.7%\n  T=0.5 (低)    ██████████████████        token_0: 73.1%\n  T=1.0 (標準)  ██████████████            token_0: 46.1%\n  T=2.0 (高)    █████████                 token_0: 33.0%" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Temperature 的本質是在 softmax 之前對 logits 做縮放。T<1 時 logits 差異被放大→分布更尖銳→更確定；T>1 時差異被縮小→分布更平坦→更隨機。T→0 退化為 greedy，T→∞ 退化為均勻分布。" },
            { type: "callout", variant: "warning", text: "Temperature = 0 會導致除以零！實作中需要特殊處理：當 temperature < 1e-8 時退回 greedy decoding（argmax）。" },
          ],
        },
        {
          title: "Top-k Sampling",
          blocks: [
            { type: "paragraph", text: "Temperature sampling 的一個問題是：即使在低溫下，長尾中那些機率極低的 token 仍有可能被取樣到。Top-k sampling 直接截斷：只保留機率最高的 k 個 token，其餘全部排除。" },
            { type: "code", language: "python", code: "# Top-k sampling\nscaled_logits = logits / temperature\ntop_k_values, _ = torch.topk(scaled_logits, k=k)\nthreshold = top_k_values[:, -1:]\nscaled_logits[scaled_logits < threshold] = float('-inf')\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "k = 1 等價於 greedy decoding。k = vocab_size 等價於純 temperature sampling。實務上 k = 40~100 是常見的選擇。" },
          ],
        },
        {
          title: "Top-p (Nucleus) Sampling",
          blocks: [
            { type: "paragraph", text: "Top-p sampling（又稱 nucleus sampling）優雅地解決了 top-k 的固定截斷問題。其核心思想是：動態選取最小的 token 集合，使得這些 token 的累積機率恰好超過門檻 p。" },
            { type: "code", language: "python", code: "# Top-p (nucleus) sampling\nscaled_logits = logits / temperature\nsorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)\ncumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)\nmask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p\nsorted_logits[mask] = float('-inf')\nscaled_logits.scatter_(1, sorted_indices, sorted_logits)\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Top-p 比 top-k 更聰明的地方在於它能自適應。當模型很確定時，top-p=0.9 可能只保留 1-2 個 token。當模型不確定時，同樣的 p=0.9 可能保留幾十個 token。這就是為什麼 GPT-4 默認用 top-p 而非 top-k。" },
          ],
        },
        {
          title: "策略比較與統一介面",
          blocks: [
            { type: "table", headers: ["策略", "參數", "確定性", "多樣性", "適用場景"], rows: [
              ["Greedy", "無", "完全確定", "無", "程式碼補全、事實問答"],
              ["Temperature", "T ∈ (0, ∞)", "T→0 確定", "T↑ 增加", "創意寫作（T=0.7~0.9）"],
              ["Top-k", "k ∈ [1, V]", "k=1 確定", "k↑ 增加", "一般文本生成（k=40~100）"],
              ["Top-p", "p ∈ (0, 1]", "p→0 確定", "p↑ 增加", "最佳通用選擇（p=0.9~0.95）"],
            ]},
            { type: "code", language: "python", code: "def generate(model, tokenizer, prompt, max_new_tokens=50,\n             strategy='greedy', device='cpu', **kwargs):\n    ids = tokenizer.encode(prompt)\n    idx = torch.tensor([ids], dtype=torch.long, device=device)\n    strategies = {\n        'greedy': greedy_decode,\n        'temperature': temperature_sample,\n        'top_k': top_k_sample,\n        'top_p': top_p_sample,\n    }\n    fn = strategies[strategy]\n    output_ids = fn(model, idx, max_new_tokens, device=device, **kwargs)\n    return tokenizer.decode(output_ids[0].tolist())" },
            { type: "callout", variant: "quote", text: "🤔 思考題：在實際的 LLM API 中，你可以同時設定 temperature 和 top_p。這兩個參數會怎麼交互作用？是先做 temperature scaling 再做 top-p truncation，還是反過來？順序不同結果一樣嗎？" },
          ],
        },
      ],
      exercises: [
        { id: "greedy_decode", title: "TODO 1: greedy_decode", description: "實作 greedy decoding——每一步選擇 logits 最大的 token（argmax）。需處理 block_size 截斷、model.eval() 與 torch.no_grad()。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["使用 model.eval() 和 torch.no_grad() 進行推論", "如果模型有 block_size 屬性，截斷 idx 到最後 block_size 個 token", "torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) 取得下一個 token"], pseudocode: "model.eval()\nwith torch.no_grad():\n  for _ in range(max_new_tokens):\n    idx_cond = idx[:, -block_size:] if hasattr(...) else idx\n    logits, _ = model(idx_cond)\n    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n    idx = torch.cat([idx, next_id], dim=1)\nreturn idx" },
        { id: "temperature_sample", title: "TODO 2: temperature_sample", description: "實作 temperature sampling——將 logits 除以 temperature 後做 softmax 再用 multinomial 取樣。當 temperature ≈ 0 時退回 greedy。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["特殊處理 temperature < 1e-8：使用 argmax 取代 sampling", "probs = torch.softmax(logits / temperature, dim=-1)", "next_id = torch.multinomial(probs, num_samples=1)"] },
        { id: "top_k_sample", title: "TODO 3: top_k_sample", description: "實作 top-k sampling——只從機率最高的 k 個 token 中取樣。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["k = min(k, logits.size(-1)) 防止越界", "torch.topk(logits, k) 取得前 k 大的值", "低於門檻的 logits 設為 float('-inf')"] },
        { id: "top_p_sample", title: "TODO 4: top_p_sample", description: "實作 top-p (nucleus) sampling——動態選取累積機率超過 p 的最小 token 集合。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["torch.sort(logits, descending=True) 降序排列", "cumsum 計算累積機率", "遮罩需偏移一位以保留至少一個 token", "logits.scatter_(1, sorted_indices, sorted_logits) 還原順序"] },
        { id: "generate", title: "TODO 5: generate (unified interface)", description: "實作統一生成介面——接受文字 prompt，encode → 選擇策略 → decode 回文字。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["用 tokenizer.encode(prompt) 取得 token IDs", "建立策略字典分派", "tokenizer.decode(output_ids[0].tolist()) 解碼回文字"] },
      ],
      acceptanceCriteria: [
        "greedy_decode 輸出 shape 為 (1, T + max_new_tokens) 且完全確定性",
        "temperature_sample 在 temperature ≈ 0 時等價於 greedy decoding",
        "top_k_sample 在 k=1 時等價於 greedy decoding",
        "top_p_sample 在 p 極小時接近 greedy，p=1.0 時包含全部 token",
        "generate() 正確分派策略並回傳字串，非法策略拋出例外",
        "所有函數保留原始 input prefix 不變",
      ],
      references: [
        { title: "The Curious Case of Neural Text Degeneration", description: "Holtzman et al. 2019 — 提出 nucleus (top-p) sampling，分析了 greedy / beam search 的退化問題", url: "https://arxiv.org/abs/1904.09751" },
        { title: "How to generate text: using different decoding methods for language generation with Transformers", description: "HuggingFace 官方部落格，圖文並茂地解釋各種解碼策略", url: "https://huggingface.co/blog/how-to-generate" },
      ],
    },
  ],
};

const phase6ContentZhTW: PhaseContent = {
  phaseId: 6,
  color: "#06B6D4",
  accent: "#22D3EE",
  lessons: [
    {
      phaseId: 6, lessonId: 1,
      title: "Classification with GPT",
      subtitle: "從語言模型到分類器——用 GPT 的隱藏表示進行下游任務",
      type: "concept",
      duration: "50 min",
      objectives: [
        "理解為什麼使用最後一個 token 的表示做分類",
        "實作 GPTClassifier：在 GPT backbone 上加入分類頭",
        "掌握 feature extraction（凍結 backbone）與 full fine-tuning 兩種策略",
        "實作 SpamDataset：從 CSV 載入、tokenize、padding 和 attention mask",
        "理解 freeze/unfreeze 對梯度流的影響",
      ],
      sections: [
        {
          title: "為什麼用最後一個 Token 的表示？",
          blocks: [
            { type: "paragraph", text: "到目前為止，我們的 GPT 模型只會一件事：生成文本。但在現實世界中，很多任務不是「寫作文」，而是「做判斷」——這封郵件是不是垃圾郵件？這條評論是正面還是負面？今天我們要把一個「說書人」（生成模型）改造成「裁判」（分類模型）。你會驚訝地發現，這個改造出奇地簡單——只需要在模型頂部加一層就夠了。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：把一個生成式模型改成分類器，聽起來有點違直覺。關鍵在於預訓練過程中，模型已經學到了豐富的語言理解能力。分類微調只是在模型頂部加一個「分類頭」，利用這些已有的語言知識來做決策。這就是遷移學習的威力。" },
            { type: "paragraph", text: "GPT 是一個 causal（自回歸）語言模型——每個 token 只能 attend 到它自己和之前的 token。這意味著最後一個 token 的隱藏狀態是唯一「看過整個輸入序列」的表示，它彙集了整個輸入的語義資訊。" },
            { type: "diagram", content: "GPT Causal Attention — 資訊流向:\n\n  Token:    [This]  [is]  [spam]  [!]\n  Attend:   [自己] [前2] [前3]  [全部]\n  Info:      ●       ●●     ●●●    ●●●●\n                                   ▲\n                          最後一個 token 包含\n                          整個序列的資訊\n                                   ▼\n                          [Classification Head]\n                                   ▼\n                          [spam / not spam]" },
            { type: "callout", variant: "info", text: "如果序列有 padding，最後一個「真實」token 才是有意義的。此時需要 attention_mask 來找到最後一個非 padding 位置：last_pos = attention_mask.sum(dim=1) - 1。" },
          ],
        },
        {
          title: "GPTClassifier 架構",
          blocks: [
            { type: "diagram", content: "GPTClassifier Architecture:\n\n  input_ids (B, T)\n       │\n       ▼\n  ┌──────────────────────────────────────┐\n  │         GPT Backbone (frozen?)       │\n  │  Token + Position Embedding          │\n  │  Transformer Block × N               │\n  │  LayerNorm                           │\n  └──────────────────┬───────────────────┘\n                     │ hidden_states (B, T, n_embd)\n                     │ Extract last token: (B, n_embd)\n                     ▼\n              ┌──────────────┐\n              │   Dropout     │\n              └──────┬───────┘\n                     ▼\n              ┌──────────────┐\n              │  Linear       │\n              │ (n_embd → C)  │\n              └──────┬───────┘\n                     ▼\n              logits (B, C)" },
            { type: "code", language: "python", code: "def forward(self, input_ids, attention_mask=None):\n    # 跑到 layer norm 為止，不經過 LM head\n    x = self.backbone.tok_emb(input_ids) + self.backbone.pos_emb(...)\n    for block in self.backbone.blocks:\n        x = block(x)\n    hidden = self.backbone.ln_f(x)  # (B, T, n_embd)\n\n    if attention_mask is not None:\n        last_pos = attention_mask.sum(dim=1) - 1\n        pooled = hidden[torch.arange(B), last_pos]\n    else:\n        pooled = hidden[:, -1, :]\n\n    return self.classifier_head(self.dropout(pooled))" },
          ],
        },
        {
          title: "Feature Extraction vs Full Fine-Tuning",
          blocks: [
            { type: "table", headers: ["特性", "Feature Extraction（凍結）", "Full Fine-Tuning（解凍）"], rows: [
              ["可訓練參數", "僅分類頭", "整個模型"],
              ["訓練速度", "快", "慢"],
              ["記憶體需求", "低", "高"],
              ["資料需求", "少量即可", "需要更多資料"],
              ["效果上限", "受限於預訓練", "可適應特定領域"],
              ["過擬合風險", "低", "高"],
            ]},
            { type: "code", language: "python", code: "def freeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = False\n\ndef unfreeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = True" },
            { type: "callout", variant: "tip", text: "💡 講師心得：數據少時，凍結骨幹防止過擬合。數據多時，解凍讓模型完全適應新任務。實務上，先試 feature extraction——效果夠好就不需要冒過擬合的風險。這也是 Phase 7 LoRA 存在的意義：用極少的可訓練參數來折衷。" },
          ],
        },
        {
          title: "SpamDataset：資料準備流程",
          blocks: [
            { type: "list", ordered: true, items: [
              "讀取 CSV：使用 csv.DictReader 解析，提取 'text' 和 'label' 欄位",
              "Tokenize：用 tokenizer.encode(text) 將文本轉為 token ID 序列",
              "截斷：如果序列長度超過 max_length，截取前 max_length 個 token",
              "Padding：如果序列長度不足 max_length，在尾部補 0",
              "Attention Mask：生成 1/0 遮罩，1 表示真實 token，0 表示 padding",
            ]},
            { type: "code", language: "python", code: "def __getitem__(self, idx):\n    return {\n        'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),\n        'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),\n        'label': torch.tensor(self.labels[idx], dtype=torch.long),\n    }" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們用最後一個 token 的 hidden state 來做分類。如果輸入是 'This movie is great!'，最後一個 token 是 '!'（感嘆號）。感嘆號本身沒有情感含義，為什麼用它的表示向量來分類卻能得到好結果？（提示：想想 causal attention 的累積效應。）" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_classifier_init", title: "TODO 1: GPTClassifier.__init__", description: "初始化分類器：儲存 GPT backbone、建立 dropout 層和線性分類頭，並根據 freeze_backbone 參數凍結 backbone 參數。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["self.backbone = gpt_model", "self.dropout = nn.Dropout(dropout)", "self.classifier_head = nn.Linear(n_embd, n_classes)", "凍結：for param in self.backbone.parameters(): param.requires_grad = False"], pseudocode: "self.backbone = gpt_model\nself.dropout = nn.Dropout(dropout)\nn_embd = gpt_model.config.n_embd\nself.classifier_head = nn.Linear(n_embd, n_classes)\nif freeze_backbone:\n  for p in self.backbone.parameters():\n    p.requires_grad = False" },
        { id: "gpt_classifier_forward", title: "TODO 2: GPTClassifier.forward", description: "前向傳播：通過 backbone 取得隱藏狀態，提取最後一個 token 的表示，經過 dropout 和分類頭輸出 logits。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["取得 LM head 之前的隱藏狀態 (B, T, n_embd)", "如果有 attention_mask：last_pos = attention_mask.sum(dim=1) - 1", "最後經過 self.dropout → self.classifier_head"] },
        { id: "freeze_model", title: "TODO 3: freeze_backbone", description: "凍結 GPT backbone 的所有參數（requires_grad = False）。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["遍歷 model.backbone.parameters()", "設定 param.requires_grad = False"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = False" },
        { id: "unfreeze_model", title: "TODO 4: unfreeze_backbone", description: "解凍 GPT backbone 的所有參數（requires_grad = True）。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["遍歷 model.backbone.parameters()", "設定 param.requires_grad = True"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = True" },
        { id: "spam_dataset_init", title: "TODO 5: SpamDataset.__init__", description: "從 CSV 載入資料，對每筆文本做 tokenize、truncation、padding 並生成 attention mask。", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["csv.DictReader 讀取，欄位為 'text' 和 'label'", "ids = tokenizer.encode(text)[:max_length]", "padding：ids + [0] * (max_length - len(ids))", "mask：[1]*actual_len + [0]*padding_len"] },
        { id: "spam_dataset_getitem", title: "TODO 6: SpamDataset.__getitem__", description: "返回第 idx 筆樣本的字典，包含 input_ids、attention_mask、label 三個 torch.long tensor。", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["torch.tensor(..., dtype=torch.long)", "返回字典：{'input_ids': ..., 'attention_mask': ..., 'label': ...}"] },
      ],
      acceptanceCriteria: [
        "GPTClassifier 輸出 shape 為 (batch_size, n_classes)",
        "freeze_backbone=True 時 backbone 參數的 requires_grad 為 False",
        "classifier_head 的 requires_grad 始終為 True",
        "freeze_backbone() 和 unfreeze_backbone() 正確切換 requires_grad",
        "SpamDataset 正確讀取 CSV 並返回正確數量的樣本",
        "Padding 在序列尾部，attention_mask 正確標記真實 token 與 padding",
      ],
      references: [
        { title: "Improving Language Understanding by Generative Pre-Training", description: "GPT-1 論文 — 首次展示 generative pre-training + discriminative fine-tuning 的範式", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" },
        { title: "Universal Language Model Fine-tuning for Text Classification", description: "ULMFiT — 提出漸進式解凍等 fine-tuning 技巧", url: "https://arxiv.org/abs/1801.06146" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// zh-CN variants
// ═══════════════════════════════════════════════════════════════════

const phase5ContentZhCN: PhaseContent = {
  phaseId: 5,
  color: "#10B981",
  accent: "#34D399",
  lessons: [
    {
      phaseId: 5, lessonId: 1,
      title: "Generation Strategies",
      subtitle: "从 Greedy 到 Nucleus Sampling——掌握文本生成的各种解码策略",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 autoregressive generation 的核心循环",
        "实现 greedy decoding 并理解其局限性",
        "理解 temperature 参数如何重塑概率分布",
        "实现 top-k sampling 与 top-p (nucleus) sampling",
        "建立统一的 generate 接口整合所有策略",
      ],
      sections: [
        {
          title: "Autoregressive Generation Loop",
          blocks: [
            { type: "paragraph", text: "模型训练好了，现在到了最令人期待的时刻——让它开口「说话」。但你可能会发现一个奇怪的现象：同一个 prompt，用不同的生成策略，模型可能写出无聊的重复句子，也可能写出惊艳的创意文章。差别在哪里？全在今天要学的这几种 sampling 策略里。你在 ChatGPT 里调的那个 Temperature 滑块，背后就是这些数学。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：文本生成是你第一次看到模型「活过来」的时刻。前面四个 Phase 都是在训练——模型是被动的学生。从这个 Phase 开始，模型要主动「创作」了。但生成策略的选择比你想象的重要得多——同一个模型，用 greedy decoding 可能输出枯燥重复的文字，换成 nucleus sampling 就能写出流畅自然的文章。" },
            { type: "paragraph", text: "语言模型的生成过程本质上是一个自回归（autoregressive）循环：模型每次只生成一个 token，然后将这个新 token 加入输入序列，再用更新后的序列预测下一个 token。这个过程反复执行，直到达到指定的长度或遇到结束标记。" },
            { type: "paragraph", text: "关键观察：模型的 forward pass 输出的是整个词汇表上的 logits（未归一化的分数），而不是单一的 token。我们只关心最后一个位置的 logits——如何从这些 logits 中选择下一个 token，就是「解码策略」要解决的问题。" },
            { type: "diagram", content: "Autoregressive Generation Loop:\n\n  ┌─────────────────────────────────────────────────┐\n  │  Input: [The, cat, sat]                         │\n  │                                                 │\n  │  ┌──────────┐    logits[:, -1, :]    ┌────────┐ │\n  │  │ GPT Model│ ──────────────────────>│Decode  │ │\n  │  │ Forward  │    (vocab_size,)       │Strategy│ │\n  │  └──────────┘                        └───┬────┘ │\n  │                                          │      │\n  │       next_token = selected token        v      │\n  │  Input: [The, cat, sat, on]  <── append token   │\n  └─────────────────────────────────────────────────┘" },
            { type: "callout", variant: "info", text: "如果模型有 block_size（最大上下文长度）属性，当序列长度超过 block_size 时需要截断。只保留最后 block_size 个 token 作为输入，避免超出位置嵌入的范围。" },
            { type: "code", language: "python", code: "# Autoregressive generation 的通用骨架\nmodel.eval()\nwith torch.no_grad():\n    for _ in range(max_new_tokens):\n        # 截断到 block_size（如果需要）\n        idx_cond = idx[:, -block_size:] if hasattr(model, 'block_size') else idx\n        # Forward pass\n        logits, _ = model(idx_cond)\n        # 只取最后一个位置的 logits\n        next_logits = logits[:, -1, :]    # (B, vocab_size)\n        # ── 这里插入解码策略 ──\n        next_token = decode_strategy(next_logits)\n        # 将新 token 附加到序列\n        idx = torch.cat([idx, next_token], dim=1)" },
          ],
        },
        {
          title: "Greedy Decoding",
          blocks: [
            { type: "paragraph", text: "最简单的解码策略：每一步都选择概率最高的 token（argmax）。这保证了输出的确定性——同样的输入永远产生同样的输出。" },
            { type: "code", language: "python", code: "# Greedy decoding: 永远选最大的\nnext_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n# next_token shape: (B, 1)" },
            { type: "heading", level: 3, text: "Greedy Decoding 的问题" },
            { type: "list", ordered: true, items: [
              "重复性高：模型容易陷入重复循环，生成 'the the the...' 或重复相同的句子",
              "缺乏创意：永远走最安全的路线，无法产生惊喜或多样性",
              "全局次优：逐步的局部最优不保证整个序列的全局最优",
            ]},
            { type: "callout", variant: "tip", text: "💡 讲师心得：Greedy decoding 的问题不是它「太确定」，而是它「局部最优但全局最差」。就像下棋只看一步——每一步都选「当前最佳」，结果却走进死胡同。具体表现就是重复（'the the the the...'）——模型进入了一个高概率的循环，但没有全局规划能力跳出来。" },
          ],
        },
        {
          title: "Temperature Scaling",
          blocks: [
            { type: "paragraph", text: "Temperature 是控制生成随机性的最直观参数。其原理是在 softmax 之前将 logits 除以一个正数 T（temperature）：" },
            { type: "code", language: "python", code: "# Temperature scaling\nscaled_logits = logits / temperature\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "diagram", content: "Temperature 对概率分布的影响（原始 logits = [2.0, 1.0, 0.5, 0.1]）:\n\n  T=0.1 (极低)  ████████████████████████  token_0: 99.7%\n  T=0.5 (低)    ██████████████████        token_0: 73.1%\n  T=1.0 (标准)  ██████████████            token_0: 46.1%\n  T=2.0 (高)    █████████                 token_0: 33.0%" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：Temperature 的本质是在 softmax 之前对 logits 做缩放。T<1 时 logits 差异被放大→分布更尖锐→更确定；T>1 时差异被缩小→分布更平坦→更随机。T→0 退化为 greedy，T→∞ 退化为均匀分布。" },
            { type: "callout", variant: "warning", text: "Temperature = 0 会导致除以零！实现中需要特殊处理：当 temperature < 1e-8 时退回 greedy decoding（argmax）。" },
          ],
        },
        {
          title: "Top-k Sampling",
          blocks: [
            { type: "paragraph", text: "Temperature sampling 的一个问题是：即使在低温下，长尾中那些概率极低的 token 仍有可能被采样到。Top-k sampling 直接截断：只保留概率最高的 k 个 token，其余全部排除。" },
            { type: "code", language: "python", code: "# Top-k sampling\nscaled_logits = logits / temperature\ntop_k_values, _ = torch.topk(scaled_logits, k=k)\nthreshold = top_k_values[:, -1:]\nscaled_logits[scaled_logits < threshold] = float('-inf')\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "k = 1 等价于 greedy decoding。k = vocab_size 等价于纯 temperature sampling。实践中 k = 40~100 是常见的选择。" },
          ],
        },
        {
          title: "Top-p (Nucleus) Sampling",
          blocks: [
            { type: "paragraph", text: "Top-p sampling（又称 nucleus sampling）优雅地解决了 top-k 的固定截断问题。其核心思想是：动态选取最小的 token 集合，使得这些 token 的累积概率恰好超过门槛 p。" },
            { type: "code", language: "python", code: "# Top-p (nucleus) sampling\nscaled_logits = logits / temperature\nsorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)\ncumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)\nmask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p\nsorted_logits[mask] = float('-inf')\nscaled_logits.scatter_(1, sorted_indices, sorted_logits)\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：Top-p 比 top-k 更聪明的地方在于它能自适应。当模型很确定时，top-p=0.9 可能只保留 1-2 个 token。当模型不确定时，同样的 p=0.9 可能保留几十个 token。这就是为什么 GPT-4 默认用 top-p 而非 top-k。" },
          ],
        },
        {
          title: "策略比较与统一接口",
          blocks: [
            { type: "table", headers: ["策略", "参数", "确定性", "多样性", "适用场景"], rows: [
              ["Greedy", "无", "完全确定", "无", "代码补全、事实问答"],
              ["Temperature", "T ∈ (0, ∞)", "T→0 确定", "T↑ 增加", "创意写作（T=0.7~0.9）"],
              ["Top-k", "k ∈ [1, V]", "k=1 确定", "k↑ 增加", "一般文本生成（k=40~100）"],
              ["Top-p", "p ∈ (0, 1]", "p→0 确定", "p↑ 增加", "最佳通用选择（p=0.9~0.95）"],
            ]},
            { type: "code", language: "python", code: "def generate(model, tokenizer, prompt, max_new_tokens=50,\n             strategy='greedy', device='cpu', **kwargs):\n    ids = tokenizer.encode(prompt)\n    idx = torch.tensor([ids], dtype=torch.long, device=device)\n    strategies = {\n        'greedy': greedy_decode,\n        'temperature': temperature_sample,\n        'top_k': top_k_sample,\n        'top_p': top_p_sample,\n    }\n    fn = strategies[strategy]\n    output_ids = fn(model, idx, max_new_tokens, device=device, **kwargs)\n    return tokenizer.decode(output_ids[0].tolist())" },
            { type: "callout", variant: "quote", text: "🤔 思考题：在实际的 LLM API 中，你可以同时设定 temperature 和 top_p。是先做 temperature scaling 再做 top-p truncation，还是反过来？顺序不同结果一样吗？" },
          ],
        },
      ],
      exercises: [
        { id: "greedy_decode", title: "TODO 1: greedy_decode", description: "实现 greedy decoding——每一步选择 logits 最大的 token（argmax）。需处理 block_size 截断、model.eval() 与 torch.no_grad()。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["使用 model.eval() 和 torch.no_grad() 进行推理", "如果模型有 block_size 属性，截断 idx 到最后 block_size 个 token", "torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) 获取下一个 token"], pseudocode: "model.eval()\nwith torch.no_grad():\n  for _ in range(max_new_tokens):\n    idx_cond = idx[:, -block_size:] if hasattr(...) else idx\n    logits, _ = model(idx_cond)\n    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n    idx = torch.cat([idx, next_id], dim=1)\nreturn idx" },
        { id: "temperature_sample", title: "TODO 2: temperature_sample", description: "实现 temperature sampling——将 logits 除以 temperature 后做 softmax 再用 multinomial 采样。当 temperature ≈ 0 时退回 greedy。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["特殊处理 temperature < 1e-8：使用 argmax 代替 sampling", "probs = torch.softmax(logits / temperature, dim=-1)", "next_id = torch.multinomial(probs, num_samples=1)"] },
        { id: "top_k_sample", title: "TODO 3: top_k_sample", description: "实现 top-k sampling——只从概率最高的 k 个 token 中采样。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["k = min(k, logits.size(-1)) 防止越界", "torch.topk(logits, k) 获取前 k 大的值", "低于门槛的 logits 设为 float('-inf')"] },
        { id: "top_p_sample", title: "TODO 4: top_p_sample", description: "实现 top-p (nucleus) sampling——动态选取累积概率超过 p 的最小 token 集合。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["torch.sort(logits, descending=True) 降序排列", "cumsum 计算累积概率", "掩码需偏移一位以保留至少一个 token", "logits.scatter_(1, sorted_indices, sorted_logits) 恢复顺序"] },
        { id: "generate", title: "TODO 5: generate (统一接口)", description: "实现统一生成接口——接受文字 prompt，encode → 选择策略 → decode 回文字。", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["tokenizer.encode(prompt) 获取 token IDs", "建立策略字典分派", "tokenizer.decode(output_ids[0].tolist()) 解码回文字"] },
      ],
      acceptanceCriteria: [
        "greedy_decode 输出 shape 为 (1, T + max_new_tokens) 且完全确定性",
        "temperature_sample 在 temperature ≈ 0 时等价于 greedy decoding",
        "top_k_sample 在 k=1 时等价于 greedy decoding",
        "top_p_sample 在 p 极小时接近 greedy，p=1.0 时包含全部 token",
        "generate() 正确分派策略并返回字符串，非法策略抛出异常",
        "所有函数保留原始 input prefix 不变",
      ],
      references: [
        { title: "The Curious Case of Neural Text Degeneration", description: "Holtzman et al. 2019 — 提出 nucleus (top-p) sampling，分析了 greedy / beam search 的退化问题", url: "https://arxiv.org/abs/1904.09751" },
        { title: "How to generate text: using different decoding methods for language generation with Transformers", description: "HuggingFace 官方博客，图文并茂地解释各种解码策略", url: "https://huggingface.co/blog/how-to-generate" },
      ],
    },
  ],
};

const phase6ContentZhCN: PhaseContent = {
  phaseId: 6,
  color: "#06B6D4",
  accent: "#22D3EE",
  lessons: [
    {
      phaseId: 6, lessonId: 1,
      title: "Classification with GPT",
      subtitle: "从语言模型到分类器——用 GPT 的隐藏表示进行下游任务",
      type: "concept",
      duration: "50 min",
      objectives: [
        "理解为什么使用最后一个 token 的表示做分类",
        "实现 GPTClassifier：在 GPT backbone 上加入分类头",
        "掌握 feature extraction（冻结 backbone）与 full fine-tuning 两种策略",
        "实现 SpamDataset：从 CSV 加载、tokenize、padding 和 attention mask",
        "理解 freeze/unfreeze 对梯度流的影响",
      ],
      sections: [
        {
          title: "为什么用最后一个 Token 的表示？",
          blocks: [
            { type: "paragraph", text: "到目前为止，我们的 GPT 模型只会一件事：生成文本。但在现实世界中，很多任务不是「写作文」，而是「做判断」——这封邮件是不是垃圾邮件？这条评论是正面还是负面？今天我们要把一个「说书人」（生成模型）改造成「裁判」（分类模型）。你会惊讶地发现，这个改造出奇地简单——只需要在模型顶部加一层就够了。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：把一个生成式模型改成分类器，听起来有点违直觉。关键在于预训练过程中，模型已经学到了丰富的语言理解能力。分类微调只是在模型顶部加一个「分类头」，利用这些已有的语言知识来做决策。这就是迁移学习的威力。" },
            { type: "paragraph", text: "GPT 是一个 causal（自回归）语言模型——每个 token 只能 attend 到它自己和之前的 token。这意味着最后一个 token 的隐藏状态是唯一「看过整个输入序列」的表示，它汇集了整个输入的语义信息。" },
            { type: "diagram", content: "GPT Causal Attention — 信息流向:\n\n  Token:    [This]  [is]  [spam]  [!]\n  Attend:   [自身] [前2] [前3]  [全部]\n  Info:      ●       ●●     ●●●    ●●●●\n                                   ▲\n                          最后一个 token 包含\n                          整个序列的信息\n                                   ▼\n                          [Classification Head]\n                                   ▼\n                          [spam / not spam]" },
            { type: "callout", variant: "info", text: "如果序列有 padding，最后一个「真实」token 才是有意义的。此时需要 attention_mask 来找到最后一个非 padding 位置：last_pos = attention_mask.sum(dim=1) - 1。" },
          ],
        },
        {
          title: "GPTClassifier 架构",
          blocks: [
            { type: "diagram", content: "GPTClassifier Architecture:\n\n  input_ids (B, T)\n       │\n       ▼\n  ┌──────────────────────────────────────┐\n  │         GPT Backbone (frozen?)       │\n  │  Token + Position Embedding          │\n  │  Transformer Block × N               │\n  │  LayerNorm                           │\n  └──────────────────┬───────────────────┘\n                     │ hidden_states (B, T, n_embd)\n                     │ 提取最后 token: (B, n_embd)\n                     ▼\n              ┌──────────────┐\n              │   Dropout     │\n              └──────┬───────┘\n                     ▼\n              ┌──────────────┐\n              │  Linear       │\n              │ (n_embd → C)  │\n              └──────┬───────┘\n                     ▼\n              logits (B, C)" },
            { type: "code", language: "python", code: "def forward(self, input_ids, attention_mask=None):\n    # 运行到 layer norm 为止，不经过 LM head\n    x = self.backbone.tok_emb(input_ids) + self.backbone.pos_emb(...)\n    for block in self.backbone.blocks:\n        x = block(x)\n    hidden = self.backbone.ln_f(x)  # (B, T, n_embd)\n\n    if attention_mask is not None:\n        last_pos = attention_mask.sum(dim=1) - 1\n        pooled = hidden[torch.arange(B), last_pos]\n    else:\n        pooled = hidden[:, -1, :]\n\n    return self.classifier_head(self.dropout(pooled))" },
          ],
        },
        {
          title: "Feature Extraction vs Full Fine-Tuning",
          blocks: [
            { type: "table", headers: ["特性", "Feature Extraction（冻结）", "Full Fine-Tuning（解冻）"], rows: [
              ["可训练参数", "仅分类头", "整个模型"],
              ["训练速度", "快", "慢"],
              ["内存需求", "低", "高"],
              ["数据需求", "少量即可", "需要更多数据"],
              ["效果上限", "受限于预训练", "可适应特定领域"],
              ["过拟合风险", "低", "高"],
            ]},
            { type: "code", language: "python", code: "def freeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = False\n\ndef unfreeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = True" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：数据少时，冻结骨干防止过拟合。数据多时，解冻让模型完全适应新任务。实践中，先试 feature extraction——效果够好就不需要冒过拟合的风险。这也是 Phase 7 LoRA 存在的意义：用极少的可训练参数来折中。" },
          ],
        },
        {
          title: "SpamDataset：数据准备流程",
          blocks: [
            { type: "list", ordered: true, items: [
              "读取 CSV：使用 csv.DictReader 解析，提取 'text' 和 'label' 字段",
              "Tokenize：用 tokenizer.encode(text) 将文本转为 token ID 序列",
              "截断：如果序列长度超过 max_length，截取前 max_length 个 token",
              "Padding：如果序列长度不足 max_length，在尾部补 0",
              "Attention Mask：生成 1/0 掩码，1 表示真实 token，0 表示 padding",
            ]},
            { type: "code", language: "python", code: "def __getitem__(self, idx):\n    return {\n        'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),\n        'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),\n        'label': torch.tensor(self.labels[idx], dtype=torch.long),\n    }" },
            { type: "callout", variant: "quote", text: "🤔 思考题：我们用最后一个 token 的 hidden state 来做分类。如果输入是 'This movie is great!'，最后一个 token 是 '!'（感叹号）。感叹号本身没有情感含义，为什么用它的表示向量来分类却能得到好结果？（提示：想想 causal attention 的累积效应。）" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_classifier_init", title: "TODO 1: GPTClassifier.__init__", description: "初始化分类器：存储 GPT backbone、建立 dropout 层和线性分类头，并根据 freeze_backbone 参数冻结 backbone 参数。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["self.backbone = gpt_model", "self.dropout = nn.Dropout(dropout)", "self.classifier_head = nn.Linear(n_embd, n_classes)", "冻结：for param in self.backbone.parameters(): param.requires_grad = False"], pseudocode: "self.backbone = gpt_model\nself.dropout = nn.Dropout(dropout)\nn_embd = gpt_model.config.n_embd\nself.classifier_head = nn.Linear(n_embd, n_classes)\nif freeze_backbone:\n  for p in self.backbone.parameters():\n    p.requires_grad = False" },
        { id: "gpt_classifier_forward", title: "TODO 2: GPTClassifier.forward", description: "前向传播：通过 backbone 获取隐藏状态，提取最后一个 token 的表示，经过 dropout 和分类头输出 logits。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["获取 LM head 之前的隐藏状态 (B, T, n_embd)", "如果有 attention_mask：last_pos = attention_mask.sum(dim=1) - 1", "最后经过 self.dropout → self.classifier_head"] },
        { id: "freeze_model", title: "TODO 3: freeze_backbone", description: "冻结 GPT backbone 的所有参数（requires_grad = False）。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["遍历 model.backbone.parameters()", "设定 param.requires_grad = False"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = False" },
        { id: "unfreeze_model", title: "TODO 4: unfreeze_backbone", description: "解冻 GPT backbone 的所有参数（requires_grad = True）。", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["遍历 model.backbone.parameters()", "设定 param.requires_grad = True"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = True" },
        { id: "spam_dataset_init", title: "TODO 5: SpamDataset.__init__", description: "从 CSV 加载数据，对每条文本做 tokenize、truncation、padding 并生成 attention mask。", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["csv.DictReader 读取，字段为 'text' 和 'label'", "ids = tokenizer.encode(text)[:max_length]", "padding：ids + [0] * (max_length - len(ids))", "mask：[1]*actual_len + [0]*padding_len"] },
        { id: "spam_dataset_getitem", title: "TODO 6: SpamDataset.__getitem__", description: "返回第 idx 条样本的字典，包含 input_ids、attention_mask、label 三个 torch.long tensor。", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["torch.tensor(..., dtype=torch.long)", "返回字典：{'input_ids': ..., 'attention_mask': ..., 'label': ...}"] },
      ],
      acceptanceCriteria: [
        "GPTClassifier 输出 shape 为 (batch_size, n_classes)",
        "freeze_backbone=True 时 backbone 参数的 requires_grad 为 False",
        "classifier_head 的 requires_grad 始终为 True",
        "freeze_backbone() 和 unfreeze_backbone() 正确切换 requires_grad",
        "SpamDataset 正确读取 CSV 并返回正确数量的样本",
        "Padding 在序列尾部，attention_mask 正确标记真实 token 与 padding",
      ],
      references: [
        { title: "Improving Language Understanding by Generative Pre-Training", description: "GPT-1 论文 — 首次展示 generative pre-training + discriminative fine-tuning 的范式", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" },
        { title: "Universal Language Model Fine-tuning for Text Classification", description: "ULMFiT — 提出渐进式解冻等 fine-tuning 技巧", url: "https://arxiv.org/abs/1801.06146" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// English variants
// ═══════════════════════════════════════════════════════════════════

const phase5ContentEn: PhaseContent = {
  phaseId: 5,
  color: "#10B981",
  accent: "#34D399",
  lessons: [
    {
      phaseId: 5, lessonId: 1,
      title: "Generation Strategies",
      subtitle: "From Greedy to Nucleus Sampling — Mastering Text Decoding Strategies",
      type: "concept",
      duration: "60 min",
      objectives: [
        "Understand the core autoregressive generation loop",
        "Implement greedy decoding and understand its limitations",
        "Understand how the temperature parameter reshapes probability distributions",
        "Implement top-k sampling and top-p (nucleus) sampling",
        "Build a unified generate interface integrating all strategies",
      ],
      sections: [
        {
          title: "Autoregressive Generation Loop",
          blocks: [
            { type: "paragraph", text: "The model is trained — now comes the most exciting part: making it talk. But here's something curious: give the same prompt to two different decoding strategies, and one might produce boring repetitive text while the other writes something genuinely creative. What's the difference? It all comes down to the sampling strategies we'll cover today. That Temperature slider you've seen in ChatGPT? This is the math behind it." },
            { type: "callout", variant: "quote", text: "Instructor's Note: Text generation is the moment you first see the model \"come alive.\" The previous four phases were all about training — the model was a passive student. Starting here, the model actively creates. But the choice of generation strategy matters more than you'd expect — the same model with greedy decoding might output dull, repetitive text, while nucleus sampling produces something fluent and natural." },
            { type: "paragraph", text: "At its core, language model generation is an autoregressive loop: the model generates one token at a time, appends it to the input sequence, then uses the updated sequence to predict the next token. This repeats until reaching the target length or hitting an end-of-sequence token." },
            { type: "paragraph", text: "Key insight: the model's forward pass outputs logits (unnormalized scores) over the entire vocabulary — not a single token. We only care about the logits at the last position. How we choose the next token from those logits is what decoding strategies are all about." },
            { type: "diagram", content: "Autoregressive Generation Loop:\n\n  ┌─────────────────────────────────────────────────┐\n  │  Input: [The, cat, sat]                         │\n  │                                                 │\n  │  ┌──────────┐    logits[:, -1, :]    ┌────────┐ │\n  │  │ GPT Model│ ──────────────────────>│Decode  │ │\n  │  │ Forward  │    (vocab_size,)       │Strategy│ │\n  │  └──────────┘                        └───┬────┘ │\n  │                                          │      │\n  │       next_token = selected token        v      │\n  │  Input: [The, cat, sat, on]  <── append token   │\n  └─────────────────────────────────────────────────┘" },
            { type: "callout", variant: "info", text: "If the model has a block_size attribute (maximum context length), truncate the input when the sequence exceeds that limit. Keep only the last block_size tokens to stay within the positional embedding range." },
            { type: "code", language: "python", code: "# General skeleton for autoregressive generation\nmodel.eval()\nwith torch.no_grad():\n    for _ in range(max_new_tokens):\n        # Truncate to block_size if needed\n        idx_cond = idx[:, -block_size:] if hasattr(model, 'block_size') else idx\n        # Forward pass\n        logits, _ = model(idx_cond)\n        # Take logits at the last position only\n        next_logits = logits[:, -1, :]    # (B, vocab_size)\n        # ── Insert decoding strategy here ──\n        next_token = decode_strategy(next_logits)\n        # Append the new token\n        idx = torch.cat([idx, next_token], dim=1)" },
          ],
        },
        {
          title: "Greedy Decoding",
          blocks: [
            { type: "paragraph", text: "The simplest strategy: always pick the highest-probability token (argmax). This makes generation completely deterministic — the same input always produces the same output." },
            { type: "code", language: "python", code: "# Greedy decoding: always pick the winner\nnext_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n# next_token shape: (B, 1)" },
            { type: "heading", level: 3, text: "The Problems with Greedy Decoding" },
            { type: "list", ordered: true, items: [
              "High repetition: the model easily falls into loops, generating 'the the the...' or repeating the same sentences",
              "Lacks creativity: always takes the safest route, no surprises or diversity",
              "Globally suboptimal: local greedy choices don't guarantee a globally optimal sequence",
            ]},
            { type: "callout", variant: "tip", text: "Instructor's Note: Greedy decoding's problem isn't that it's \"too certain\" — it's that it's \"locally optimal but globally terrible.\" It's like playing chess one move at a time, always choosing the immediate best move, then walking into a trap. The symptom is repetition ('the the the...') — the model gets stuck in a high-probability loop with no global planning to break out." },
          ],
        },
        {
          title: "Temperature Scaling",
          blocks: [
            { type: "paragraph", text: "Temperature is the most intuitive parameter for controlling generation randomness. The idea: divide logits by a positive number T (temperature) before the softmax." },
            { type: "code", language: "python", code: "# Temperature scaling\nscaled_logits = logits / temperature\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "diagram", content: "Effect of temperature on probability distribution (logits = [2.0, 1.0, 0.5, 0.1]):\n\n  T=0.1 (very low)  ████████████████████████  token_0: 99.7%\n  T=0.5 (low)       ██████████████████        token_0: 73.1%\n  T=1.0 (standard)  ██████████████            token_0: 46.1%\n  T=2.0 (high)      █████████                 token_0: 33.0%" },
            { type: "callout", variant: "quote", text: "Instructor's Note: Temperature scales logits before the softmax. T<1 amplifies differences → sharper distribution → more confident. T>1 shrinks differences → flatter distribution → more random. T→0 degenerates to greedy; T→∞ degenerates to uniform random." },
            { type: "callout", variant: "warning", text: "Temperature = 0 causes division by zero! Handle this in your implementation: when temperature < 1e-8, fall back to greedy decoding (argmax)." },
          ],
        },
        {
          title: "Top-k Sampling",
          blocks: [
            { type: "paragraph", text: "One issue with temperature sampling: even at low temperatures, the long tail of very-low-probability tokens can still be sampled. Top-k sampling cuts it off directly: keep only the k highest-probability tokens and discard the rest." },
            { type: "code", language: "python", code: "# Top-k sampling\nscaled_logits = logits / temperature\ntop_k_values, _ = torch.topk(scaled_logits, k=k)\nthreshold = top_k_values[:, -1:]\nscaled_logits[scaled_logits < threshold] = float('-inf')\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "k=1 is equivalent to greedy decoding. k=vocab_size is equivalent to pure temperature sampling. In practice, k=40~100 is a common choice." },
          ],
        },
        {
          title: "Top-p (Nucleus) Sampling",
          blocks: [
            { type: "paragraph", text: "Top-p sampling (a.k.a. nucleus sampling) elegantly solves top-k's fixed-cutoff problem. The idea: dynamically select the smallest set of tokens whose cumulative probability just exceeds threshold p." },
            { type: "code", language: "python", code: "# Top-p (nucleus) sampling\nscaled_logits = logits / temperature\nsorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)\ncumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)\nmask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p\nsorted_logits[mask] = float('-inf')\nscaled_logits.scatter_(1, sorted_indices, sorted_logits)\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "Instructor's Note: Top-p is smarter than top-k because it's adaptive. When the model is very confident, p=0.9 might keep only 1-2 tokens. When the model is uncertain, the same p=0.9 might keep dozens. Top-k can't do this — k=50 always picks 50 tokens regardless of model confidence. That's why GPT-4 defaults to top-p rather than top-k." },
          ],
        },
        {
          title: "Strategy Comparison & Unified Interface",
          blocks: [
            { type: "table", headers: ["Strategy", "Parameters", "Determinism", "Diversity", "Best For"], rows: [
              ["Greedy", "none", "fully deterministic", "none", "code completion, factual Q&A"],
              ["Temperature", "T ∈ (0, ∞)", "T→0 deterministic", "T↑ more", "creative writing (T=0.7~0.9)"],
              ["Top-k", "k ∈ [1, V]", "k=1 deterministic", "k↑ more", "general text (k=40~100)"],
              ["Top-p", "p ∈ (0, 1]", "p→0 deterministic", "p↑ more", "best all-purpose choice (p=0.9~0.95)"],
            ]},
            { type: "code", language: "python", code: "def generate(model, tokenizer, prompt, max_new_tokens=50,\n             strategy='greedy', device='cpu', **kwargs):\n    ids = tokenizer.encode(prompt)\n    idx = torch.tensor([ids], dtype=torch.long, device=device)\n    strategies = {\n        'greedy': greedy_decode,\n        'temperature': temperature_sample,\n        'top_k': top_k_sample,\n        'top_p': top_p_sample,\n    }\n    fn = strategies[strategy]\n    output_ids = fn(model, idx, max_new_tokens, device=device, **kwargs)\n    return tokenizer.decode(output_ids[0].tolist())" },
            { type: "callout", variant: "quote", text: "Think About It: Real LLM APIs let you set both temperature and top_p simultaneously. How do they interact? Is temperature scaling applied before or after top-p truncation? Does the order matter?" },
          ],
        },
      ],
      exercises: [
        { id: "greedy_decode", title: "TODO 1: greedy_decode", description: "Implement greedy decoding — at each step pick the highest-logit token (argmax). Handle block_size truncation, model.eval(), and torch.no_grad().", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["Use model.eval() and torch.no_grad() for inference", "If model has block_size attribute, truncate idx to last block_size tokens", "torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) gets the next token"], pseudocode: "model.eval()\nwith torch.no_grad():\n  for _ in range(max_new_tokens):\n    idx_cond = idx[:, -block_size:] if hasattr(...) else idx\n    logits, _ = model(idx_cond)\n    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n    idx = torch.cat([idx, next_id], dim=1)\nreturn idx" },
        { id: "temperature_sample", title: "TODO 2: temperature_sample", description: "Implement temperature sampling — divide logits by temperature, apply softmax, then sample with multinomial. Fall back to greedy when temperature ≈ 0.", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["Handle temperature < 1e-8: use argmax instead of sampling", "probs = torch.softmax(logits / temperature, dim=-1)", "next_id = torch.multinomial(probs, num_samples=1)"] },
        { id: "top_k_sample", title: "TODO 3: top_k_sample", description: "Implement top-k sampling — sample only from the k highest-probability tokens.", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["k = min(k, logits.size(-1)) to avoid out-of-bounds", "torch.topk(logits, k) gets the top-k values", "Set logits below threshold to float('-inf')"] },
        { id: "top_p_sample", title: "TODO 4: top_p_sample", description: "Implement top-p (nucleus) sampling — dynamically select the smallest token set with cumulative probability exceeding p.", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["torch.sort(logits, descending=True) sort in descending order", "cumsum to compute cumulative probabilities", "Shift mask by one to always keep at least one token", "logits.scatter_(1, sorted_indices, sorted_logits) to restore order"] },
        { id: "generate", title: "TODO 5: generate (unified interface)", description: "Implement the unified generation interface — take a text prompt, encode → dispatch strategy → decode back to text.", labFile: "labs/phase5_generation/phase_5/generate.py", hints: ["tokenizer.encode(prompt) for token IDs", "Build a strategy dispatch dict", "tokenizer.decode(output_ids[0].tolist()) to decode back to text"] },
      ],
      acceptanceCriteria: [
        "greedy_decode output shape is (1, T + max_new_tokens) and fully deterministic",
        "temperature_sample is equivalent to greedy when temperature ≈ 0",
        "top_k_sample is equivalent to greedy when k=1",
        "top_p_sample approaches greedy at very small p, includes all tokens at p=1.0",
        "generate() correctly dispatches strategies and returns a string; invalid strategy raises an exception",
        "All functions preserve the original input prefix unchanged",
      ],
      references: [
        { title: "The Curious Case of Neural Text Degeneration", description: "Holtzman et al. 2019 — Introduces nucleus (top-p) sampling and analyzes degeneration in greedy/beam search", url: "https://arxiv.org/abs/1904.09751" },
        { title: "How to generate text: using different decoding methods for language generation with Transformers", description: "HuggingFace blog — visual walkthrough of all major decoding strategies", url: "https://huggingface.co/blog/how-to-generate" },
      ],
    },
  ],
};

const phase6ContentEn: PhaseContent = {
  phaseId: 6,
  color: "#06B6D4",
  accent: "#22D3EE",
  lessons: [
    {
      phaseId: 6, lessonId: 1,
      title: "Classification with GPT",
      subtitle: "From Language Model to Classifier — Using GPT's Hidden Representations for Downstream Tasks",
      type: "concept",
      duration: "50 min",
      objectives: [
        "Understand why we use the last token's representation for classification",
        "Implement GPTClassifier: add a classification head on top of a GPT backbone",
        "Master feature extraction (frozen backbone) vs full fine-tuning strategies",
        "Implement SpamDataset: load from CSV, tokenize, pad, and generate attention masks",
        "Understand how freeze/unfreeze affects gradient flow",
      ],
      sections: [
        {
          title: "Why the Last Token's Representation?",
          blocks: [
            { type: "paragraph", text: "So far our GPT model does exactly one thing: generate text. But in the real world, many tasks aren't about \"writing\" — they're about \"judging.\" Is this email spam? Is this review positive or negative? Today we're going to transform a \"storyteller\" (generative model) into a \"judge\" (classification model). You'll be surprised how simple it is — just one extra layer on top." },
            { type: "callout", variant: "quote", text: "Instructor's Note: Turning a generative model into a classifier sounds counterintuitive — GPT is for writing, not spam detection. The key is that during pre-training, the model already learned rich language understanding. Fine-tuning for classification just adds a \"classification head\" on top, leveraging all that existing linguistic knowledge. This is the power of transfer learning." },
            { type: "paragraph", text: "GPT is a causal (autoregressive) language model — each token can only attend to itself and preceding tokens. This means the last token's hidden state is the only representation that has \"seen the entire input sequence,\" making it a natural aggregation of the whole input's semantics." },
            { type: "diagram", content: "GPT Causal Attention — Information Flow:\n\n  Token:    [This]  [is]  [spam]  [!]\n  Attends:  [self] [2 prev] [3 prev] [all]\n  Info:      ●       ●●     ●●●    ●●●●\n                                   ▲\n                          Last token contains\n                          entire sequence info\n                                   ▼\n                          [Classification Head]\n                                   ▼\n                          [spam / not spam]" },
            { type: "callout", variant: "info", text: "If the sequence has padding, only the last \"real\" token is meaningful. Use attention_mask to find the last non-padding position: last_pos = attention_mask.sum(dim=1) - 1." },
          ],
        },
        {
          title: "GPTClassifier Architecture",
          blocks: [
            { type: "diagram", content: "GPTClassifier Architecture:\n\n  input_ids (B, T)\n       │\n       ▼\n  ┌──────────────────────────────────────┐\n  │         GPT Backbone (frozen?)       │\n  │  Token + Position Embedding          │\n  │  Transformer Block × N               │\n  │  LayerNorm                           │\n  └──────────────────┬───────────────────┘\n                     │ hidden_states (B, T, n_embd)\n                     │ Extract last token: (B, n_embd)\n                     ▼\n              ┌──────────────┐\n              │   Dropout     │\n              └──────┬───────┘\n                     ▼\n              ┌──────────────┐\n              │  Linear       │\n              │ (n_embd → C)  │\n              └──────┬───────┘\n                     ▼\n              logits (B, C)" },
            { type: "code", language: "python", code: "def forward(self, input_ids, attention_mask=None):\n    # Run through the backbone up to the final layer norm, skipping LM head\n    x = self.backbone.tok_emb(input_ids) + self.backbone.pos_emb(...)\n    for block in self.backbone.blocks:\n        x = block(x)\n    hidden = self.backbone.ln_f(x)  # (B, T, n_embd)\n\n    if attention_mask is not None:\n        last_pos = attention_mask.sum(dim=1) - 1\n        pooled = hidden[torch.arange(B), last_pos]\n    else:\n        pooled = hidden[:, -1, :]\n\n    return self.classifier_head(self.dropout(pooled))" },
          ],
        },
        {
          title: "Feature Extraction vs Full Fine-Tuning",
          blocks: [
            { type: "table", headers: ["Property", "Feature Extraction (frozen)", "Full Fine-Tuning (unfrozen)"], rows: [
              ["Trainable params", "Only the classification head", "The entire model"],
              ["Training speed", "Fast", "Slow"],
              ["Memory usage", "Low", "High (stores all gradients)"],
              ["Data required", "Small dataset works", "Needs more data to avoid overfitting"],
              ["Performance ceiling", "Limited by pre-training", "Can adapt to specific domain"],
              ["Overfitting risk", "Low", "High (especially with small datasets)"],
            ]},
            { type: "code", language: "python", code: "def freeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = False\n\ndef unfreeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = True" },
            { type: "callout", variant: "tip", text: "Instructor's Note: Feature extraction vs full fine-tuning is a bias-variance tradeoff. With little data, freeze the backbone to prevent overfitting. With lots of data, unfreeze to fully adapt. In practice, start with feature extraction — if it's good enough, no need to risk overfitting. This is also why LoRA (Phase 7) exists: fine-tuning quality with minimal trainable parameters." },
          ],
        },
        {
          title: "SpamDataset: Data Preparation Pipeline",
          blocks: [
            { type: "list", ordered: true, items: [
              "Read CSV: use csv.DictReader to parse, extract 'text' and 'label' columns",
              "Tokenize: use tokenizer.encode(text) to convert text to token ID sequences",
              "Truncate: if the sequence exceeds max_length, keep the first max_length tokens",
              "Pad: if the sequence is shorter than max_length, append 0s at the end",
              "Attention Mask: generate a 1/0 mask — 1 for real tokens, 0 for padding",
            ]},
            { type: "code", language: "python", code: "def __getitem__(self, idx):\n    return {\n        'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),\n        'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),\n        'label': torch.tensor(self.labels[idx], dtype=torch.long),\n    }" },
            { type: "callout", variant: "quote", text: "Think About It: We use the last token's hidden state for classification. If the input is 'This movie is great!', the last token is '!' — an exclamation mark with no inherent sentiment. So why does classifying based on its representation vector actually work well? (Hint: think about causal attention's cumulative effect.)" },
          ],
        },
      ],
      exercises: [
        { id: "gpt_classifier_init", title: "TODO 1: GPTClassifier.__init__", description: "Initialize the classifier: store the GPT backbone, create a dropout layer and linear classification head, and freeze the backbone if specified.", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["self.backbone = gpt_model", "self.dropout = nn.Dropout(dropout)", "self.classifier_head = nn.Linear(n_embd, n_classes)", "Freeze: for param in self.backbone.parameters(): param.requires_grad = False"], pseudocode: "self.backbone = gpt_model\nself.dropout = nn.Dropout(dropout)\nn_embd = gpt_model.config.n_embd\nself.classifier_head = nn.Linear(n_embd, n_classes)\nif freeze_backbone:\n  for p in self.backbone.parameters():\n    p.requires_grad = False" },
        { id: "gpt_classifier_forward", title: "TODO 2: GPTClassifier.forward", description: "Forward pass: get hidden states from backbone, extract last token representation, pass through dropout and classification head.", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["Get hidden states before LM head: (B, T, n_embd)", "With attention_mask: last_pos = attention_mask.sum(dim=1) - 1", "Final: self.dropout → self.classifier_head"] },
        { id: "freeze_model", title: "TODO 3: freeze_backbone", description: "Freeze all GPT backbone parameters (requires_grad = False).", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["Iterate model.backbone.parameters()", "Set param.requires_grad = False"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = False" },
        { id: "unfreeze_model", title: "TODO 4: unfreeze_backbone", description: "Unfreeze all GPT backbone parameters (requires_grad = True) to enable full fine-tuning.", labFile: "labs/phase6_classification/phase_6/classifier.py", hints: ["Iterate model.backbone.parameters()", "Set param.requires_grad = True"], pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = True" },
        { id: "spam_dataset_init", title: "TODO 5: SpamDataset.__init__", description: "Load data from CSV, tokenize each text, apply truncation and padding, and generate attention masks.", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["csv.DictReader to read, fields are 'text' and 'label'", "ids = tokenizer.encode(text)[:max_length]", "Padding: ids + [0] * (max_length - len(ids))", "Mask: [1]*actual_len + [0]*padding_len"] },
        { id: "spam_dataset_getitem", title: "TODO 6: SpamDataset.__getitem__", description: "Return the idx-th sample as a dict with input_ids, attention_mask, and label as torch.long tensors.", labFile: "labs/phase6_classification/phase_6/dataset.py", hints: ["torch.tensor(..., dtype=torch.long)", "Return dict: {'input_ids': ..., 'attention_mask': ..., 'label': ...}"] },
      ],
      acceptanceCriteria: [
        "GPTClassifier output shape is (batch_size, n_classes)",
        "freeze_backbone=True makes backbone parameters have requires_grad=False",
        "classifier_head always has requires_grad=True",
        "freeze_backbone() and unfreeze_backbone() correctly toggle requires_grad",
        "SpamDataset correctly reads CSV and returns the right number of samples",
        "Padding is at the end, attention_mask correctly marks real tokens vs padding",
      ],
      references: [
        { title: "Improving Language Understanding by Generative Pre-Training", description: "GPT-1 paper — first demonstrates generative pre-training + discriminative fine-tuning paradigm", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" },
        { title: "Universal Language Model Fine-tuning for Text Classification", description: "ULMFiT — introduces gradual unfreezing and other fine-tuning tricks", url: "https://arxiv.org/abs/1801.06146" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// Locale dispatch
// ═══════════════════════════════════════════════════════════════════

const phase5Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase5ContentZhTW,
  "zh-CN": phase5ContentZhCN,
  "en": phase5ContentEn,
};

const phase6Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase6ContentZhTW,
  "zh-CN": phase6ContentZhCN,
  "en": phase6ContentEn,
};

export function getPhase5Content(locale: Locale): PhaseContent { return phase5Map[locale]; }
export function getPhase6Content(locale: Locale): PhaseContent { return phase6Map[locale]; }
