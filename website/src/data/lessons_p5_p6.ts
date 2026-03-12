import type { PhaseContent } from "./types";

// ═══════════════════════════════════════════════════════════════════
// Phase 5: Text Generation
// ═══════════════════════════════════════════════════════════════════

export const phase5Content: PhaseContent = {
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
        // ── Section 1: Autoregressive Generation ──
        {
          title: "Autoregressive Generation Loop",
          blocks: [
            { type: "paragraph", text: "模型訓練好了，現在到了最令人期待的時刻——讓它開口「說話」。但你可能會發現一個奇怪的現象：同一個 prompt，用不同的生成策略，模型可能寫出無聊的重複句子，也可能寫出驚艷的創意文章。差別在哪裡？全在今天要學的這幾種 sampling 策略裡。你在 ChatGPT 裡調的那個 Temperature 滑桿，背後就是這些數學。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：文本生成是你第一次看到模型「活過來」的時刻。前面四個 Phase 都是在訓練——模型是被動的學生。從這個 Phase 開始，模型要主動「創作」了。但生成策略的選擇比你想像的重要得多——同一個模型，用 greedy decoding 可能輸出枯燥重複的文字，換成 nucleus sampling 就能寫出流暢自然的文章。" },
            { type: "paragraph", text: "語言模型的生成過程本質上是一個自回歸（autoregressive）迴圈：模型每次只生成一個 token，然後將這個新 token 加入輸入序列，再用更新後的序列預測下一個 token。這個過程反覆執行，直到達到指定的長度或遇到結束標記。" },
            { type: "paragraph", text: "關鍵觀察：模型的 forward pass 輸出的是整個詞彙表上的 logits（未歸一化的分數），而不是單一的 token。我們只關心最後一個位置的 logits——因為前面位置的預測已經在訓練時使用過了。如何從這些 logits 中選擇下一個 token，就是「解碼策略」要解決的問題。" },
            { type: "diagram", content: "Autoregressive Generation Loop:\n\n  ┌─────────────────────────────────────────────────┐\n  │  Input: [The, cat, sat]                         │\n  │                                                 │\n  │  ┌──────────┐    logits[:, -1, :]    ┌────────┐ │\n  │  │ GPT Model│ ──────────────────────>│Decode  │ │\n  │  │ Forward  │    (vocab_size,)       │Strategy│ │\n  │  └──────────┘                        └───┬────┘ │\n  │                                          │      │\n  │       next_token = selected token        │      │\n  │                                          v      │\n  │  Input: [The, cat, sat, on]  <── append token   │\n  │                                                 │\n  │  ┌──────────┐    logits[:, -1, :]    ┌────────┐ │\n  │  │ GPT Model│ ──────────────────────>│Decode  │ │\n  │  │ Forward  │                        │Strategy│ │\n  │  └──────────┘                        └───┬────┘ │\n  │                                          │      │\n  │  Input: [The, cat, sat, on, the] <─ append      │\n  │                                                 │\n  │  ... repeat max_new_tokens times ...            │\n  └─────────────────────────────────────────────────┘" },
            { type: "callout", variant: "info", text: "如果模型有 block_size（最大上下文長度）屬性，當序列長度超過 block_size 時需要截斷。只保留最後 block_size 個 token 作為輸入，避免超出位置嵌入的範圍。" },
            { type: "code", language: "python", code: "# Autoregressive generation 的通用骨架\nmodel.eval()\nwith torch.no_grad():\n    for _ in range(max_new_tokens):\n        # 截斷到 block_size（如果需要）\n        idx_cond = idx[:, -block_size:] if hasattr(model, 'block_size') else idx\n        # Forward pass\n        logits, _ = model(idx_cond)\n        # 只取最後一個位置的 logits\n        next_logits = logits[:, -1, :]    # (B, vocab_size)\n        # ── 這裡插入解碼策略 ──\n        next_token = decode_strategy(next_logits)\n        # 將新 token 附加到序列\n        idx = torch.cat([idx, next_token], dim=1)" },
          ],
        },
        // ── Section 2: Greedy Decoding ──
        {
          title: "Greedy Decoding",
          blocks: [
            { type: "paragraph", text: "最簡單的解碼策略：每一步都選擇機率最高的 token（argmax）。這保證了輸出的確定性——同樣的輸入永遠產生同樣的輸出。" },
            { type: "code", language: "python", code: "# Greedy decoding: 永遠選最大的\nnext_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)\n# next_token shape: (B, 1)" },
            { type: "heading", level: 3, text: "Greedy Decoding 的問題" },
            { type: "list", ordered: true, items: [
              "重複性高：模型容易陷入重複迴圈，生成 'the the the...' 或重複相同的句子",
              "缺乏創意：永遠走最安全的路線，無法產生驚喜或多樣性",
              "全域次優：逐步的局部最優不保證整個序列的全域最優（beam search 試圖緩解這個問題）",
            ]},
            { type: "callout", variant: "tip", text: "💡 講師心得：Greedy decoding 的問題不是它「太確定」，而是它「局部最優但全局最差」。就像下棋只看一步——每一步都選「當前最佳」，結果卻走進死胡同。具體表現就是重複（'the the the the...'）——模型進入了一個高概率的循環，但沒有全局規劃能力跳出來。" },
            { type: "callout", variant: "tip", text: "Greedy decoding 雖然簡單但並非一無是處——在需要高度確定性的場景（如程式碼補全、事實性回答）中，greedy 反而可能是最佳選擇。" },
          ],
        },
        // ── Section 3: Temperature Scaling ──
        {
          title: "Temperature Scaling",
          blocks: [
            { type: "paragraph", text: "Temperature 是控制生成隨機性的最直觀參數。其原理是在 softmax 之前將 logits 除以一個正數 T（temperature）：" },
            { type: "code", language: "python", code: "# Temperature scaling\nscaled_logits = logits / temperature\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "paragraph", text: "Temperature 改變了 softmax 輸出的「銳利程度」。直覺上可以這樣理解：temperature 控制模型有多「自信」。低溫度使模型更確定，高溫度讓模型更開放。" },
            { type: "diagram", content: "Temperature 對機率分布的影響（原始 logits = [2.0, 1.0, 0.5, 0.1]）:\n\n  T=0.1 (極低)  ████████████████████████  token_0: 99.7%\n                █                         token_1:  0.3%\n                                          token_2:  0.0%\n                                          token_3:  0.0%\n\n  T=0.5 (低)    ██████████████████        token_0: 73.1%\n                ████████                  token_1: 19.7%\n                ████                      token_2:  5.6%\n                ██                        token_3:  1.6%\n\n  T=1.0 (標準)  ██████████████            token_0: 46.1%\n                ████████                  token_1: 25.2%\n                ██████                    token_2: 15.4%\n                ████                      token_3: 13.3%\n\n  T=2.0 (高)    █████████                 token_0: 33.0%\n                ████████                  token_1: 26.4%\n                ██████                    token_2: 22.4%\n                ██████                    token_3: 18.2%\n\n  T→∞           ████████                  趨近均勻分布\n                ████████                  每個 token 機率相等\n                ████████\n                ████████" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Temperature 的本質是在 softmax 之前對 logits 做縮放。T<1 時 logits 差異被放大→分布更尖銳→更確定；T>1 時差異被縮小→分布更平坦→更隨機。T→0 退化為 greedy，T→∞ 退化為均勻分布。理解這一點，你就理解了為什麼 temperature 是所有 LLM API 都暴露的第一個參數。" },
            { type: "callout", variant: "warning", text: "Temperature = 0 會導致除以零！實作中需要特殊處理：當 temperature < 1e-8 時退回 greedy decoding（argmax）。" },
            { type: "paragraph", text: "數學上，softmax(x/T) 在 T→0 時退化為 argmax（one-hot 分布），在 T→∞ 時退化為均勻分布。temperature = 1.0 就是標準的 softmax。" },
          ],
        },
        // ── Section 4: Top-k Sampling ──
        {
          title: "Top-k Sampling",
          blocks: [
            { type: "paragraph", text: "Temperature sampling 的一個問題是：即使在低溫下，長尾中那些機率極低的 token 仍有可能被取樣到。當詞彙表有 50,000+ 個 token 時，累積的小機率不可忽略。Top-k sampling 直接截斷：只保留機率最高的 k 個 token，其餘全部排除。" },
            { type: "code", language: "python", code: "# Top-k sampling\nscaled_logits = logits / temperature\n# 找到第 k 大的值作為門檻\ntop_k_values, _ = torch.topk(scaled_logits, k=k)\nthreshold = top_k_values[:, -1:]  # 第 k 大的值\n# 低於門檻的 logits 設為 -inf\nscaled_logits[scaled_logits < threshold] = float('-inf')\n# 正常 softmax + sampling\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "heading", level: 3, text: "Top-k 的局限" },
            { type: "paragraph", text: "k 是固定的，但最佳的候選數量隨上下文變化。有些位置分布很尖銳（只有一兩個合理的下一個詞），有些位置分布很平坦（許多詞都合理）。固定的 k 無法適應這種變化：k 太小會錯過合理選項，k 太大會引入噪聲。" },
            { type: "callout", variant: "tip", text: "k = 1 等價於 greedy decoding。k = vocab_size 等價於純 temperature sampling。實務上 k = 40~100 是常見的選擇。" },
          ],
        },
        // ── Section 5: Top-p / Nucleus Sampling ──
        {
          title: "Top-p (Nucleus) Sampling",
          blocks: [
            { type: "paragraph", text: "Top-p sampling（又稱 nucleus sampling）由 Holtzman et al. (2019) 提出，優雅地解決了 top-k 的固定截斷問題。其核心思想是：動態選取最小的 token 集合，使得這些 token 的累積機率恰好超過門檻 p。" },
            { type: "paragraph", text: "例如 p=0.9 表示：從最高機率的 token 開始，依序加入候選集，直到累積機率超過 90%。在分布尖銳時，可能只需要 2-3 個 token 就達到門檻；在分布平坦時，可能需要 50 個以上。" },
            { type: "code", language: "python", code: "# Top-p (nucleus) sampling\nscaled_logits = logits / temperature\n# 降序排列\nsorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)\n# 計算累積機率\ncumulative_probs = torch.cumsum(\n    F.softmax(sorted_logits, dim=-1), dim=-1\n)\n# 建立遮罩：累積機率超過 p 的位置（向右偏移 1 以保留至少一個 token）\nmask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p\n# 將被遮罩的位置設為 -inf\nsorted_logits[mask] = float('-inf')\n# 還原到原始順序\nscaled_logits.scatter_(1, sorted_indices, sorted_logits)\n# 正常 sampling\nprobs = torch.softmax(scaled_logits, dim=-1)\nnext_token = torch.multinomial(probs, num_samples=1)" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Top-p 比 top-k 更聰明的地方在於它能自適應。當模型很確定時（如「法國的首都是巴」→「黎」），top-p=0.9 可能只保留 1-2 個 token。當模型不確定時（如「今天天氣」→ 很多合理的續寫），同樣的 p=0.9 可能保留幾十個 token。Top-k 做不到這一點——k=50 永遠選 50 個，不管模型有多確定。這就是為什麼 GPT-4 默認用 top-p 而非 top-k。" },
            { type: "callout", variant: "info", text: "遮罩的偏移（shift right by 1）是關鍵細節：我們要保證至少保留一個 token（機率最高的那個）。偏移的方式是從累積機率中減去當前位置的機率，確保第一個位置的遮罩值永遠是 0（不被遮罩）。" },
          ],
        },
        // ── Section 6: Strategy Comparison ──
        {
          title: "策略比較與統一介面",
          blocks: [
            { type: "paragraph", text: "每種解碼策略都有其適用場景。以下是綜合比較：" },
            { type: "table", headers: ["策略", "參數", "確定性", "多樣性", "品質風險", "適用場景"], rows: [
              ["Greedy", "無", "完全確定", "無", "重複、呆板", "程式碼補全、事實問答"],
              ["Temperature", "T ∈ (0, ∞)", "T→0 確定", "T↑ 增加", "T 過高時亂碼", "創意寫作（T=0.7~0.9）"],
              ["Top-k", "k ∈ [1, V]", "k=1 確定", "k↑ 增加", "k 固定不靈活", "一般文本生成（k=40~100）"],
              ["Top-p", "p ∈ (0, 1]", "p→0 確定", "p↑ 增加", "幾乎無", "最佳通用選擇（p=0.9~0.95）"],
            ]},
            { type: "heading", level: 3, text: "Unified Generate Interface" },
            { type: "paragraph", text: "實務上，我們需要一個統一的入口函數，接受使用者的文本 prompt，自動處理 tokenize → tensor 轉換 → 解碼策略分派 → detokenize 的完整流程。" },
            { type: "code", language: "python", code: "def generate(model, tokenizer, prompt, max_new_tokens=50,\n             strategy='greedy', device='cpu', **kwargs):\n    # 1. Encode prompt → token IDs\n    ids = tokenizer.encode(prompt)\n    idx = torch.tensor([ids], dtype=torch.long, device=device)\n    # 2. 分派到對應的策略函數\n    strategies = {\n        'greedy': greedy_decode,\n        'temperature': temperature_sample,\n        'top_k': top_k_sample,\n        'top_p': top_p_sample,\n    }\n    fn = strategies[strategy]\n    # 3. 生成\n    output_ids = fn(model, idx, max_new_tokens, device=device, **kwargs)\n    # 4. Decode 回文字\n    return tokenizer.decode(output_ids[0].tolist())" },
            { type: "callout", variant: "tip", text: "使用 **kwargs 透傳參數（如 temperature, k, p）到策略函數。這讓統一介面無需了解每個策略的具體參數，保持了擴展性。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：在實際的 LLM API（如 OpenAI 或 Anthropic）中，你可以同時設定 temperature 和 top_p。這兩個參數會怎麼交互作用？如果 temperature=0.7 且 top_p=0.9，是先做 temperature scaling 再做 top-p truncation，還是反過來？順序不同結果一樣嗎？" },
          ],
        },
      ],
      exercises: [
        {
          id: "greedy_decode", title: "TODO 1: greedy_decode",
          description: "實作 greedy decoding——每一步選擇 logits 最大的 token（argmax）。需處理 block_size 截斷、model.eval() 與 torch.no_grad()。",
          labFile: "labs/phase5_generation/phase_5/generate.py",
          hints: [
            "使用 model.eval() 和 torch.no_grad() 進行推論",
            "如果模型有 block_size 屬性，截斷 idx 到最後 block_size 個 token",
            "logits, _ = model(idx_cond) 取得 logits",
            "torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) 取得下一個 token",
          ],
          pseudocode: "model.eval()\nidx = idx.to(device)\nwith torch.no_grad():\n  for _ in range(max_new_tokens):\n    idx_cond = idx[:, -block_size:] if hasattr(...) else idx\n    logits, _ = model(idx_cond)\n    next_logits = logits[:, -1, :]\n    next_id = torch.argmax(next_logits, dim=-1, keepdim=True)\n    idx = torch.cat([idx, next_id], dim=1)\nreturn idx",
        },
        {
          id: "temperature_sample", title: "TODO 2: temperature_sample",
          description: "實作 temperature sampling——將 logits 除以 temperature 後做 softmax 再用 multinomial 取樣。當 temperature ≈ 0 時退回 greedy。",
          labFile: "labs/phase5_generation/phase_5/generate.py",
          hints: [
            "特殊處理 temperature < 1e-8：使用 argmax 取代 sampling",
            "scaled_logits = logits / temperature",
            "probs = torch.softmax(scaled_logits, dim=-1)",
            "next_id = torch.multinomial(probs, num_samples=1)",
          ],
          pseudocode: "model.eval()\nidx = idx.to(device)\nwith torch.no_grad():\n  for _ in range(max_new_tokens):\n    logits = model(idx_cond)[0][:, -1, :]\n    if temperature < 1e-8:\n      next_id = argmax(logits)\n    else:\n      scaled = logits / temperature\n      probs = softmax(scaled, dim=-1)\n      next_id = multinomial(probs, 1)\n    idx = cat([idx, next_id], dim=1)\nreturn idx",
        },
        {
          id: "top_k_sample", title: "TODO 3: top_k_sample",
          description: "實作 top-k sampling——只從機率最高的 k 個 token 中取樣。需先做 temperature scaling，再截斷低機率 token（設為 -inf），最後 softmax + multinomial。",
          labFile: "labs/phase5_generation/phase_5/generate.py",
          hints: [
            "k 需要 clamp 到 vocab_size 以內：k = min(k, logits.size(-1))",
            "torch.topk(logits, k) 取得前 k 大的值和索引",
            "top_k_values[:, -1:] 是第 k 大的值，作為門檻",
            "logits[logits < threshold] = float('-inf') 截斷低機率 token",
          ],
          pseudocode: "model.eval()\nfor _ in range(max_new_tokens):\n  logits = model(idx_cond)[0][:, -1, :] / temperature\n  k_clamped = min(k, logits.size(-1))\n  top_vals, _ = torch.topk(logits, k_clamped)\n  threshold = top_vals[:, -1:]\n  logits[logits < threshold] = -inf\n  probs = softmax(logits, dim=-1)\n  next_id = multinomial(probs, 1)\n  idx = cat([idx, next_id], dim=1)",
        },
        {
          id: "top_p_sample", title: "TODO 4: top_p_sample",
          description: "實作 top-p (nucleus) sampling——動態選取累積機率超過 p 的最小 token 集合。需排序 logits、計算 cumulative probabilities、建立遮罩並還原順序。",
          labFile: "labs/phase5_generation/phase_5/generate.py",
          hints: [
            "torch.sort(logits, descending=True) 降序排列",
            "torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) 計算累積機率",
            "遮罩需偏移一位以保留至少一個 token",
            "logits.scatter_(1, sorted_indices, sorted_logits) 還原到原始順序",
          ],
          pseudocode: "model.eval()\nfor _ in range(max_new_tokens):\n  logits = model(idx_cond)[0][:, -1, :] / temperature\n  sorted_logits, sorted_idx = sort(logits, descending=True)\n  cum_probs = cumsum(softmax(sorted_logits), dim=-1)\n  # 偏移遮罩：減去當前位置機率\n  mask = (cum_probs - softmax(sorted_logits)) >= p\n  sorted_logits[mask] = -inf\n  logits.scatter_(1, sorted_idx, sorted_logits)\n  probs = softmax(logits, dim=-1)\n  next_id = multinomial(probs, 1)\n  idx = cat([idx, next_id], dim=1)",
        },
        {
          id: "generate", title: "TODO 5: generate (unified interface)",
          description: "實作統一生成介面——接受文字 prompt，encode → 選擇策略 → decode 回文字。使用字典分派策略函數，透過 **kwargs 傳遞策略參數。",
          labFile: "labs/phase5_generation/phase_5/generate.py",
          hints: [
            "用 tokenizer.encode(prompt) 取得 token IDs",
            "轉為 tensor：torch.tensor([ids], dtype=torch.long, device=device)",
            "建立策略字典：{'greedy': greedy_decode, 'temperature': temperature_sample, ...}",
            "用 tokenizer.decode(output_ids[0].tolist()) 解碼回文字",
          ],
          pseudocode: "ids = tokenizer.encode(prompt)\nidx = tensor([ids], dtype=long, device=device)\nstrategies = {'greedy': greedy_decode, ...}\nfn = strategies[strategy]  # KeyError if invalid\noutput = fn(model, idx, max_new_tokens, device=device, **kwargs)\nreturn tokenizer.decode(output[0].tolist())",
        },
      ],
      acceptanceCriteria: [
        "greedy_decode 輸出 shape 為 (1, T + max_new_tokens) 且完全確定性",
        "temperature_sample 在 temperature ≈ 0 時等價於 greedy decoding",
        "top_k_sample 在 k=1 時等價於 greedy decoding",
        "top_p_sample 在 p 極小時接近 greedy，p=1.0 時包含全部 token",
        "generate() 正確分派策略並回傳字串，非法策略拋出例外",
        "所有函數保留原始 input prefix 不變",
        "所有生成的 token ID 在 [0, vocab_size) 範圍內",
      ],
      references: [
        { title: "The Curious Case of Neural Text Degeneration", description: "Holtzman et al. 2019 — 提出 nucleus (top-p) sampling，分析了 greedy / beam search 的退化問題", url: "https://arxiv.org/abs/1904.09751" },
        { title: "How to generate text: using different decoding methods for language generation with Transformers", description: "HuggingFace 官方部落格，圖文並茂地解釋各種解碼策略", url: "https://huggingface.co/blog/how-to-generate" },
        { title: "Language Models are Unsupervised Multitask Learners", description: "GPT-2 論文——文本生成的基礎架構", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
      ],
    },
  ],
};


// ═══════════════════════════════════════════════════════════════════
// Phase 6: Classification Fine-Tuning
// ═══════════════════════════════════════════════════════════════════

export const phase6Content: PhaseContent = {
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
        // ── Section 1: Why Last Token? ──
        {
          title: "為什麼用最後一個 Token 的表示？",
          blocks: [
            { type: "paragraph", text: "到目前為止，我們的 GPT 模型只會一件事：生成文本。但在現實世界中，很多任務不是「寫作文」，而是「做判斷」——這封郵件是不是垃圾郵件？這條評論是正面還是負面？這篇文章屬於哪個分類？今天我們要玩一個有趣的把戲：把一個「說書人」（生成模型）改造成「裁判」（分類模型）。你會驚訝地發現，這個改造出奇地簡單——只需要在模型頂部加一層就夠了。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：把一個生成式模型改成分類器，聽起來有點違直覺——GPT 是用來寫文章的，怎麼拿來分類垃圾郵件？關鍵在於預訓練過程中，模型已經學到了豐富的語言理解能力。分類微調只是在模型頂部加一個「分類頭」，利用這些已有的語言知識來做決策。這就是遷移學習的威力。" },
            { type: "paragraph", text: "GPT 是一個 causal（自回歸）語言模型——每個 token 只能 attend 到它自己和之前的 token。這意味著最後一個 token 的隱藏狀態是唯一「看過整個輸入序列」的表示，它彙集了整個輸入的語義資訊。" },
            { type: "paragraph", text: "相較之下，BERT 這類雙向模型通常使用 [CLS] token（第一個 token）的表示做分類，因為 BERT 的每個 token 都能看到完整序列。但在 GPT 中，第一個 token 只看到自己，資訊量極少。因此我們取最後一個 token 的隱藏狀態作為整個序列的表示。" },
            { type: "diagram", content: "GPT Causal Attention — 資訊流向:\n\n  Token:    [This]  [is]  [spam]  [!]    ← 輸入序列\n             │       │      │      │\n  Attend:   [自己] [前2] [前3]  [全部]   ← 每個 token 能看到的範圍\n             │       │      │      │\n  Info:      ●       ●●     ●●●    ●●●●  ← 資訊量遞增\n                                   ▲\n                                   │\n                          最後一個 token 包含\n                          整個序列的資訊\n                                   │\n                                   ▼\n                          [Classification Head]\n                                   │\n                                   ▼\n                          [spam / not spam]" },
            { type: "callout", variant: "info", text: "如果序列有 padding，最後一個「真實」token 才是有意義的。此時需要 attention_mask 來找到最後一個非 padding 位置：last_pos = attention_mask.sum(dim=1) - 1。" },
          ],
        },
        // ── Section 2: GPT → Classification Head Architecture ──
        {
          title: "GPTClassifier 架構",
          blocks: [
            { type: "paragraph", text: "GPTClassifier 的架構非常直接：保留預訓練的 GPT 模型作為特徵提取器（backbone），在其上方加一個線性分類頭（Linear layer）。完整的前向傳播路徑是：" },
            { type: "diagram", content: "GPTClassifier Architecture:\n\n  input_ids (B, T)\n       │\n       ▼\n  ┌──────────────────────────────────────┐\n  │         GPT Backbone (frozen?)       │\n  │  ┌──────────────────────────────┐    │\n  │  │  Token Embedding             │    │\n  │  │  + Position Embedding        │    │\n  │  └──────────────┬───────────────┘    │\n  │                 ▼                    │\n  │  ┌──────────────────────────────┐    │\n  │  │  Transformer Block × N       │    │\n  │  │  (Self-Attention + FFN)      │    │\n  │  └──────────────┬───────────────┘    │\n  │                 ▼                    │\n  │  ┌──────────────────────────────┐    │\n  │  │  LayerNorm                   │    │\n  │  └──────────────┬───────────────┘    │\n  └─────────────────┼────────────────────┘\n                    │\n       hidden_states (B, T, n_embd)\n                    │\n       Extract last token: (B, n_embd)\n                    │\n                    ▼\n            ┌──────────────┐\n            │   Dropout     │\n            └──────┬───────┘\n                   ▼\n            ┌──────────────┐\n            │  Linear       │\n            │ (n_embd → C)  │  C = n_classes\n            └──────┬───────┘\n                   ▼\n            logits (B, C)" },
            { type: "heading", level: 3, text: "取得隱藏狀態" },
            { type: "paragraph", text: "GPT 的 forward 方法通常返回 (logits, loss)，其中 logits 是語言模型頭（LM head）的輸出，形狀為 (B, T, vocab_size)。但分類任務需要的是 LM head 之前的隱藏狀態 (B, T, n_embd)。實作時有兩種策略：" },
            { type: "list", ordered: true, items: [
              "直接存取 backbone 內部：依序跑 embedding → transformer blocks → layer norm，跳過最後的 LM head",
              "使用完整 forward 的 logits 作為替代（shape 不同但功能上可行，適用於測試環境中的簡化模型）",
            ]},
            { type: "code", language: "python", code: "# 方法 1: 直接存取 backbone 內部層（推薦）\ndef forward(self, input_ids, attention_mask=None):\n    # 跑到 layer norm 為止，不經過 LM head\n    x = self.backbone.tok_emb(input_ids) + self.backbone.pos_emb(...)\n    for block in self.backbone.blocks:\n        x = block(x)\n    hidden = self.backbone.ln_f(x)  # (B, T, n_embd)\n\n    # 取最後一個 token 的隱藏狀態\n    if attention_mask is not None:\n        last_pos = attention_mask.sum(dim=1) - 1\n        pooled = hidden[torch.arange(B), last_pos]\n    else:\n        pooled = hidden[:, -1, :]  # (B, n_embd)\n\n    return self.classifier_head(self.dropout(pooled))" },
          ],
        },
        // ── Section 3: Feature Extraction vs Full Fine-Tuning ──
        {
          title: "Feature Extraction vs Full Fine-Tuning",
          blocks: [
            { type: "paragraph", text: "將預訓練模型用於下游任務有兩種主要策略，它們在訓練效率、資料需求和最終效果上有顯著差異：" },
            { type: "table", headers: ["特性", "Feature Extraction（凍結 backbone）", "Full Fine-Tuning（解凍 backbone）"], rows: [
              ["可訓練參數", "僅分類頭（極少）", "整個模型（數百萬+）"],
              ["訓練速度", "快（梯度只算分類頭）", "慢（梯度流過整個模型）"],
              ["記憶體需求", "低", "高（需儲存所有參數的梯度）"],
              ["資料需求", "少量資料即可", "需要更多資料避免 overfitting"],
              ["效果上限", "受限於預訓練表示", "可適應特定領域，效果通常更好"],
              ["過擬合風險", "低", "高（尤其資料少時）"],
              ["適用場景", "快速 baseline、資料稀少", "有足夠資料且追求最佳效果"],
            ]},
            { type: "heading", level: 3, text: "實作凍結與解凍" },
            { type: "paragraph", text: "凍結 backbone 的實作非常簡單：遍歷 backbone 的所有參數，設定 requires_grad = False。PyTorch 的 autograd 引擎會自動跳過不需要梯度的參數，既節省計算又節省記憶體。" },
            { type: "code", language: "python", code: "def freeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = False\n\ndef unfreeze_backbone(model: GPTClassifier):\n    for param in model.backbone.parameters():\n        param.requires_grad = True" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Feature extraction（凍結骨幹）vs Full fine-tuning 的選擇本質上是偏差-方差的權衡。數據少時，凍結骨幹防止過擬合（高偏差但低方差）。數據多時，解凍讓模型完全適應新任務（低偏差但可能高方差）。實務上，我建議先試 feature extraction——如果效果夠好就不需要冒過擬合的風險。這也是 Phase 7 LoRA 存在的意義：用極少的可訓練參數來折衷。" },
            { type: "callout", variant: "tip", text: "常見的實務策略是先凍結 backbone 訓練分類頭（warm-up），再解凍 backbone 用很小的 learning rate 做 full fine-tuning。這種「漸進式解凍」往往能取得最佳效果。" },
            { type: "callout", variant: "warning", text: "凍結 backbone 後，classifier_head 的 requires_grad 必須保持 True！否則整個模型都不會更新。__init__ 中新建的 nn.Linear 預設就是 requires_grad=True，不需要額外設定。" },
          ],
        },
        // ── Section 4: SpamDataset ──
        {
          title: "SpamDataset：資料準備流程",
          blocks: [
            { type: "paragraph", text: "分類任務的資料格式與 next-token prediction 不同：每個樣本是一對（文本, 標籤），而非連續的 token 流。我們需要一個 Dataset 類別來處理 CSV 讀取、tokenization、padding/truncation 和 attention mask 生成。" },
            { type: "heading", level: 3, text: "資料處理流程" },
            { type: "list", ordered: true, items: [
              "讀取 CSV：使用 csv.DictReader 解析，提取 'text' 和 'label' 欄位",
              "Tokenize：用 tokenizer.encode(text) 將文本轉為 token ID 序列",
              "截斷：如果序列長度超過 max_length，截取前 max_length 個 token",
              "Padding：如果序列長度不足 max_length，在尾部補 0（padding token）",
              "Attention Mask：生成 1/0 遮罩，1 表示真實 token，0 表示 padding",
            ]},
            { type: "code", language: "python", code: "# SpamDataset.__init__ 的核心邏輯\nwith open(csv_path, 'r') as f:\n    reader = csv.DictReader(f)\n    for row in reader:\n        text = row['text']\n        label = int(row['label'])\n        ids = tokenizer.encode(text)\n        # 截斷\n        ids = ids[:max_length]\n        actual_len = len(ids)\n        # Padding\n        ids = ids + [0] * (max_length - actual_len)\n        # Attention mask\n        mask = [1] * actual_len + [0] * (max_length - actual_len)\n        # 儲存\n        self.input_ids.append(ids)\n        self.attention_masks.append(mask)\n        self.labels.append(label)" },
            { type: "heading", level: 3, text: "__getitem__ 返回格式" },
            { type: "paragraph", text: "PyTorch DataLoader 會自動 collate 每個樣本。為了與 PyTorch 標準流程相容，__getitem__ 應返回一個字典，包含三個 torch.long tensor：" },
            { type: "code", language: "python", code: "def __getitem__(self, idx):\n    return {\n        'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),\n        'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),\n        'label': torch.tensor(self.labels[idx], dtype=torch.long),\n    }" },
            { type: "callout", variant: "info", text: "DataLoader 的 default_collate 會自動將字典的每個 key 對應的 tensor 堆疊為 batch。所以 batch['input_ids'] 的 shape 會是 (batch_size, max_length)。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們用最後一個 token 的 hidden state 來做分類。但想想看——如果輸入是 'This movie is great!'，最後一個 token 是 '!'（感嘆號）。感嘆號本身沒有情感含義，為什麼用它的表示向量來分類卻能得到好結果？（提示：想想 causal attention 的累積效應。）" },
          ],
        },
      ],
      exercises: [
        {
          id: "gpt_classifier_init", title: "TODO 1: GPTClassifier.__init__",
          description: "初始化分類器：儲存 GPT backbone、建立 dropout 層和線性分類頭（nn.Linear(n_embd, n_classes)），並根據 freeze_backbone 參數凍結 backbone 參數。",
          labFile: "labs/phase6_classification/phase_6/classifier.py",
          hints: [
            "用 gpt_model.config.n_embd 或類似屬性取得隱藏維度",
            "self.backbone = gpt_model",
            "self.dropout = nn.Dropout(dropout)",
            "self.classifier_head = nn.Linear(n_embd, n_classes)",
            "凍結：for param in self.backbone.parameters(): param.requires_grad = False",
          ],
          pseudocode: "super().__init__()\nself.backbone = gpt_model\nself.dropout = nn.Dropout(dropout)\nn_embd = gpt_model.config.n_embd  # 或其他方式取得\nself.classifier_head = nn.Linear(n_embd, n_classes)\nif freeze_backbone:\n  for p in self.backbone.parameters():\n    p.requires_grad = False",
        },
        {
          id: "gpt_classifier_forward", title: "TODO 2: GPTClassifier.forward",
          description: "前向傳播：通過 backbone 取得隱藏狀態，提取最後一個 token（或最後一個非 padding token）的表示，經過 dropout 和分類頭輸出 (B, n_classes) 的 logits。",
          labFile: "labs/phase6_classification/phase_6/classifier.py",
          hints: [
            "需要取得 LM head 之前的隱藏狀態 (B, T, n_embd)",
            "如果有 attention_mask：last_pos = attention_mask.sum(dim=1) - 1",
            "如果沒有 attention_mask：使用 hidden[:, -1, :]",
            "最後經過 self.dropout → self.classifier_head",
          ],
          pseudocode: "# 取得隱藏狀態（具體方式取決於 backbone 結構）\nhidden = backbone_forward_to_hidden(input_ids)  # (B, T, n_embd)\n# 提取最後 token\nif attention_mask is not None:\n  last_pos = attention_mask.sum(1) - 1\n  pooled = hidden[arange(B), last_pos]\nelse:\n  pooled = hidden[:, -1, :]\n# 分類\nreturn self.classifier_head(self.dropout(pooled))",
        },
        {
          id: "freeze_model", title: "TODO 3: freeze_backbone",
          description: "凍結 GPT backbone 的所有參數（requires_grad = False），使其在訓練時不更新。",
          labFile: "labs/phase6_classification/phase_6/classifier.py",
          hints: [
            "遍歷 model.backbone.parameters()",
            "設定每個 param.requires_grad = False",
          ],
          pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = False",
        },
        {
          id: "unfreeze_model", title: "TODO 4: unfreeze_backbone",
          description: "解凍 GPT backbone 的所有參數（requires_grad = True），啟用 full fine-tuning。",
          labFile: "labs/phase6_classification/phase_6/classifier.py",
          hints: [
            "遍歷 model.backbone.parameters()",
            "設定每個 param.requires_grad = True",
          ],
          pseudocode: "for param in model.backbone.parameters():\n  param.requires_grad = True",
        },
        {
          id: "spam_dataset_init", title: "TODO 5: SpamDataset.__init__",
          description: "從 CSV 載入資料，對每筆文本做 tokenize、truncation、padding 並生成 attention mask。",
          labFile: "labs/phase6_classification/phase_6/dataset.py",
          hints: [
            "使用 csv.DictReader 讀取 CSV，欄位為 'text' 和 'label'",
            "tokenizer.encode(text) 取得 token IDs",
            "截斷：ids = ids[:max_length]",
            "Padding：ids + [0] * (max_length - len(ids))",
            "Attention mask：[1] * actual_len + [0] * padding_len",
          ],
          pseudocode: "self.input_ids, self.masks, self.labels = [], [], []\nwith open(csv_path) as f:\n  for row in csv.DictReader(f):\n    ids = tokenizer.encode(row['text'])[:max_length]\n    actual = len(ids)\n    ids += [0] * (max_length - actual)\n    mask = [1]*actual + [0]*(max_length - actual)\n    self.input_ids.append(ids)\n    self.masks.append(mask)\n    self.labels.append(int(row['label']))",
        },
        {
          id: "spam_dataset_getitem", title: "TODO 6: SpamDataset.__getitem__",
          description: "返回第 idx 筆樣本的字典，包含 input_ids、attention_mask、label 三個 torch.long tensor。",
          labFile: "labs/phase6_classification/phase_6/dataset.py",
          hints: [
            "用 torch.tensor(..., dtype=torch.long) 建立 tensor",
            "返回字典：{'input_ids': ..., 'attention_mask': ..., 'label': ...}",
          ],
          pseudocode: "return {\n  'input_ids': tensor(self.input_ids[idx], dtype=long),\n  'attention_mask': tensor(self.masks[idx], dtype=long),\n  'label': tensor(self.labels[idx], dtype=long),\n}",
        },
      ],
      acceptanceCriteria: [
        "GPTClassifier 輸出 shape 為 (batch_size, n_classes)",
        "freeze_backbone=True 時 backbone 參數的 requires_grad 為 False",
        "classifier_head 的 requires_grad 始終為 True",
        "freeze_backbone() 和 unfreeze_backbone() 正確切換 requires_grad",
        "凍結狀態下 backward 不會更新 backbone 參數",
        "forward 在有/無 attention_mask 時都能正確運作",
        "SpamDataset 正確讀取 CSV 並返回正確數量的樣本",
        "input_ids 和 attention_mask 的 shape 為 (max_length,)，dtype 為 torch.long",
        "Padding 在序列尾部，attention_mask 正確標記真實 token 與 padding",
        "截斷時 attention_mask 全為 1（所有位置都是真實 token）",
      ],
      references: [
        { title: "Improving Language Understanding by Generative Pre-Training", description: "GPT-1 論文 (Radford et al. 2018) — 首次展示 generative pre-training + discriminative fine-tuning 的範式", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" },
        { title: "Universal Language Model Fine-tuning for Text Classification", description: "ULMFiT (Howard & Ruder 2018) — 提出漸進式解凍等 fine-tuning 技巧", url: "https://arxiv.org/abs/1801.06146" },
        { title: "BERT: Pre-training of Deep Bidirectional Transformers", description: "BERT 論文 (Devlin et al. 2018) — feature extraction vs fine-tuning 的經典對比", url: "https://arxiv.org/abs/1810.04805" },
        { title: "PyTorch Transfer Learning Tutorial", description: "PyTorch 官方教學——凍結/解凍模型參數的最佳實踐", url: "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html" },
      ],
    },
  ],
};
