import type { PhaseContent } from "@/data/types";
import type { Locale } from "@/i18n";

// ═══════════════════════════════════════════════════════════════════
// Phase 1 & 2 — zh-TW
// ═══════════════════════════════════════════════════════════════════

const phase1ContentZhTW: PhaseContent = {
  phaseId: 1,
  color: "#3B82F6",
  accent: "#60A5FA",
  lessons: [
    {
      phaseId: 1, lessonId: 1,
      title: "BPE Tokenizer",
      subtitle: "從字元到子詞——構建你自己的 Byte-Pair Encoding 分詞器",
      type: "concept",
      duration: "45 min",
      objectives: [
        "理解字元級、詞級與子詞級分詞的差異與取捨",
        "完整實作 BPE 演算法的訓練過程（merge 規則學習）",
        "實作 encode 與 decode 函數，確保完美的 roundtrip",
      ],
      sections: [
        {
          title: "Why Tokenization Matters",
          blocks: [
            { type: "paragraph", text: "你有沒有想過，ChatGPT 看到的「文字」跟你看到的其實不一樣？你看到的是「Hello World」兩個英文單字。但 GPT 看到的是 [15496, 2159] 兩個數字。從人類的文字到模型的數字，中間就需要一個「翻譯官」——這就是 Tokenizer。今天我們要從零搭建這個翻譯官。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Tokenizer 是 LLM 最被低估的組件。很多 LLM 看起來「愚蠢」的行為——數不清單詞裡有幾個字母、做不好簡單算術——其實是 tokenization 的鍋。模型看到的不是你以為的那些字元，而是被切割過的子詞碎片。理解 tokenizer，就理解了模型「視力」的極限。" },
            { type: "paragraph", text: "大型語言模型不直接處理文字——它們處理的是整數序列。Tokenization 就是將原始文字轉換成模型能理解的數字序列的過程。選擇什麼樣的 tokenization 策略，直接影響模型的詞彙表大小、序列長度、以及對未知詞的處理能力。" },
            { type: "table", headers: ["策略", "詞彙表大小", "序列長度", "OOV 問題"], rows: [
              ["Character-level", "~256", "非常長", "無"],
              ["Word-level", "100K+", "短", "嚴重"],
              ["BPE (Subword)", "~50K", "適中", "幾乎無"],
            ]},
            { type: "callout", variant: "info", text: "GPT-2 使用約 50,257 個 BPE token。這個數字在詞彙表大小（影響 embedding 矩陣大小）和序列長度（影響 attention 計算量）之間取得了良好的平衡。" },
            { type: "paragraph", text: "BPE 的核心思想很簡單：從字元開始，反覆合併最常出現的相鄰 pair，直到達到目標詞彙表大小。這讓常見詞保持完整（如 'the'），同時罕見詞被拆成有意義的子片段（如 'tokenization' → 'token' + 'ization'）。" },
          ],
        },
        {
          title: "BPE Algorithm Step by Step",
          blocks: [
            { type: "paragraph", text: "BPE 訓練過程可以分為三個核心步驟：建立初始詞彙表、統計 pair 頻率、執行合併。我們反覆執行後兩步，直到詞彙表達到目標大小。" },
            { type: "heading", level: 3, text: "Step 1: 初始化" },
            { type: "paragraph", text: "將訓練文本轉換為 UTF-8 bytes 序列。初始詞彙表就是 0-255 這 256 個 byte 值。每個字元（或多 byte 字元的每個 byte）都是一個獨立的 token。" },
            { type: "heading", level: 3, text: "Step 2: 統計相鄰 Pair" },
            { type: "code", language: "python", code: "def get_pair_counts(token_ids: list[int]) -> dict[tuple[int,int], int]:\n    counts = {}\n    for i in range(len(token_ids) - 1):\n        pair = (token_ids[i], token_ids[i + 1])\n        counts[pair] = counts.get(pair, 0) + 1\n    return counts" },
            { type: "heading", level: 3, text: "Step 3: 合併最高頻 Pair" },
            { type: "diagram", content: "訓練語料: 'aaabdaaabac'\n\n初始: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]\n\nMerge 1: (97,97)→256  →  [256, 97, 98, 100, 256, 97, 98, 97, 99]\nMerge 2: (256,97)→257 →  [257, 98, 100, 257, 98, 97, 99]\nMerge 3: (257,98)→258 →  [258, 100, 258, 97, 99]" },
            { type: "callout", variant: "tip", text: "合併順序很重要！encode 時必須按照訓練時學到的順序依次應用 merge 規則。這就是為什麼我們需要保存一個有序的 merges 列表。" },
            { type: "callout", variant: "tip", text: "💡 講師心得：BPE 本質上是一個壓縮演算法，被巧妙地借用到了 NLP 領域。它在 1994 年被發明用於資料壓縮（Philip Gage），2016 年 Sennrich 等人發現它完美適用於子詞分割。這個跨領域的借用提醒我們：好的工程方案往往來自意想不到的地方。" },
          ],
        },
        {
          title: "Encoding & Decoding",
          blocks: [
            { type: "paragraph", text: "訓練完成後，我們得到了一個有序的 merge 規則列表。Encoding 就是將文本先轉為 bytes，再按順序套用每條 merge 規則。Decoding 則是將 token ID 序列轉回 bytes 再解碼為文字。" },
            { type: "code", language: "python", code: "def encode(self, text: str) -> list[int]:\n    tokens = list(text.encode('utf-8'))  # 轉為 byte list\n    for (p0, p1), new_id in self.merges.items():\n        i = 0\n        new_tokens = []\n        while i < len(tokens):\n            if i < len(tokens)-1 and tokens[i]==p0 and tokens[i+1]==p1:\n                new_tokens.append(new_id)\n                i += 2\n            else:\n                new_tokens.append(tokens[i])\n                i += 1\n        tokens = new_tokens\n    return tokens" },
            { type: "code", language: "python", code: "def decode(self, token_ids: list[int]) -> str:\n    byte_list = b''.join(self.vocab[t] for t in token_ids)\n    return byte_list.decode('utf-8', errors='replace')" },
            { type: "callout", variant: "warning", text: "encode(decode(ids)) 不一定等於 ids，但 decode(encode(text)) 必須等於原始 text。這是最重要的不變量。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們的 BPE tokenizer 是 byte-level 的，可以處理任何 UTF-8 文字。但如果要處理中文、日文、韓文這類 CJK 字元，BPE 的效率如何？一個中文字佔 3 個 UTF-8 bytes，這對詞彙表大小和序列長度有什麼影響？GPT-4 的 tokenizer 做了什麼特殊處理來優化多語言效率？" },
          ],
        },
      ],
      exercises: [
        { id: "build_initial_vocab", title: "TODO 1: build_initial_vocab", description: "建立初始詞彙表，包含 0-255 的所有 byte 值。每個 entry 是 token_id → bytes 的映射。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["使用 dict comprehension", "bytes([i]) 可以將整數轉為單一 byte"], pseudocode: "vocab = {}\nfor i in 0..255:\n  vocab[i] = bytes([i])\nreturn vocab" },
        { id: "get_pair_counts", title: "TODO 2: get_pair_counts", description: "統計 token 序列中所有相鄰 pair 的出現次數。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["遍歷 range(len(ids)-1)", "用 dict 儲存計數"] },
        { id: "merge_pair", title: "TODO 3: merge_pair", description: "在 token 序列中，將所有 (p0, p1) pair 替換為 new_id。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["用 while loop 而非 for loop", "找到 pair 就 append new_id 並 i+=2"] },
        { id: "train", title: "TODO 4: train", description: "執行 BPE 訓練迴圈：反覆找最高頻 pair → 合併 → 更新詞彙表，直到達到目標大小。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["num_merges = target_vocab_size - 256", "每次 merge 後新 token ID = 256 + merge_count"] },
        { id: "encode", title: "TODO 5: encode", description: "將文本編碼為 token ID 序列，按訓練順序應用所有 merge 規則。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["先轉 UTF-8 bytes", "按 merges 的順序依次應用每條規則"] },
        { id: "decode", title: "TODO 6: decode", description: "將 token ID 序列解碼回文本字串。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["查表取得每個 token 的 bytes", "用 b''.join() 串接後 decode('utf-8')"] },
      ],
      acceptanceCriteria: [
        "vocab_size 可自訂設定，且訓練後詞彙表大小正確",
        "decode(encode(text)) == text 對所有 UTF-8 文本成立",
        "get_pair_counts 正確統計所有相鄰 pair",
        "merge 順序與訓練順序一致",
      ],
      references: [
        { title: "Neural Machine Translation of Rare Words with Subword Units", description: "Sennrich et al. 2016 — BPE 的原始論文，提出將 BPE 壓縮演算法用於 NLP", url: "https://arxiv.org/abs/1508.07909" },
        { title: "tiktoken", description: "OpenAI 的高效 BPE 實作，GPT-4 使用的 tokenizer", url: "https://github.com/openai/tiktoken" },
      ],
    },
    {
      phaseId: 1, lessonId: 2,
      title: "DataLoader for LLM Training",
      subtitle: "建立高效的資料管線——從原始文本到訓練批次",
      type: "concept",
      duration: "30 min",
      objectives: [
        "理解 next-token prediction 的核心訓練目標",
        "實作 sliding-window 方式建立 (x, y) 訓練對",
        "建立 train / validation 資料分割",
      ],
      sections: [
        {
          title: "Next-Token Prediction",
          blocks: [
            { type: "paragraph", text: "Tokenizer 搞定了，模型現在可以「看懂」文字了。但光有翻譯官還不夠——我們還需要一個「廚師」把原始食材加工成模型能消化的「便當」。這就是 DataLoader 的角色。模型到底要學什麼？答案是：給定前面的 token 序列，預測下一個 token。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Next-token prediction 是整個 LLM 世界的基石，但初學者常低估它的威力。看似只是「猜下一個字」，但要做好這件事，模型必須學會語法、語義、邏輯推理、甚至世界知識。" },
            { type: "paragraph", text: "具體來說，對於一個長度為 T 的序列 [t₀, t₁, ..., t_{T-1}]，我們構造：\n- 輸入 x = [t₀, t₁, ..., t_{T-2}]\n- 目標 y = [t₁, t₂, ..., t_{T-1}]\n\ny 就是 x 向右偏移一個位置。" },
            { type: "callout", variant: "info", text: "由於 causal masking，模型在位置 i 只能看到位置 0..i 的 token。所以每個訓練樣本實際上提供了 context_length 個訓練信號。" },
          ],
        },
        {
          title: "Sliding Window",
          blocks: [
            { type: "paragraph", text: "我們用固定大小的滑動視窗從已 tokenize 的文本中提取訓練樣本。視窗大小 = context_length + 1。每個視窗產生一對 (x, y)：x 是前 context_length 個 token，y 是後 context_length 個 token（偏移 1）。" },
            { type: "diagram", content: "Token sequence: [a, b, c, d, e, f, g, h, i, j]\ncontext_length = 4, stride = 4\n\nWindow 1: [a, b, c, d, e]  →  x=[a,b,c,d]  y=[b,c,d,e]\nWindow 2: [e, f, g, h, i]  →  x=[e,f,g,h]  y=[f,g,h,i]" },
            { type: "code", language: "python", code: "class TextDataset(Dataset):\n    def __init__(self, token_ids: list[int], context_length: int, stride: int):\n        self.x = []\n        self.y = []\n        for i in range(0, len(token_ids) - context_length, stride):\n            self.x.append(torch.tensor(token_ids[i:i+context_length]))\n            self.y.append(torch.tensor(token_ids[i+1:i+context_length+1]))\n\n    def __len__(self): return len(self.x)\n    def __getitem__(self, idx): return self.x[idx], self.y[idx]" },
          ],
        },
        {
          title: "Token & Position Embeddings",
          blocks: [
            { type: "paragraph", text: "Token ID 只是整數，模型需要的是連續向量。Token Embedding 是一個 V×D 的查找表，每個 token ID 對應一個 D 維向量。Position Embedding 是另一個 T×D 的查找表，每個位置對應一個 D 維向量。兩者相加得到最終的輸入表示。" },
            { type: "code", language: "python", code: "class Embeddings(nn.Module):\n    def __init__(self, vocab_size, embed_dim, context_length):\n        super().__init__()\n        self.token_emb = nn.Embedding(vocab_size, embed_dim)\n        self.pos_emb = nn.Embedding(context_length, embed_dim)\n\n    def forward(self, x):  # x: (batch, seq_len)\n        tok = self.token_emb(x)       # (batch, seq_len, embed_dim)\n        pos = self.pos_emb(torch.arange(x.size(1), device=x.device))\n        return tok + pos" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Token embedding 和 position embedding 的相加（而非拼接）是一個優雅的設計。兩個 embedding 共享同一個向量空間，相加後模型可以同時利用「這是什麼詞」和「這個詞在哪裡」的資訊。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們用「相加」而非「拼接」來合併 token embedding 和 position embedding。但這樣做不會互相干擾嗎？模型怎麼區分哪些資訊來自 token、哪些來自 position？（提示：想想高維空間中兩個隨機向量的夾角。）" },
          ],
        },
      ],
      exercises: [
        { id: "dataset_init", title: "TODO 1: TextDataset.__init__", description: "實作 sliding-window 邏輯，從 token 序列中提取 (x, y) 對。", labFile: "labs/phase1/dataloader.py", hints: ["用 range(0, len(ids) - context_length, stride) 作為起始索引", "x 和 y 之間偏移 1 個位置"] },
        { id: "dataset_len", title: "TODO 2: TextDataset.__len__", description: "返回資料集中的樣本數量。", labFile: "labs/phase1/dataloader.py", hints: ["直接返回 self.x 的長度"] },
        { id: "dataset_getitem", title: "TODO 3: TextDataset.__getitem__", description: "返回第 idx 個 (x, y) 訓練對。", labFile: "labs/phase1/dataloader.py", hints: ["return self.x[idx], self.y[idx]"] },
        { id: "create_dataloaders", title: "TODO 4: create_dataloaders", description: "將文本分割為 train/val，建立 DataLoader。", labFile: "labs/phase1/dataloader.py", hints: ["先 tokenize 整個文本", "用前 90% 做 train，後 10% 做 val"] },
      ],
      acceptanceCriteria: [
        "DataLoader 產生正確 shape 的 (x, y) tensor",
        "y 是 x 向右偏移 1 位的結果",
        "train/val split 比例正確，無資料洩漏",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "GPT-2 論文，描述了 next-token prediction 的訓練方式", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
      ],
    },
  ],
};

const phase2ContentZhTW: PhaseContent = {
  phaseId: 2,
  color: "#8B5CF6",
  accent: "#A78BFA",
  lessons: [
    {
      phaseId: 2, lessonId: 1,
      title: "Attention Mechanisms",
      subtitle: "從圖書館比喻到數學實作——理解注意力的核心",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 Query、Key、Value 的語義含義",
        "實作 Scaled Dot-Product Attention",
        "理解 Causal Masking 的原理與實作",
        "組裝 Multi-Head Attention 模組",
      ],
      sections: [
        {
          title: "The Library Analogy",
          blocks: [
            { type: "paragraph", text: "到目前為止，每個 token 都是「孤島」——它只知道自己是什麼，在哪裡，但完全不知道周圍有什麼其他 token。要理解語言，token 們必須能互相「交流」。Attention 就是拿掉眼罩和耳塞的魔法。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Attention 是 Transformer 的靈魂。在 RNN 時代，遠距離的詞要一步一步傳遞資訊，資訊會衰減。Attention 讓每個詞都能「直接」跟序列中任何其他詞對話——這就像從排隊傳話升級成視訊會議。" },
            { type: "paragraph", text: "理解 Attention 最好的方式是圖書館比喻。想像你走進一間巨大的圖書館，帶著一個問題（Query）。書架上每本書都有索引卡（Key），書本身是內容（Value）。你將你的問題與每本書的索引卡做比較（dot product），找到最相關的書，然後根據相關度加權混合這些書的內容，得到你的答案。" },
            { type: "list", ordered: true, items: [
              "Query (Q): 你的問題——「我在找什麼？」",
              "Key (K): 書的索引——「這本書是關於什麼的？」",
              "Value (V): 書的內容——「這本書說了什麼？」",
              "Attention Score: Q 和 K 的相似度",
              "Attention Weight: softmax 後的分數",
            ]},
          ],
        },
        {
          title: "Scaled Dot-Product Attention",
          blocks: [
            { type: "diagram", content: "Attention(Q, K, V) = softmax(QK^T / √d_k) · V\n\n其中：\n  Q: (seq_len, d_k) — Query 矩陣\n  K: (seq_len, d_k) — Key 矩陣  \n  V: (seq_len, d_v) — Value 矩陣\n  d_k: Key 的維度" },
            { type: "heading", level: 3, text: "為什麼要除以 √d_k？" },
            { type: "paragraph", text: "當 d_k 很大時，QK^T 的值也會很大，讓 softmax 的輸出接近 one-hot（梯度幾乎為零），導致訓練困難。除以 √d_k 將方差標準化回 1。" },
            { type: "code", language: "python", code: "def scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    weights = torch.softmax(scores, dim=-1)\n    return torch.matmul(weights, V), weights" },
            { type: "callout", variant: "tip", text: "💡 講師心得：假設 Q 和 K 的每個元素都是均值 0、方差 1 的隨機變數，那麼它們的點積的方差是 d_k。除以 √d_k 把方差拉回到 1，讓 softmax 的輸入保持在合理範圍。這不是經驗性的 trick，而是有嚴格數學依據的設計。" },
          ],
        },
        {
          title: "Causal Masking",
          blocks: [
            { type: "paragraph", text: "在 GPT 這樣的自回歸模型中，生成 token t_i 時只能看到 t_0 到 t_{i-1}。Causal mask 是一個下三角矩陣：位置 (i, j) 為 1 當 j ≤ i，否則為 0。" },
            { type: "diagram", content: "Causal Mask (seq_len=4):\n\n  t₀  t₁  t₂  t₃\nt₀ [1   0   0   0]    t₀ 只能看自己\nt₁ [1   1   0   0]    t₁ 可以看 t₀, t₁\nt₂ [1   1   1   0]    t₂ 可以看 t₀, t₁, t₂\nt₃ [1   1   1   1]    t₃ 可以看所有" },
            { type: "code", language: "python", code: "mask = torch.tril(torch.ones(seq_len, seq_len))\nscores = scores.masked_fill(mask == 0, float('-inf'))" },
            { type: "callout", variant: "warning", text: "masked_fill 要在 softmax 之前做！如果在 softmax 之後把權重設為 0，剩餘權重的和就不再是 1，打破了概率分布的性質。" },
          ],
        },
        {
          title: "Multi-Head Attention",
          blocks: [
            { type: "paragraph", text: "單一的 attention head 只能學到一種「注意力模式」。Multi-head attention 讓模型同時關注不同的特徵：有的 head 可能關注語法關係，有的關注語義相似性，有的關注位置相鄰性。" },
            { type: "code", language: "python", code: "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.n_heads = n_heads\n        self.d_k = d_model // n_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):  # x: (batch, seq_len, d_model)\n        B, T, C = x.shape\n        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        attn_out, _ = scaled_dot_product_attention(Q, K, V, causal_mask)\n        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)\n        return self.W_o(out)" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Multi-head attention 的精髓不只是「多個 head」，而是「子空間分解」。研究發現，有的 head 專注語法結構，有的追蹤指代消解，有的捕捉局部相鄰性。這就像一個團隊，每個人負責觀察不同的面向。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：如果我們的模型有 12 個 attention head，每個 head 維度是 64，那 d_model = 768。現在假設我們把 head 數量從 12 增加到 24（每個 head 維度降為 32），模型的總參數量不變——但效果會一樣嗎？更多的 head 一定更好嗎？" },
          ],
        },
      ],
      exercises: [
        { id: "qkv_projection", title: "TODO 1: QKV Projection", description: "實作 Q、K、V 的線性投影，並 reshape 為多頭格式。", labFile: "labs/phase2/attention.py", hints: ["nn.Linear(d_model, d_model) 做投影", "view(B, T, n_heads, d_k).transpose(1, 2) 做 reshape"] },
        { id: "dot_product", title: "TODO 2: Scaled Dot-Product", description: "計算 QK^T / √d_k 並返回 attention scores。", labFile: "labs/phase2/attention.py", hints: ["torch.matmul(Q, K.transpose(-2, -1))", "除以 math.sqrt(d_k)"], pseudocode: "scores = Q @ K^T / sqrt(d_k)" },
        { id: "causal_mask", title: "TODO 3: Apply Causal Mask", description: "建立下三角 causal mask 並應用到 attention scores。", labFile: "labs/phase2/attention.py", hints: ["torch.tril(torch.ones(T, T))", "scores.masked_fill(mask == 0, -inf)"] },
        { id: "attention_weights", title: "TODO 4: Compute Attention Weights", description: "對 masked scores 做 softmax，然後與 V 相乘。", labFile: "labs/phase2/attention.py", hints: ["softmax(scores, dim=-1)", "結果 @ V"] },
        { id: "concat_heads", title: "TODO 5: Concatenate Heads", description: "將多頭輸出拼接回 (B, T, d_model) 形狀，通過輸出投影。", labFile: "labs/phase2/attention.py", hints: ["transpose(1,2).contiguous().view(B, T, C)", "最後過 self.W_o"] },
      ],
      acceptanceCriteria: [
        "attention weights 每行和為 1（softmax 性質）",
        "causal mask 確保位置 i 不能 attend 到位置 j > i",
        "多頭輸出 shape 為 (batch, seq_len, d_model)",
      ],
      references: [
        { title: "Attention Is All You Need", description: "Vaswani et al. 2017 — Transformer 原始論文，提出 multi-head attention", url: "https://arxiv.org/abs/1706.03762" },
        { title: "The Illustrated Transformer", description: "Jay Alammar 的經典圖解文章，用動畫解釋 attention 機制", url: "https://jalammar.github.io/illustrated-transformer/" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// zh-CN variants
// ═══════════════════════════════════════════════════════════════════

const phase1ContentZhCN: PhaseContent = {
  phaseId: 1,
  color: "#3B82F6",
  accent: "#60A5FA",
  lessons: [
    {
      phaseId: 1, lessonId: 1,
      title: "BPE Tokenizer",
      subtitle: "从字符到子词——构建你自己的 Byte-Pair Encoding 分词器",
      type: "concept",
      duration: "45 min",
      objectives: [
        "理解字符级、词级与子词级分词的差异与取舍",
        "完整实现 BPE 算法的训练过程（merge 规则学习）",
        "实现 encode 与 decode 函数，确保完美的 roundtrip",
      ],
      sections: [
        {
          title: "Why Tokenization Matters",
          blocks: [
            { type: "paragraph", text: "你有没有想过，ChatGPT 看到的「文字」跟你看到的其实不一样？你看到的是「Hello World」两个英文单词。但 GPT 看到的是 [15496, 2159] 两个数字。从人类的文字到模型的数字，中间就需要一个「翻译官」——这就是 Tokenizer。今天我们要从零搭建这个翻译官。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：Tokenizer 是 LLM 最被低估的组件。很多 LLM 看起来「愚蠢」的行为——数不清单词里有几个字母、做不好简单算术——其实是 tokenization 的问题。模型看到的不是你以为的那些字符，而是被切割过的子词碎片。理解 tokenizer，就理解了模型「视力」的极限。" },
            { type: "paragraph", text: "大型语言模型不直接处理文字——它们处理的是整数序列。Tokenization 就是将原始文字转换成模型能理解的数字序列的过程。选择什么样的 tokenization 策略，直接影响模型的词汇表大小、序列长度、以及对未知词的处理能力。" },
            { type: "table", headers: ["策略", "词汇表大小", "序列长度", "OOV 问题"], rows: [
              ["Character-level", "~256", "非常长", "无"],
              ["Word-level", "100K+", "短", "严重"],
              ["BPE (Subword)", "~50K", "适中", "几乎无"],
            ]},
            { type: "callout", variant: "info", text: "GPT-2 使用约 50,257 个 BPE token。这个数字在词汇表大小（影响 embedding 矩阵大小）和序列长度（影响 attention 计算量）之间取得了良好的平衡。" },
            { type: "paragraph", text: "BPE 的核心思想很简单：从字符开始，反复合并最常出现的相邻 pair，直到达到目标词汇表大小。这让常见词保持完整（如 'the'），同时罕见词被拆成有意义的子片段（如 'tokenization' → 'token' + 'ization'）。" },
          ],
        },
        {
          title: "BPE Algorithm Step by Step",
          blocks: [
            { type: "paragraph", text: "BPE 训练过程可以分为三个核心步骤：建立初始词汇表、统计 pair 频率、执行合并。我们反复执行后两步，直到词汇表达到目标大小。" },
            { type: "heading", level: 3, text: "Step 1: 初始化" },
            { type: "paragraph", text: "将训练文本转换为 UTF-8 bytes 序列。初始词汇表就是 0-255 这 256 个 byte 值。" },
            { type: "heading", level: 3, text: "Step 2: 统计相邻 Pair" },
            { type: "code", language: "python", code: "def get_pair_counts(token_ids: list[int]) -> dict[tuple[int,int], int]:\n    counts = {}\n    for i in range(len(token_ids) - 1):\n        pair = (token_ids[i], token_ids[i + 1])\n        counts[pair] = counts.get(pair, 0) + 1\n    return counts" },
            { type: "heading", level: 3, text: "Step 3: 合并最高频 Pair" },
            { type: "diagram", content: "训练语料: 'aaabdaaabac'\n\n初始: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]\n\nMerge 1: (97,97)→256  →  [256, 97, 98, 100, 256, 97, 98, 97, 99]\nMerge 2: (256,97)→257 →  [257, 98, 100, 257, 98, 97, 99]\nMerge 3: (257,98)→258 →  [258, 100, 258, 97, 99]" },
            { type: "callout", variant: "tip", text: "合并顺序很重要！encode 时必须按照训练时学到的顺序依次应用 merge 规则。这就是为什么我们需要保存一个有序的 merges 列表。" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：BPE 本质上是一个压缩算法，被巧妙地借用到了 NLP 领域。它在 1994 年被发明用于数据压缩（Philip Gage），2016 年 Sennrich 等人发现它完美适用于子词分割。这个跨领域的借用提醒我们：好的工程方案往往来自意想不到的地方。" },
          ],
        },
        {
          title: "Encoding & Decoding",
          blocks: [
            { type: "paragraph", text: "训练完成后，我们得到了一个有序的 merge 规则列表。Encoding 就是将文本先转为 bytes，再按顺序套用每条 merge 规则。Decoding 则是将 token ID 序列转回 bytes 再解码为文字。" },
            { type: "code", language: "python", code: "def encode(self, text: str) -> list[int]:\n    tokens = list(text.encode('utf-8'))  # 转为 byte list\n    for (p0, p1), new_id in self.merges.items():\n        i = 0\n        new_tokens = []\n        while i < len(tokens):\n            if i < len(tokens)-1 and tokens[i]==p0 and tokens[i+1]==p1:\n                new_tokens.append(new_id)\n                i += 2\n            else:\n                new_tokens.append(tokens[i])\n                i += 1\n        tokens = new_tokens\n    return tokens" },
            { type: "code", language: "python", code: "def decode(self, token_ids: list[int]) -> str:\n    byte_list = b''.join(self.vocab[t] for t in token_ids)\n    return byte_list.decode('utf-8', errors='replace')" },
            { type: "callout", variant: "warning", text: "encode(decode(ids)) 不一定等于 ids，但 decode(encode(text)) 必须等于原始 text。这是最重要的不变量。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：我们的 BPE tokenizer 是 byte-level 的，可以处理任何 UTF-8 文字。但如果要处理中文、日文、韩文这类 CJK 字符，BPE 的效率如何？一个中文字占 3 个 UTF-8 bytes，这对词汇表大小和序列长度有什么影响？" },
          ],
        },
      ],
      exercises: [
        { id: "build_initial_vocab", title: "TODO 1: build_initial_vocab", description: "建立初始词汇表，包含 0-255 的所有 byte 值。每个 entry 是 token_id → bytes 的映射。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["使用 dict comprehension", "bytes([i]) 可以将整数转为单一 byte"], pseudocode: "vocab = {}\nfor i in 0..255:\n  vocab[i] = bytes([i])\nreturn vocab" },
        { id: "get_pair_counts", title: "TODO 2: get_pair_counts", description: "统计 token 序列中所有相邻 pair 的出现次数。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["遍历 range(len(ids)-1)", "用 dict 存储计数"] },
        { id: "merge_pair", title: "TODO 3: merge_pair", description: "在 token 序列中，将所有 (p0, p1) pair 替换为 new_id。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["用 while loop 而非 for loop", "找到 pair 就 append new_id 并 i+=2"] },
        { id: "train", title: "TODO 4: train", description: "执行 BPE 训练循环：反复找最高频 pair → 合并 → 更新词汇表，直到达到目标大小。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["num_merges = target_vocab_size - 256", "每次 merge 后新 token ID = 256 + merge_count"] },
        { id: "encode", title: "TODO 5: encode", description: "将文本编码为 token ID 序列，按训练顺序应用所有 merge 规则。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["先转 UTF-8 bytes", "按 merges 的顺序依次应用每条规则"] },
        { id: "decode", title: "TODO 6: decode", description: "将 token ID 序列解码回文本字符串。", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["查表获取每个 token 的 bytes", "用 b''.join() 拼接后 decode('utf-8')"] },
      ],
      acceptanceCriteria: [
        "vocab_size 可自定义设置，且训练后词汇表大小正确",
        "decode(encode(text)) == text 对所有 UTF-8 文本成立",
        "get_pair_counts 正确统计所有相邻 pair",
        "merge 顺序与训练顺序一致",
      ],
      references: [
        { title: "Neural Machine Translation of Rare Words with Subword Units", description: "Sennrich et al. 2016 — BPE 的原始论文，提出将 BPE 压缩算法用于 NLP", url: "https://arxiv.org/abs/1508.07909" },
        { title: "tiktoken", description: "OpenAI 的高效 BPE 实现，GPT-4 使用的 tokenizer", url: "https://github.com/openai/tiktoken" },
      ],
    },
    {
      phaseId: 1, lessonId: 2,
      title: "DataLoader for LLM Training",
      subtitle: "建立高效的数据管道——从原始文本到训练批次",
      type: "concept",
      duration: "30 min",
      objectives: [
        "理解 next-token prediction 的核心训练目标",
        "实现 sliding-window 方式建立 (x, y) 训练对",
        "建立 train / validation 数据分割",
      ],
      sections: [
        {
          title: "Next-Token Prediction",
          blocks: [
            { type: "paragraph", text: "Tokenizer 搞定了，模型现在可以「看懂」文字了。但光有翻译官还不够——我们还需要一个「厨师」把原始食材加工成模型能消化的「便当」。这就是 DataLoader 的角色。模型到底要学什么？答案是：给定前面的 token 序列，预测下一个 token。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：Next-token prediction 是整个 LLM 世界的基石，但初学者常低估它的威力。看似只是「猜下一个字」，但要做好这件事，模型必须学会语法、语义、逻辑推理、甚至世界知识。" },
            { type: "paragraph", text: "具体来说，对于一个长度为 T 的序列 [t₀, t₁, ..., t_{T-1}]，我们构造：\n- 输入 x = [t₀, t₁, ..., t_{T-2}]\n- 目标 y = [t₁, t₂, ..., t_{T-1}]\n\ny 就是 x 向右偏移一个位置。" },
            { type: "callout", variant: "info", text: "由于 causal masking，模型在位置 i 只能看到位置 0..i 的 token。所以每个训练样本实际上提供了 context_length 个训练信号。" },
          ],
        },
        {
          title: "Sliding Window",
          blocks: [
            { type: "paragraph", text: "我们用固定大小的滑动窗口从已 tokenize 的文本中提取训练样本。窗口大小 = context_length + 1。每个窗口产生一对 (x, y)：x 是前 context_length 个 token，y 是后 context_length 个 token（偏移 1）。" },
            { type: "diagram", content: "Token sequence: [a, b, c, d, e, f, g, h, i, j]\ncontext_length = 4, stride = 4\n\nWindow 1: [a, b, c, d, e]  →  x=[a,b,c,d]  y=[b,c,d,e]\nWindow 2: [e, f, g, h, i]  →  x=[e,f,g,h]  y=[f,g,h,i]" },
            { type: "code", language: "python", code: "class TextDataset(Dataset):\n    def __init__(self, token_ids: list[int], context_length: int, stride: int):\n        self.x = []\n        self.y = []\n        for i in range(0, len(token_ids) - context_length, stride):\n            self.x.append(torch.tensor(token_ids[i:i+context_length]))\n            self.y.append(torch.tensor(token_ids[i+1:i+context_length+1]))\n\n    def __len__(self): return len(self.x)\n    def __getitem__(self, idx): return self.x[idx], self.y[idx]" },
          ],
        },
        {
          title: "Token & Position Embeddings",
          blocks: [
            { type: "paragraph", text: "Token ID 只是整数，模型需要的是连续向量。Token Embedding 是一个 V×D 的查找表，每个 token ID 对应一个 D 维向量。Position Embedding 是另一个 T×D 的查找表，每个位置对应一个 D 维向量。两者相加得到最终的输入表示。" },
            { type: "code", language: "python", code: "class Embeddings(nn.Module):\n    def __init__(self, vocab_size, embed_dim, context_length):\n        super().__init__()\n        self.token_emb = nn.Embedding(vocab_size, embed_dim)\n        self.pos_emb = nn.Embedding(context_length, embed_dim)\n\n    def forward(self, x):  # x: (batch, seq_len)\n        tok = self.token_emb(x)\n        pos = self.pos_emb(torch.arange(x.size(1), device=x.device))\n        return tok + pos" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：Token embedding 和 position embedding 的相加（而非拼接）是一个优雅的设计。两个 embedding 共享同一个向量空间，相加后模型可以同时利用「这是什么词」和「这个词在哪里」的信息。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：我们用「相加」而非「拼接」来合并 token embedding 和 position embedding。但这样做不会互相干扰吗？模型怎么区分哪些信息来自 token、哪些来自 position？（提示：想想高维空间中两个随机向量的夹角。）" },
          ],
        },
      ],
      exercises: [
        { id: "dataset_init", title: "TODO 1: TextDataset.__init__", description: "实现 sliding-window 逻辑，从 token 序列中提取 (x, y) 对。", labFile: "labs/phase1/dataloader.py", hints: ["用 range(0, len(ids) - context_length, stride) 作为起始索引", "x 和 y 之间偏移 1 个位置"] },
        { id: "dataset_len", title: "TODO 2: TextDataset.__len__", description: "返回数据集中的样本数量。", labFile: "labs/phase1/dataloader.py", hints: ["直接返回 self.x 的长度"] },
        { id: "dataset_getitem", title: "TODO 3: TextDataset.__getitem__", description: "返回第 idx 个 (x, y) 训练对。", labFile: "labs/phase1/dataloader.py", hints: ["return self.x[idx], self.y[idx]"] },
        { id: "create_dataloaders", title: "TODO 4: create_dataloaders", description: "将文本分割为 train/val，建立 DataLoader。", labFile: "labs/phase1/dataloader.py", hints: ["先 tokenize 整个文本", "用前 90% 做 train，后 10% 做 val"] },
      ],
      acceptanceCriteria: [
        "DataLoader 产生正确 shape 的 (x, y) tensor",
        "y 是 x 向右偏移 1 位的结果",
        "train/val split 比例正确，无数据泄漏",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "GPT-2 论文，描述了 next-token prediction 的训练方式", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
      ],
    },
  ],
};

const phase2ContentZhCN: PhaseContent = {
  phaseId: 2,
  color: "#8B5CF6",
  accent: "#A78BFA",
  lessons: [
    {
      phaseId: 2, lessonId: 1,
      title: "Attention Mechanisms",
      subtitle: "从图书馆比喻到数学实现——理解注意力的核心",
      type: "concept",
      duration: "60 min",
      objectives: [
        "理解 Query、Key、Value 的语义含义",
        "实现 Scaled Dot-Product Attention",
        "理解 Causal Masking 的原理与实现",
        "组装 Multi-Head Attention 模块",
      ],
      sections: [
        {
          title: "The Library Analogy",
          blocks: [
            { type: "paragraph", text: "到目前为止，每个 token 都是「孤岛」——它只知道自己是什么，在哪里，但完全不知道周围有什么其他 token。要理解语言，token 们必须能互相「交流」。Attention 就是拿掉眼罩和耳塞的魔法。" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：Attention 是 Transformer 的灵魂。在 RNN 时代，远距离的词要一步一步传递信息，信息会衰减。Attention 让每个词都能「直接」跟序列中任何其他词对话——这就像从排队传话升级成视频会议。" },
            { type: "paragraph", text: "理解 Attention 最好的方式是图书馆比喻。你带着一个问题（Query）走进图书馆。书架上每本书都有索引卡（Key），书本身是内容（Value）。你将问题与每本书的索引卡做比较（dot product），找到最相关的书，然后根据相关度加权混合内容，得到答案。" },
            { type: "list", ordered: true, items: [
              "Query (Q): 你的问题——「我在找什么？」",
              "Key (K): 书的索引——「这本书是关于什么的？」",
              "Value (V): 书的内容——「这本书说了什么？」",
              "Attention Score: Q 和 K 的相似度",
              "Attention Weight: softmax 后的分数",
            ]},
          ],
        },
        {
          title: "Scaled Dot-Product Attention",
          blocks: [
            { type: "diagram", content: "Attention(Q, K, V) = softmax(QK^T / √d_k) · V\n\n其中：\n  Q: (seq_len, d_k) — Query 矩阵\n  K: (seq_len, d_k) — Key 矩阵  \n  V: (seq_len, d_v) — Value 矩阵\n  d_k: Key 的维度" },
            { type: "heading", level: 3, text: "为什么要除以 √d_k？" },
            { type: "paragraph", text: "当 d_k 很大时，QK^T 的值也会很大，让 softmax 的输出接近 one-hot（梯度几乎为零），导致训练困难。除以 √d_k 将方差标准化回 1。" },
            { type: "code", language: "python", code: "def scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    weights = torch.softmax(scores, dim=-1)\n    return torch.matmul(weights, V), weights" },
            { type: "callout", variant: "tip", text: "💡 讲师心得：假设 Q 和 K 的每个元素都是均值 0、方差 1 的随机变量，那么它们的点积的方差是 d_k。除以 √d_k 把方差拉回到 1，让 softmax 的输入保持在合理范围。这不是经验性的 trick，而是有严格数学依据的设计。" },
          ],
        },
        {
          title: "Causal Masking",
          blocks: [
            { type: "paragraph", text: "在 GPT 这样的自回归模型中，生成 token t_i 时只能看到 t_0 到 t_{i-1}。Causal mask 是一个下三角矩阵：位置 (i, j) 为 1 当 j ≤ i，否则为 0。" },
            { type: "diagram", content: "Causal Mask (seq_len=4):\n\n  t₀  t₁  t₂  t₃\nt₀ [1   0   0   0]    t₀ 只能看自身\nt₁ [1   1   0   0]    t₁ 可以看 t₀, t₁\nt₂ [1   1   1   0]    t₂ 可以看 t₀, t₁, t₂\nt₃ [1   1   1   1]    t₃ 可以看所有" },
            { type: "code", language: "python", code: "mask = torch.tril(torch.ones(seq_len, seq_len))\nscores = scores.masked_fill(mask == 0, float('-inf'))" },
            { type: "callout", variant: "warning", text: "masked_fill 要在 softmax 之前做！如果在 softmax 之后把权重设为 0，剩余权重的和就不再是 1，打破了概率分布的性质。" },
          ],
        },
        {
          title: "Multi-Head Attention",
          blocks: [
            { type: "paragraph", text: "单一的 attention head 只能学到一种「注意力模式」。Multi-head attention 让模型同时关注不同的特征：有的 head 可能关注语法关系，有的关注语义相似性，有的关注位置相邻性。" },
            { type: "code", language: "python", code: "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.n_heads = n_heads\n        self.d_k = d_model // n_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):  # x: (batch, seq_len, d_model)\n        B, T, C = x.shape\n        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        attn_out, _ = scaled_dot_product_attention(Q, K, V, causal_mask)\n        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)\n        return self.W_o(out)" },
            { type: "callout", variant: "quote", text: "💡 讲师心得：Multi-head attention 的精髓不只是「多个 head」，而是「子空间分解」。研究发现，有的 head 专注语法结构，有的追踪指代消解，有的捕捉局部相邻性。" },
            { type: "callout", variant: "quote", text: "🤔 思考题：如果我们的模型有 12 个 attention head，每个 head 维度是 64，那 d_model = 768。现在假设我们把 head 数量从 12 增加到 24（每个 head 维度降为 32），模型的总参数量不变——但效果会一样吗？更多的 head 一定更好吗？" },
          ],
        },
      ],
      exercises: [
        { id: "qkv_projection", title: "TODO 1: QKV Projection", description: "实现 Q、K、V 的线性投影，并 reshape 为多头格式。", labFile: "labs/phase2/attention.py", hints: ["nn.Linear(d_model, d_model) 做投影", "view(B, T, n_heads, d_k).transpose(1, 2) 做 reshape"] },
        { id: "dot_product", title: "TODO 2: Scaled Dot-Product", description: "计算 QK^T / √d_k 并返回 attention scores。", labFile: "labs/phase2/attention.py", hints: ["torch.matmul(Q, K.transpose(-2, -1))", "除以 math.sqrt(d_k)"], pseudocode: "scores = Q @ K^T / sqrt(d_k)" },
        { id: "causal_mask", title: "TODO 3: Apply Causal Mask", description: "建立下三角 causal mask 并应用到 attention scores。", labFile: "labs/phase2/attention.py", hints: ["torch.tril(torch.ones(T, T))", "scores.masked_fill(mask == 0, -inf)"] },
        { id: "attention_weights", title: "TODO 4: Compute Attention Weights", description: "对 masked scores 做 softmax，然后与 V 相乘。", labFile: "labs/phase2/attention.py", hints: ["softmax(scores, dim=-1)", "结果 @ V"] },
        { id: "concat_heads", title: "TODO 5: Concatenate Heads", description: "将多头输出拼接回 (B, T, d_model) 形状，通过输出投影。", labFile: "labs/phase2/attention.py", hints: ["transpose(1,2).contiguous().view(B, T, C)", "最后过 self.W_o"] },
      ],
      acceptanceCriteria: [
        "attention weights 每行和为 1（softmax 性质）",
        "causal mask 确保位置 i 不能 attend 到位置 j > i",
        "多头输出 shape 为 (batch, seq_len, d_model)",
      ],
      references: [
        { title: "Attention Is All You Need", description: "Vaswani et al. 2017 — Transformer 原始论文，提出 multi-head attention", url: "https://arxiv.org/abs/1706.03762" },
        { title: "The Illustrated Transformer", description: "Jay Alammar 的经典图解文章，用动画解释 attention 机制", url: "https://jalammar.github.io/illustrated-transformer/" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// English variants
// ═══════════════════════════════════════════════════════════════════

const phase1ContentEn: PhaseContent = {
  phaseId: 1,
  color: "#3B82F6",
  accent: "#60A5FA",
  lessons: [
    {
      phaseId: 1, lessonId: 1,
      title: "BPE Tokenizer",
      subtitle: "From Characters to Subwords — Build Your Own Byte-Pair Encoding Tokenizer",
      type: "concept",
      duration: "45 min",
      objectives: [
        "Understand the tradeoffs between character-level, word-level, and subword tokenization",
        "Implement the full BPE training algorithm (learning merge rules)",
        "Implement encode and decode functions with a perfect roundtrip guarantee",
      ],
      sections: [
        {
          title: "Why Tokenization Matters",
          blocks: [
            { type: "paragraph", text: "Have you ever wondered that ChatGPT doesn't \"see\" text the same way you do? You see \"Hello World\" — two English words. GPT sees [15496, 2159] — two integers. To get from human text to model numbers, you need a translator. That's the Tokenizer. Today we're going to build one from scratch." },
            { type: "callout", variant: "quote", text: "Instructor's Note: The tokenizer is the most underrated component in LLMs. Many of LLM's \"dumb\" behaviors — miscounting letters in a word, struggling with simple arithmetic — are actually the tokenizer's fault. The model doesn't see the characters you think it does; it sees chopped-up subword fragments. Understanding the tokenizer means understanding the limits of the model's \"vision.\"" },
            { type: "paragraph", text: "Large language models don't process text directly — they process sequences of integers. Tokenization converts raw text into numerical sequences the model can understand. The tokenization strategy you choose directly affects vocabulary size, sequence length, and the ability to handle unknown words." },
            { type: "table", headers: ["Strategy", "Vocabulary Size", "Sequence Length", "OOV Problem"], rows: [
              ["Character-level", "~256", "Very long", "None"],
              ["Word-level", "100K+", "Short", "Severe"],
              ["BPE (Subword)", "~50K", "Moderate", "Almost none"],
            ]},
            { type: "callout", variant: "info", text: "GPT-2 uses ~50,257 BPE tokens — a sweet spot between vocabulary size (affects embedding matrix size) and sequence length (affects attention computation)." },
            { type: "paragraph", text: "BPE's core idea is simple: start from characters, repeatedly merge the most frequent adjacent pair, until reaching the target vocabulary size. Common words stay intact (like 'the'), while rare words get split into meaningful subpieces (e.g. 'tokenization' → 'token' + 'ization')." },
          ],
        },
        {
          title: "BPE Algorithm Step by Step",
          blocks: [
            { type: "paragraph", text: "BPE training has three core steps: build the initial vocabulary, count pair frequencies, execute merges. We repeat the last two until the vocabulary reaches the target size." },
            { type: "heading", level: 3, text: "Step 1: Initialize" },
            { type: "paragraph", text: "Convert the training text into a UTF-8 byte sequence. The initial vocabulary is the 256 byte values 0–255. Each character (or each byte of a multi-byte character) is its own token." },
            { type: "heading", level: 3, text: "Step 2: Count Adjacent Pairs" },
            { type: "code", language: "python", code: "def get_pair_counts(token_ids: list[int]) -> dict[tuple[int,int], int]:\n    counts = {}\n    for i in range(len(token_ids) - 1):\n        pair = (token_ids[i], token_ids[i + 1])\n        counts[pair] = counts.get(pair, 0) + 1\n    return counts" },
            { type: "heading", level: 3, text: "Step 3: Merge the Most Frequent Pair" },
            { type: "diagram", content: "Training corpus: 'aaabdaaabac'\n\nInitial: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]\n\nMerge 1: (97,97)→256  →  [256, 97, 98, 100, 256, 97, 98, 97, 99]\nMerge 2: (256,97)→257 →  [257, 98, 100, 257, 98, 97, 99]\nMerge 3: (257,98)→258 →  [258, 100, 258, 97, 99]" },
            { type: "callout", variant: "tip", text: "Merge order matters! During encoding, you must apply merge rules in the exact order they were learned during training. That's why we store an ordered list of merges." },
            { type: "callout", variant: "tip", text: "Instructor's Note: BPE is fundamentally a compression algorithm, cleverly borrowed for NLP. It was invented in 1994 for data compression (Philip Gage), then in 2016 Sennrich et al. discovered it was perfect for subword segmentation. This cross-domain insight reminds us: good engineering solutions often come from unexpected places." },
          ],
        },
        {
          title: "Encoding & Decoding",
          blocks: [
            { type: "paragraph", text: "After training, you have an ordered list of merge rules. Encoding converts text to bytes, then applies each merge rule in order. Decoding converts a token ID sequence back to bytes, then decodes to text." },
            { type: "code", language: "python", code: "def encode(self, text: str) -> list[int]:\n    tokens = list(text.encode('utf-8'))  # convert to byte list\n    for (p0, p1), new_id in self.merges.items():\n        i = 0\n        new_tokens = []\n        while i < len(tokens):\n            if i < len(tokens)-1 and tokens[i]==p0 and tokens[i+1]==p1:\n                new_tokens.append(new_id)\n                i += 2\n            else:\n                new_tokens.append(tokens[i])\n                i += 1\n        tokens = new_tokens\n    return tokens" },
            { type: "code", language: "python", code: "def decode(self, token_ids: list[int]) -> str:\n    byte_list = b''.join(self.vocab[t] for t in token_ids)\n    return byte_list.decode('utf-8', errors='replace')" },
            { type: "callout", variant: "warning", text: "encode(decode(ids)) won't necessarily equal ids, but decode(encode(text)) must equal the original text. This is the most important invariant." },
            { type: "callout", variant: "quote", text: "Think About It: Our byte-level BPE tokenizer can handle any UTF-8 text. But for CJK characters (Chinese, Japanese, Korean), how efficient is BPE? A single Chinese character takes 3 UTF-8 bytes — what does that mean for vocabulary size and sequence length? What special treatment does GPT-4's tokenizer use to optimize multilingual efficiency?" },
          ],
        },
      ],
      exercises: [
        { id: "build_initial_vocab", title: "TODO 1: build_initial_vocab", description: "Build the initial vocabulary containing all 256 byte values (0–255). Each entry maps token_id → bytes.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["Use a dict comprehension", "bytes([i]) converts an integer to a single byte"], pseudocode: "vocab = {}\nfor i in 0..255:\n  vocab[i] = bytes([i])\nreturn vocab" },
        { id: "get_pair_counts", title: "TODO 2: get_pair_counts", description: "Count all adjacent pair occurrences in the token sequence.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["Iterate over range(len(ids)-1)", "Use a dict to store counts"] },
        { id: "merge_pair", title: "TODO 3: merge_pair", description: "In the token sequence, replace all (p0, p1) pairs with new_id.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["Use a while loop, not a for loop (length changes after replacement)", "When you find a pair, append new_id and skip two positions (i+=2)"] },
        { id: "train", title: "TODO 4: train", description: "Run the BPE training loop: repeatedly find the highest-frequency pair → merge → update vocabulary, until reaching the target size.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["num_merges = target_vocab_size - 256", "Each merge's new token ID = 256 + merge_count"] },
        { id: "encode", title: "TODO 5: encode", description: "Encode text into a token ID sequence, applying all merge rules in training order.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["First convert to UTF-8 bytes", "Apply each merge rule in order"] },
        { id: "decode", title: "TODO 6: decode", description: "Decode a token ID sequence back to a text string.", labFile: "labs/phase1/bpe_tokenizer.py", hints: ["Look up each token's bytes", "Join with b''.join() then decode('utf-8')"] },
      ],
      acceptanceCriteria: [
        "vocab_size is configurable; vocabulary size is correct after training",
        "decode(encode(text)) == text for all UTF-8 text",
        "get_pair_counts correctly counts all adjacent pairs",
        "Merge order matches the training order",
      ],
      references: [
        { title: "Neural Machine Translation of Rare Words with Subword Units", description: "Sennrich et al. 2016 — The original BPE paper applying the compression algorithm to NLP", url: "https://arxiv.org/abs/1508.07909" },
        { title: "tiktoken", description: "OpenAI's efficient BPE implementation — the tokenizer used by GPT-4", url: "https://github.com/openai/tiktoken" },
      ],
    },
    {
      phaseId: 1, lessonId: 2,
      title: "DataLoader for LLM Training",
      subtitle: "Building an Efficient Data Pipeline — From Raw Text to Training Batches",
      type: "concept",
      duration: "30 min",
      objectives: [
        "Understand the core training objective: next-token prediction",
        "Implement sliding-window (x, y) training pair construction",
        "Build a train/validation data split",
      ],
      sections: [
        {
          title: "Next-Token Prediction",
          blocks: [
            { type: "paragraph", text: "The tokenizer is done — the model can now \"read\" text. But we still need a chef to turn the raw ingredients into a meal the model can digest. That's the DataLoader's job. What exactly does the model learn? The answer: given a sequence of tokens, predict the next one." },
            { type: "callout", variant: "quote", text: "Instructor's Note: Next-token prediction is the foundation of the entire LLM world, but beginners often underestimate its power. \"Guess the next word\" sounds simple — but to do it well, the model must learn grammar, semantics, logical reasoning, and even world knowledge." },
            { type: "paragraph", text: "Concretely, for a sequence of length T [t₀, t₁, ..., t_{T-1}], we construct:\n- Input x = [t₀, t₁, ..., t_{T-2}]\n- Target y = [t₁, t₂, ..., t_{T-1}]\n\ny is x shifted right by one position." },
            { type: "callout", variant: "info", text: "Due to causal masking, the model at position i can only see tokens 0..i. So each training sample actually provides context_length training signals." },
          ],
        },
        {
          title: "Sliding Window",
          blocks: [
            { type: "paragraph", text: "We extract training samples from tokenized text using a fixed-size sliding window. Window size = context_length + 1. Each window yields a (x, y) pair: x is the first context_length tokens, y is the next context_length tokens (shifted by 1)." },
            { type: "diagram", content: "Token sequence: [a, b, c, d, e, f, g, h, i, j]\ncontext_length = 4, stride = 4\n\nWindow 1: [a, b, c, d, e]  →  x=[a,b,c,d]  y=[b,c,d,e]\nWindow 2: [e, f, g, h, i]  →  x=[e,f,g,h]  y=[f,g,h,i]" },
            { type: "code", language: "python", code: "class TextDataset(Dataset):\n    def __init__(self, token_ids: list[int], context_length: int, stride: int):\n        self.x = []\n        self.y = []\n        for i in range(0, len(token_ids) - context_length, stride):\n            self.x.append(torch.tensor(token_ids[i:i+context_length]))\n            self.y.append(torch.tensor(token_ids[i+1:i+context_length+1]))\n\n    def __len__(self): return len(self.x)\n    def __getitem__(self, idx): return self.x[idx], self.y[idx]" },
          ],
        },
        {
          title: "Token & Position Embeddings",
          blocks: [
            { type: "paragraph", text: "Token IDs are just integers — the model needs continuous vectors. Token Embedding is a V×D lookup table where each token ID maps to a D-dimensional vector. Position Embedding is a T×D lookup table where each position maps to a D-dimensional vector. The two are added together to get the final input representation." },
            { type: "code", language: "python", code: "class Embeddings(nn.Module):\n    def __init__(self, vocab_size, embed_dim, context_length):\n        super().__init__()\n        self.token_emb = nn.Embedding(vocab_size, embed_dim)\n        self.pos_emb = nn.Embedding(context_length, embed_dim)\n\n    def forward(self, x):  # x: (batch, seq_len)\n        tok = self.token_emb(x)\n        pos = self.pos_emb(torch.arange(x.size(1), device=x.device))\n        return tok + pos  # element-wise addition" },
            { type: "callout", variant: "tip", text: "Instructor's Note: Adding (rather than concatenating) token and position embeddings is an elegant design choice. Both embeddings share the same vector space, and after addition the model can simultaneously use \"what word is this\" and \"where is this word\" information — at no extra parameter cost." },
            { type: "callout", variant: "quote", text: "Think About It: We add token and position embeddings instead of concatenating them. But doesn't that cause interference? How does the model distinguish which information comes from the token vs. the position? (Hint: think about the angle between two random vectors in high-dimensional space.)" },
          ],
        },
      ],
      exercises: [
        { id: "dataset_init", title: "TODO 1: TextDataset.__init__", description: "Implement the sliding-window logic to extract (x, y) pairs from a token sequence.", labFile: "labs/phase1/dataloader.py", hints: ["Use range(0, len(ids) - context_length, stride) for start indices", "x and y are offset by 1 position"] },
        { id: "dataset_len", title: "TODO 2: TextDataset.__len__", description: "Return the number of samples in the dataset.", labFile: "labs/phase1/dataloader.py", hints: ["Return len(self.x)"] },
        { id: "dataset_getitem", title: "TODO 3: TextDataset.__getitem__", description: "Return the idx-th (x, y) training pair.", labFile: "labs/phase1/dataloader.py", hints: ["return self.x[idx], self.y[idx]"] },
        { id: "create_dataloaders", title: "TODO 4: create_dataloaders", description: "Split text into train/val and create DataLoaders.", labFile: "labs/phase1/dataloader.py", hints: ["Tokenize the entire text first", "Use the first 90% for train, last 10% for val"] },
      ],
      acceptanceCriteria: [
        "DataLoader produces (x, y) tensors of the correct shape",
        "y is x shifted right by 1 position",
        "Train/val split ratio is correct with no data leakage",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "GPT-2 paper describing the next-token prediction training objective", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
      ],
    },
  ],
};

const phase2ContentEn: PhaseContent = {
  phaseId: 2,
  color: "#8B5CF6",
  accent: "#A78BFA",
  lessons: [
    {
      phaseId: 2, lessonId: 1,
      title: "Attention Mechanisms",
      subtitle: "From the Library Analogy to Math — Understanding the Core of Attention",
      type: "concept",
      duration: "60 min",
      objectives: [
        "Understand the semantic meaning of Query, Key, and Value",
        "Implement Scaled Dot-Product Attention",
        "Understand the principle and implementation of Causal Masking",
        "Assemble a Multi-Head Attention module",
      ],
      sections: [
        {
          title: "The Library Analogy",
          blocks: [
            { type: "paragraph", text: "So far, every token is an island — it knows what it is and where it is, but has no idea what tokens surround it. To understand language, tokens must be able to \"talk\" to each other. Attention is the magic that removes the blindfolds and earplugs." },
            { type: "callout", variant: "quote", text: "Instructor's Note: Attention is the soul of the Transformer. In the RNN era, distant words had to relay information step by step — like a game of telephone — and the signal degraded. Attention lets every word talk directly to any other word in the sequence. It's like upgrading from passing notes in a chain to a video conference." },
            { type: "paragraph", text: "The best way to understand Attention is the library analogy. Imagine walking into a huge library with a question (Query). Each book on the shelf has an index card (Key), and the book itself is the content (Value). You compare your question against each card (dot product), find the most relevant books, then blend their content weighted by relevance to get your answer." },
            { type: "list", ordered: true, items: [
              "Query (Q): Your question — \"What am I looking for?\"",
              "Key (K): The book's index card — \"What is this book about?\"",
              "Value (V): The book's content — \"What does this book say?\"",
              "Attention Score: Similarity between Q and K",
              "Attention Weight: The score after softmax",
            ]},
          ],
        },
        {
          title: "Scaled Dot-Product Attention",
          blocks: [
            { type: "diagram", content: "Attention(Q, K, V) = softmax(QK^T / √d_k) · V\n\nWhere:\n  Q: (seq_len, d_k) — Query matrix\n  K: (seq_len, d_k) — Key matrix  \n  V: (seq_len, d_v) — Value matrix\n  d_k: Key dimension" },
            { type: "heading", level: 3, text: "Why Divide by √d_k?" },
            { type: "paragraph", text: "When d_k is large, QK^T values grow large too (sum of d_k products), pushing softmax outputs toward one-hot (near-zero gradients) and making training difficult. Dividing by √d_k normalizes the variance back to 1." },
            { type: "code", language: "python", code: "def scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    weights = torch.softmax(scores, dim=-1)\n    return torch.matmul(weights, V), weights" },
            { type: "callout", variant: "tip", text: "Instructor's Note: If each element of Q and K is a random variable with mean 0 and variance 1, their dot product has variance d_k (sum of d_k independent products). Dividing by √d_k brings the variance back to 1, keeping softmax inputs in a sensible range. This isn't an empirical trick — it has a solid mathematical foundation." },
          ],
        },
        {
          title: "Causal Masking",
          blocks: [
            { type: "paragraph", text: "In an autoregressive model like GPT, generating token t_i should only see t_0 through t_{i-1}. The causal mask is a lower-triangular matrix: position (i, j) is 1 when j ≤ i, and 0 otherwise." },
            { type: "diagram", content: "Causal Mask (seq_len=4):\n\n  t₀  t₁  t₂  t₃\nt₀ [1   0   0   0]    t₀ can only see itself\nt₁ [1   1   0   0]    t₁ can see t₀, t₁\nt₂ [1   1   1   0]    t₂ can see t₀, t₁, t₂\nt₃ [1   1   1   1]    t₃ can see everything" },
            { type: "code", language: "python", code: "mask = torch.tril(torch.ones(seq_len, seq_len))\nscores = scores.masked_fill(mask == 0, float('-inf'))" },
            { type: "callout", variant: "warning", text: "Apply masked_fill before softmax! Setting weights to 0 after softmax breaks the probability distribution — the remaining weights no longer sum to 1." },
          ],
        },
        {
          title: "Multi-Head Attention",
          blocks: [
            { type: "paragraph", text: "A single attention head can only learn one \"attention pattern.\" Multi-head attention lets the model simultaneously focus on different features: one head might track syntactic relationships, another semantic similarity, another positional proximity." },
            { type: "code", language: "python", code: "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.n_heads = n_heads\n        self.d_k = d_model // n_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):  # x: (batch, seq_len, d_model)\n        B, T, C = x.shape\n        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        attn_out, _ = scaled_dot_product_attention(Q, K, V, causal_mask)\n        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)\n        return self.W_o(out)" },
            { type: "callout", variant: "quote", text: "Instructor's Note: Multi-head attention's power isn't just \"more heads\" — it's subspace decomposition. Research shows some heads focus on syntactic structure, some track coreference resolution, some capture local adjacency. It's like a team where each member watches a different aspect, then pools their observations." },
            { type: "callout", variant: "quote", text: "Think About It: If our model has 12 attention heads with dimension 64 each, d_model = 768. Now suppose we increase heads from 12 to 24 (each head dimension shrinks to 32), keeping total parameters the same — will performance be identical? Are more heads always better?" },
          ],
        },
      ],
      exercises: [
        { id: "qkv_projection", title: "TODO 1: QKV Projection", description: "Implement linear projections for Q, K, V and reshape into multi-head format.", labFile: "labs/phase2/attention.py", hints: ["nn.Linear(d_model, d_model) for projection", "view(B, T, n_heads, d_k).transpose(1, 2) to reshape"] },
        { id: "dot_product", title: "TODO 2: Scaled Dot-Product", description: "Compute QK^T / √d_k and return attention scores.", labFile: "labs/phase2/attention.py", hints: ["torch.matmul(Q, K.transpose(-2, -1))", "Divide by math.sqrt(d_k)"], pseudocode: "scores = Q @ K^T / sqrt(d_k)" },
        { id: "causal_mask", title: "TODO 3: Apply Causal Mask", description: "Build a lower-triangular causal mask and apply it to attention scores.", labFile: "labs/phase2/attention.py", hints: ["torch.tril(torch.ones(T, T))", "scores.masked_fill(mask == 0, -inf)"] },
        { id: "attention_weights", title: "TODO 4: Compute Attention Weights", description: "Apply softmax to masked scores, then multiply by V.", labFile: "labs/phase2/attention.py", hints: ["softmax(scores, dim=-1)", "Result @ V"] },
        { id: "concat_heads", title: "TODO 5: Concatenate Heads", description: "Concatenate multi-head outputs back to (B, T, d_model) shape and apply output projection.", labFile: "labs/phase2/attention.py", hints: ["transpose(1,2).contiguous().view(B, T, C)", "Pass through self.W_o"] },
      ],
      acceptanceCriteria: [
        "Attention weights in each row sum to 1 (softmax property)",
        "Causal mask ensures position i cannot attend to position j > i",
        "Multi-head output shape is (batch, seq_len, d_model)",
      ],
      references: [
        { title: "Attention Is All You Need", description: "Vaswani et al. 2017 — The original Transformer paper introducing multi-head attention", url: "https://arxiv.org/abs/1706.03762" },
        { title: "The Illustrated Transformer", description: "Jay Alammar's classic illustrated guide with animations explaining the attention mechanism", url: "https://jalammar.github.io/illustrated-transformer/" },
      ],
    },
  ],
};

// ═══════════════════════════════════════════════════════════════════
// Locale dispatch
// ═══════════════════════════════════════════════════════════════════

const phase1Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase1ContentZhTW,
  "zh-CN": phase1ContentZhCN,
  "en": phase1ContentEn,
};

const phase2Map: Record<Locale, PhaseContent> = {
  "zh-TW": phase2ContentZhTW,
  "zh-CN": phase2ContentZhCN,
  "en": phase2ContentEn,
};

export function getPhase1Content(locale: Locale): PhaseContent { return phase1Map[locale]; }
export function getPhase2Content(locale: Locale): PhaseContent { return phase2Map[locale]; }
