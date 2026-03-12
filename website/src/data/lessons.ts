import type { PhaseContent } from "./types";
import { phase0Content } from "./lessons_p0";
import { phase3Content } from "./lessons_p3_p4";
import { phase4Content } from "./lessons_p3_p4";
import { phase5Content } from "./lessons_p5_p6";
import { phase6Content } from "./lessons_p5_p6";
import { phase7Content } from "./lessons_p7_p8_p9";
import { phase8Content } from "./lessons_p7_p8_p9";
import { phase9Content } from "./lessons_p7_p8_p9";

const phase1Content: PhaseContent = {
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
            { type: "paragraph", text: "掃描整個序列，統計每一對相鄰 token 出現的次數。例如在 'aabaa' 中，pair (a,a) 出現 2 次，pair (a,b) 出現 1 次，pair (b,a) 出現 1 次。" },
            { type: "code", language: "python", code: "def get_pair_counts(token_ids: list[int]) -> dict[tuple[int,int], int]:\n    counts = {}\n    for i in range(len(token_ids) - 1):\n        pair = (token_ids[i], token_ids[i + 1])\n        counts[pair] = counts.get(pair, 0) + 1\n    return counts" },
            { type: "heading", level: 3, text: "Step 3: 合併最高頻 Pair" },
            { type: "paragraph", text: "找到出現次數最多的 pair，將它們合併成一個新 token，分配一個新的 token ID（從 256 開始）。然後在整個序列中執行這個替換。" },
            { type: "diagram", content: "訓練語料: 'aaabdaaabac'\n\n初始: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]\n\nMerge 1: (97,97)→256  →  [256, 97, 98, 100, 256, 97, 98, 97, 99]\nMerge 2: (256,97)→257 →  [257, 98, 100, 257, 98, 97, 99]\nMerge 3: (257,98)→258 →  [258, 100, 258, 97, 99]\n..." },
            { type: "callout", variant: "tip", text: "合併順序很重要！encode 時必須按照訓練時學到的順序依次應用 merge 規則。這就是為什麼我們需要保存一個有序的 merges 列表。" },
            { type: "callout", variant: "tip", text: "💡 講師心得：BPE 本質上是一個壓縮演算法，被巧妙地借用到了 NLP 領域。它在 1994 年被發明用於資料壓縮（Philip Gage），2016 年 Sennrich 等人發現它完美適用於子詞分割。這個跨領域的借用提醒我們：好的工程方案往往來自意想不到的地方。" },
          ],
        },
        {
          title: "Encoding & Decoding",
          blocks: [
            { type: "paragraph", text: "訓練完成後，我們得到了一個有序的 merge 規則列表。Encoding 就是將文本先轉為 bytes，再按順序套用每條 merge 規則。Decoding 則是將 token ID 序列轉回 bytes 再解碼為文字。" },
            { type: "heading", level: 3, text: "Encode" },
            { type: "code", language: "python", code: "def encode(self, text: str) -> list[int]:\n    tokens = list(text.encode('utf-8'))  # 轉為 byte list\n    for (p0, p1), new_id in self.merges.items():\n        i = 0\n        new_tokens = []\n        while i < len(tokens):\n            if i < len(tokens)-1 and tokens[i]==p0 and tokens[i+1]==p1:\n                new_tokens.append(new_id)\n                i += 2\n            else:\n                new_tokens.append(tokens[i])\n                i += 1\n        tokens = new_tokens\n    return tokens" },
            { type: "heading", level: 3, text: "Decode" },
            { type: "paragraph", text: "Decode 需要一個反向查找表：每個 token ID 對應它所代表的 bytes。對於基礎 token（0-255），它就是那個 byte。對於合併產生的 token，它是兩個子 token 的 bytes 串接。" },
            { type: "code", language: "python", code: "def decode(self, token_ids: list[int]) -> str:\n    byte_list = b''.join(self.vocab[t] for t in token_ids)\n    return byte_list.decode('utf-8', errors='replace')" },
            { type: "callout", variant: "warning", text: "encode(decode(ids)) 不一定等於 ids（因為可能有多種等價的 token 序列），但 decode(encode(text)) 必須等於原始 text。這是最重要的不變量。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們的 BPE tokenizer 是 byte-level 的，可以處理任何 UTF-8 文字。但如果要處理中文、日文、韓文這類 CJK 字元，BPE 的效率如何？一個中文字佔 3 個 UTF-8 bytes，這對詞彙表大小和序列長度有什麼影響？GPT-4 的 tokenizer 做了什麼特殊處理來優化多語言效率？" },
          ],
        },
      ],
      exercises: [
        {
          id: "build_initial_vocab", title: "TODO 1: build_initial_vocab",
          description: "建立初始詞彙表，包含 0-255 的所有 byte 值。每個 entry 是 token_id → bytes 的映射。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["使用 dict comprehension", "bytes([i]) 可以將整數轉為單一 byte"],
          pseudocode: "vocab = {}\nfor i in 0..255:\n  vocab[i] = bytes([i])\nreturn vocab",
        },
        {
          id: "get_pair_counts", title: "TODO 2: get_pair_counts",
          description: "統計 token 序列中所有相鄰 pair 的出現次數。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["遍歷 range(len(ids)-1)", "用 dict 儲存計數"],
          pseudocode: "counts = {}\nfor i in 0..len(ids)-2:\n  pair = (ids[i], ids[i+1])\n  counts[pair] += 1\nreturn counts",
        },
        {
          id: "merge_pair", title: "TODO 3: merge_pair",
          description: "在 token 序列中，將所有 (p0, p1) pair 替換為 new_id。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["用 while loop 而非 for loop，因為替換後長度會變", "找到 pair 就 append new_id 並 i+=2"],
          pseudocode: "result = []\ni = 0\nwhile i < len(ids):\n  if i < len-1 and ids[i]==p0 and ids[i+1]==p1:\n    result.append(new_id)\n    i += 2\n  else:\n    result.append(ids[i])\n    i += 1\nreturn result",
        },
        {
          id: "train", title: "TODO 4: train",
          description: "執行 BPE 訓練迴圈：反覆找最高頻 pair → 合併 → 更新詞彙表，直到達到目標大小。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["num_merges = target_vocab_size - 256", "每次 merge 後新 token ID = 256 + merge_count"],
          pseudocode: "ids = list(text.encode('utf-8'))\nfor i in 0..num_merges-1:\n  counts = get_pair_counts(ids)\n  best = max(counts, key=counts.get)\n  new_id = 256 + i\n  ids = merge_pair(ids, best, new_id)\n  merges[best] = new_id\n  vocab[new_id] = vocab[best[0]] + vocab[best[1]]",
        },
        {
          id: "encode", title: "TODO 5: encode",
          description: "將文本編碼為 token ID 序列，按訓練順序應用所有 merge 規則。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["先轉 UTF-8 bytes", "按 merges 的順序依次應用每條規則"],
        },
        {
          id: "decode", title: "TODO 6: decode",
          description: "將 token ID 序列解碼回文本字串。",
          labFile: "labs/phase1/bpe_tokenizer.py",
          hints: ["查表取得每個 token 的 bytes", "用 b''.join() 串接後 decode('utf-8')"],
        },
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
        { title: "HuggingFace Tokenizer Tutorial", description: "完整的 BPE tokenizer 教學，含視覺化步驟", url: "https://huggingface.co/learn/nlp-course/chapter6/5" },
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
            { type: "paragraph", text: "Tokenizer 搞定了，模型現在可以「看懂」文字了。但光有翻譯官還不夠——我們還需要一個「廚師」把原始食材加工成模型能消化的「便當」。這就是 DataLoader 的角色。但在建 DataLoader 之前，我們得先搞清楚一個更根本的問題：模型到底要學什麼？" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Next-token prediction 是整個 LLM 世界的基石，但初學者常低估它的威力。看似只是「猜下一個字」，但要做好這件事，模型必須學會語法、語義、邏輯推理、甚至世界知識。這就是為什麼 GPT-3 只用這個目標就能做翻譯、寫程式碼、回答問題——因為要精確預測下一個 token，你需要理解一切。" },
            { type: "paragraph", text: "GPT 類模型的訓練目標極為簡單：給定前面的 token 序列，預測下一個 token。這個看似簡單的目標，卻能讓模型學到語言的深層結構。" },
            { type: "paragraph", text: "具體來說，對於一個長度為 T 的序列 [t₀, t₁, ..., t_{T-1}]，我們構造：\n- 輸入 x = [t₀, t₁, ..., t_{T-2}]\n- 目標 y = [t₁, t₂, ..., t_{T-1}]\n\ny 就是 x 向右偏移一個位置。模型在每個位置都要預測下一個 token。" },
            { type: "diagram", content: "文本: \"Hello World\"\nToken IDs: [15496, 2159]\n\n如果 context_length = 4:\n  x = [15496, 2159, ...] (前 4 個 token)\n  y = [2159, ..., ...]   (偏移 1 的 4 個 token)\n\n位置 0: 看到 t₀ → 預測 t₁\n位置 1: 看到 t₀,t₁ → 預測 t₂\n位置 2: 看到 t₀,t₁,t₂ → 預測 t₃\n位置 3: 看到 t₀,t₁,t₂,t₃ → 預測 t₄" },
            { type: "callout", variant: "info", text: "由於 causal masking（後續 Phase 2 會實作），模型在位置 i 只能看到位置 0..i 的 token。所以每個訓練樣本實際上提供了 context_length 個訓練信號。" },
          ],
        },
        {
          title: "Sliding Window",
          blocks: [
            { type: "paragraph", text: "我們用固定大小的滑動視窗從已 tokenize 的文本中提取訓練樣本。視窗大小 = context_length + 1（因為需要多一個 token 做目標）。" },
            { type: "paragraph", text: "每個視窗產生一對 (x, y)：x 是前 context_length 個 token，y 是後 context_length 個 token（偏移 1）。我們可以用步長（stride）控制視窗間的重疊程度。stride = 1 產生最多樣本但高度重疊；stride = context_length 無重疊但樣本較少。" },
            { type: "diagram", content: "Token sequence: [a, b, c, d, e, f, g, h, i, j]\ncontext_length = 4, stride = 4\n\nWindow 1: [a, b, c, d, e]  →  x=[a,b,c,d]  y=[b,c,d,e]\nWindow 2: [e, f, g, h, i]  →  x=[e,f,g,h]  y=[f,g,h,i]\n..." },
            { type: "code", language: "python", code: "class TextDataset(Dataset):\n    def __init__(self, token_ids: list[int], context_length: int, stride: int):\n        self.x = []\n        self.y = []\n        for i in range(0, len(token_ids) - context_length, stride):\n            self.x.append(torch.tensor(token_ids[i:i+context_length]))\n            self.y.append(torch.tensor(token_ids[i+1:i+context_length+1]))\n\n    def __len__(self): return len(self.x)\n    def __getitem__(self, idx): return self.x[idx], self.y[idx]" },
          ],
        },
        {
          title: "Token & Position Embeddings",
          blocks: [
            { type: "paragraph", text: "Token ID 只是整數，模型需要的是連續向量。Token Embedding 是一個 V×D 的查找表（V = vocab_size, D = embedding_dim），每個 token ID 對應一個 D 維向量。" },
            { type: "paragraph", text: "但光有 token embedding 不夠——模型無法區分同一個詞在不同位置的意義。Position Embedding 是另一個 T×D 的查找表（T = max_context_length），每個位置對應一個 D 維向量。" },
            { type: "code", language: "python", code: "class Embeddings(nn.Module):\n    def __init__(self, vocab_size, embed_dim, context_length):\n        super().__init__()\n        self.token_emb = nn.Embedding(vocab_size, embed_dim)\n        self.pos_emb = nn.Embedding(context_length, embed_dim)\n\n    def forward(self, x):  # x: (batch, seq_len)\n        tok = self.token_emb(x)       # (batch, seq_len, embed_dim)\n        pos = self.pos_emb(torch.arange(x.size(1), device=x.device))\n        return tok + pos              # 逐元素相加" },
            { type: "callout", variant: "tip", text: "GPT-2 使用可學習的 position embedding（而非 Transformer 原始論文的 sinusoidal）。可學習的方式更靈活，讓模型自己決定位置的表示方式。" },
            { type: "callout", variant: "tip", text: "💡 講師心得：Token embedding 和 position embedding 的相加（而非拼接）是一個優雅的設計。兩個 embedding 共享同一個向量空間，相加後模型可以同時利用「這是什麼詞」和「這個詞在哪裡」的資訊。為什麼不用拼接？因為拼接會使維度翻倍，而實驗證明相加的效果一樣好，且更省參數。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：我們用「相加」而非「拼接」來合併 token embedding 和 position embedding。但這樣做不會互相干擾嗎？模型怎麼區分哪些資訊來自 token、哪些來自 position？（提示：想想高維空間中兩個隨機向量的夾角。）" },
          ],
        },
      ],
      exercises: [
        {
          id: "dataset_init", title: "TODO 1: TextDataset.__init__",
          description: "實作 sliding-window 邏輯，從 token 序列中提取 (x, y) 對。",
          labFile: "labs/phase1/dataloader.py",
          hints: ["用 range(0, len(ids) - context_length, stride) 作為起始索引", "x 和 y 之間偏移 1 個位置"],
          pseudocode: "for i in range(0, len(ids) - ctx_len, stride):\n  self.x.append(tensor(ids[i : i+ctx_len]))\n  self.y.append(tensor(ids[i+1 : i+ctx_len+1]))",
        },
        {
          id: "dataset_len", title: "TODO 2: TextDataset.__len__",
          description: "返回資料集中的樣本數量。",
          labFile: "labs/phase1/dataloader.py",
          hints: ["直接返回 self.x 的長度"],
        },
        {
          id: "dataset_getitem", title: "TODO 3: TextDataset.__getitem__",
          description: "返回第 idx 個 (x, y) 訓練對。",
          labFile: "labs/phase1/dataloader.py",
          hints: ["return self.x[idx], self.y[idx]"],
        },
        {
          id: "create_dataloaders", title: "TODO 4: create_dataloaders",
          description: "將文本分割為 train/val，建立 DataLoader。",
          labFile: "labs/phase1/dataloader.py",
          hints: ["先 tokenize 整個文本", "用前 90% 做 train，後 10% 做 val", "用 torch.utils.data.DataLoader 包裝"],
          pseudocode: "ids = tokenizer.encode(text)\nsplit = int(0.9 * len(ids))\ntrain_ds = TextDataset(ids[:split], ctx_len, stride)\nval_ds = TextDataset(ids[split:], ctx_len, stride)\nreturn DataLoader(train_ds, batch_size, shuffle=True), DataLoader(val_ds, batch_size)",
        },
      ],
      acceptanceCriteria: [
        "DataLoader 產生正確 shape 的 (x, y) tensor",
        "y 是 x 向右偏移 1 位的結果",
        "train/val split 比例正確，無資料洩漏",
        "batch_size 和 context_length 可自訂設定",
      ],
      references: [
        { title: "Language Models are Unsupervised Multitask Learners", description: "GPT-2 論文，描述了 next-token prediction 的訓練方式", url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" },
        { title: "Efficient Estimation of Word Representations in Vector Space", description: "Word2Vec 論文，embedding 查找表的概念源頭", url: "https://arxiv.org/abs/1301.3781" },
      ],
    },
  ],
};

const phase2Content: PhaseContent = {
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
            { type: "paragraph", text: "到目前為止，每個 token 都是「孤島」——它只知道自己是什麼（embedding），在哪裡（position），但完全不知道周圍有什麼其他 token。這就像一群人站在一個房間裡，每個人都戴著眼罩和耳塞。要理解語言，token 們必須能互相「交流」。Attention 就是拿掉眼罩和耳塞的魔法。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Attention 是 Transformer 的靈魂。在 RNN 時代，遠距離的詞要一步一步傳遞資訊（像傳話遊戲），資訊會衰減。Attention 讓每個詞都能「直接」跟序列中任何其他詞對話——這就像從排隊傳話升級成視訊會議。這一個改變，奠定了整個 LLM 革命的基礎。" },
            { type: "paragraph", text: "理解 Attention 最好的方式是圖書館比喻。想像你走進一間巨大的圖書館，帶著一個問題（Query）。書架上每本書都有索引卡（Key），書本身是內容（Value）。" },
            { type: "paragraph", text: "你將你的問題與每本書的索引卡做比較（dot product），找到最相關的書（高分），然後根據相關度加權混合這些書的內容，得到你的答案。" },
            { type: "list", ordered: true, items: [
              "Query (Q): 你的問題——「我在找什麼？」",
              "Key (K): 書的索引——「這本書是關於什麼的？」",
              "Value (V): 書的內容——「這本書說了什麼？」",
              "Attention Score: Q 和 K 的相似度——「這本書和我的問題有多相關？」",
              "Attention Weight: softmax 後的分數——「我該花多少注意力在這本書上？」",
            ]},
            { type: "callout", variant: "quote", text: "在 self-attention 中，Q、K、V 都來自同一個序列的不同線性投影。每個 token 同時扮演「提問者」和「被查詢者」的角色。" },
          ],
        },
        {
          title: "Scaled Dot-Product Attention",
          blocks: [
            { type: "paragraph", text: "Attention 的數學定義非常簡潔：" },
            { type: "diagram", content: "Attention(Q, K, V) = softmax(QK^T / √d_k) · V\n\n其中：\n  Q: (seq_len, d_k) — Query 矩陣\n  K: (seq_len, d_k) — Key 矩陣  \n  V: (seq_len, d_v) — Value 矩陣\n  d_k: Key 的維度" },
            { type: "heading", level: 3, text: "為什麼要除以 √d_k？" },
            { type: "paragraph", text: "當 d_k 很大時，QK^T 的值也會很大（因為是 d_k 個乘積的總和）。大的值會讓 softmax 的輸出接近 one-hot（梯度幾乎為零），導致訓練困難。除以 √d_k 將方差標準化回 1。" },
            { type: "code", language: "python", code: "def scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    weights = torch.softmax(scores, dim=-1)\n    return torch.matmul(weights, V), weights" },
            { type: "callout", variant: "info", text: "注意 softmax 是在最後一個維度（key 維度）上做的。這確保了對每個 query，所有 key 的權重和為 1。" },
            { type: "callout", variant: "tip", text: "💡 講師心得：很多教材只告訴你 √d_k 是為了「穩定梯度」，但更深的原因是：假設 Q 和 K 的每個元素都是均值 0、方差 1 的隨機變數，那麼它們的點積的方差是 d_k（因為是 d_k 個獨立乘積之和）。除以 √d_k 把方差拉回到 1，讓 softmax 的輸入保持在合理範圍。這不是經驗性的 trick，而是有嚴格數學依據的設計。" },
          ],
        },
        {
          title: "Causal Masking",
          blocks: [
            { type: "paragraph", text: "在 GPT 這樣的自回歸模型中，生成 token t_i 時只能看到 t_0 到 t_{i-1}。如果模型在訓練時能看到未來的 token，它就會「作弊」而不是真正學習語言模式。" },
            { type: "paragraph", text: "Causal mask 是一個下三角矩陣：位置 (i, j) 為 1 當 j ≤ i，否則為 0。我們在 softmax 之前將 mask=0 的位置填入 -∞，softmax 後這些位置的權重就變成 0。" },
            { type: "diagram", content: "Causal Mask (seq_len=4):\n\n  t₀  t₁  t₂  t₃\nt₀ [1   0   0   0]    t₀ 只能看自己\nt₁ [1   1   0   0]    t₁ 可以看 t₀, t₁\nt₂ [1   1   1   0]    t₂ 可以看 t₀, t₁, t₂\nt₃ [1   1   1   1]    t₃ 可以看所有\n\n在 scores 中 mask=0 的位置填入 -∞ → softmax 後變成 0" },
            { type: "code", language: "python", code: "# 建立 causal mask\nmask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩陣\n# mask shape: (seq_len, seq_len)\n\n# 應用到 attention scores\nscores = scores.masked_fill(mask == 0, float('-inf'))" },
            { type: "callout", variant: "warning", text: "masked_fill 要在 softmax 之前做！如果在 softmax 之後把權重設為 0，剩餘權重的和就不再是 1，打破了概率分布的性質。" },
          ],
        },
        {
          title: "Multi-Head Attention",
          blocks: [
            { type: "paragraph", text: "單一的 attention head 只能學到一種「注意力模式」。Multi-head attention 讓模型同時關注不同的特徵：有的 head 可能關注語法關係，有的關注語義相似性，有的關注位置相鄰性。" },
            { type: "paragraph", text: "做法是將 d_model 維度的 Q、K、V 分割成 h 個 head，每個 head 處理 d_k = d_model / h 維的子空間。各 head 獨立計算 attention，最後拼接輸出。" },
            { type: "code", language: "python", code: "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.n_heads = n_heads\n        self.d_k = d_model // n_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x):  # x: (batch, seq_len, d_model)\n        B, T, C = x.shape\n        # 投影 + reshape: (B, T, C) → (B, T, n_heads, d_k) → (B, n_heads, T, d_k)\n        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)\n        # Attention per head\n        attn_out, _ = scaled_dot_product_attention(Q, K, V, causal_mask)\n        # 拼接所有 heads: (B, n_heads, T, d_k) → (B, T, C)\n        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)\n        return self.W_o(out)" },
            { type: "callout", variant: "tip", text: "transpose(1,2) 將 head 維度移到 batch 維度之後，讓我們可以用一次矩陣乘法同時計算所有 head 的 attention。這是一個關鍵的效率技巧。" },
            { type: "callout", variant: "quote", text: "💡 講師心得：Multi-head attention 的精髓不只是「多個 head」，而是「子空間分解」。每個 head 在 d_model/h 維的子空間中獨立運算，學到不同的注意力模式。研究發現，有的 head 專注語法結構（主語動詞關係），有的追蹤指代消解（代詞指向誰），有的捕捉局部相鄰性。這就像一個團隊，每個人負責觀察不同的面向，最後匯總成完整的理解。" },
            { type: "callout", variant: "quote", text: "🤔 思考題：如果我們的模型有 12 個 attention head，每個 head 維度是 64，那 d_model = 768。現在假設我們把 head 數量從 12 增加到 24（每個 head 維度降為 32），模型的總參數量不變——但效果會一樣嗎？更多的 head 一定更好嗎？為什麼？" },
          ],
        },
      ],
      exercises: [
        {
          id: "qkv_projection", title: "TODO 1: QKV Projection",
          description: "實作 Q、K、V 的線性投影，並 reshape 為多頭格式。",
          labFile: "labs/phase2/attention.py",
          hints: ["nn.Linear(d_model, d_model) 做投影", "view(B, T, n_heads, d_k).transpose(1, 2) 做 reshape"],
        },
        {
          id: "dot_product", title: "TODO 2: Scaled Dot-Product",
          description: "計算 QK^T / √d_k 並返回 attention scores。",
          labFile: "labs/phase2/attention.py",
          hints: ["torch.matmul(Q, K.transpose(-2, -1))", "除以 math.sqrt(d_k)"],
          pseudocode: "scores = Q @ K^T / sqrt(d_k)",
        },
        {
          id: "causal_mask", title: "TODO 3: Apply Causal Mask",
          description: "建立下三角 causal mask 並應用到 attention scores。",
          labFile: "labs/phase2/attention.py",
          hints: ["torch.tril(torch.ones(T, T))", "scores.masked_fill(mask == 0, -inf)"],
          pseudocode: "mask = tril(ones(T, T))\nscores = scores.masked_fill(mask == 0, -inf)",
        },
        {
          id: "attention_weights", title: "TODO 4: Compute Attention Weights",
          description: "對 masked scores 做 softmax，然後與 V 相乘。",
          labFile: "labs/phase2/attention.py",
          hints: ["softmax(scores, dim=-1)", "結果 @ V"],
          pseudocode: "weights = softmax(scores, dim=-1)\noutput = weights @ V",
        },
        {
          id: "concat_heads", title: "TODO 5: Concatenate Heads",
          description: "將多頭輸出拼接回 (B, T, d_model) 形狀，通過輸出投影。",
          labFile: "labs/phase2/attention.py",
          hints: ["transpose(1,2).contiguous().view(B, T, C)", "最後過 self.W_o"],
          pseudocode: "out = attn_out.transpose(1,2).reshape(B, T, d_model)\nreturn W_o(out)",
        },
      ],
      acceptanceCriteria: [
        "attention weights 每行和為 1（softmax 性質）",
        "causal mask 確保位置 i 不能 attend 到位置 j > i",
        "多頭輸出 shape 為 (batch, seq_len, d_model)",
        "Q、K、V 投影正確分割為多頭",
      ],
      references: [
        { title: "Attention Is All You Need", description: "Vaswani et al. 2017 — Transformer 原始論文，提出 multi-head attention", url: "https://arxiv.org/abs/1706.03762" },
        { title: "The Illustrated Transformer", description: "Jay Alammar 的經典圖解文章，用動畫解釋 attention 機制", url: "https://jalammar.github.io/illustrated-transformer/" },
        { title: "FlashAttention", description: "Dao et al. 2022 — 高效 attention 實作，理解記憶體存取模式", url: "https://arxiv.org/abs/2205.14135" },
      ],
    },
  ],
};

export const allPhaseContent: PhaseContent[] = [
  phase0Content, phase1Content, phase2Content, phase3Content, phase4Content,
  phase5Content, phase6Content, phase7Content, phase8Content, phase9Content,
];
