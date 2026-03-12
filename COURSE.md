# LLM From Scratch — 從零構建大型語言模型

> 用 PyTorch 手把手實現完整的 GPT-like 大型語言模型，從 Tokenizer 到 Mixture of Experts。

## 課程總覽

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM From Scratch                         │
│              從零構建大型語言模型完整課程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: 文本與數據     ──── Tokenizer + DataLoader        │
│       ↓                                                     │
│  Phase 2: 注意力機制     ──── Self-Attention → Multi-Head    │
│       ↓                                                     │
│  Phase 3: Transformer    ──── GPT 模型組裝                   │
│       ↓                                                     │
│  Phase 4: 預訓練         ──── Training Loop + Loss           │
│       ↓                                                     │
│  Phase 5: 文本生成       ──── Inference + Sampling           │
│       ↓                                                     │
│  Phase 6: 微調分類       ──── Transfer Learning              │
│       ↓                                                     │
│  Phase 7: LoRA           ──── Parameter-Efficient Tuning     │
│       ↓                                                     │
│  Phase 8: 指令微調       ──── SFT + DPO                      │
│       ↓                                                     │
│  Phase 9: MoE            ──── Mixture of Experts             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**技術棧**：Python + PyTorch + tiktoken
**時長**：6-8 週
**先決條件**：Python 基礎、線性代數基本概念
**運行環境**：筆電即可，不需要 GPU

---

## Phase 1: 文本與數據

### 概覽

> **核心問題**：LLM 如何「看到」文字？

電腦不懂文字，只懂數字。Phase 1 要解決的根本問題是：如何把人類的自然語言轉換成模型能處理的數值表示？

這個看似簡單的問題其實包含兩個關鍵決策：
1. **切分粒度**：按字符？按單詞？按子詞？
2. **上下文窗口**：模型一次能「看到」多少文本？

```
                    Phase 1 架構
┌──────────────────────────────────────────────┐
│                                              │
│   "Hello world"                              │
│        │                                     │
│        ▼                                     │
│   ┌──────────┐                               │
│   │ Tokenizer│  BPE: "Hel" "lo" " world"    │
│   └────┬─────┘                               │
│        │  [15496, 995]                       │
│        ▼                                     │
│   ┌──────────┐                               │
│   │ Embedding│  ID → 向量                    │
│   └────┬─────┘                               │
│        │  [[0.12, -0.34, ...], ...]          │
│        ▼                                     │
│   ┌──────────┐                               │
│   │DataLoader│  滑動窗口 + 批次              │
│   └──────────┘                               │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：Tokenizer 是 LLM 的「眼睛」— 它決定了模型看到世界的解析度。

---

### 概念 1.1：從字符到 Token

#### 為什麼不直接用字符？

最直覺的方式是把每個字符當作一個 token：

```python
text = "Hello world"
chars = list(text)  # ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
```

問題：序列太長。一篇 1000 字的文章就有 ~5000 個字符 token，注意力機制的計算複雜度是 O(n²)，這會非常慢。

#### 為什麼不直接用單詞？

```python
text = "Hello world"
words = text.split()  # ['Hello', 'world']
```

問題：詞彙表太大。英語有 ~170,000 個常用單詞，加上變形（running, ran, runs）、專有名詞、技術術語，詞彙表會膨脹到數百萬。而且遇到沒見過的新詞（OOV）就完全無法處理。

#### 子詞切分：最佳平衡點

| 方法 | 詞彙表大小 | 序列長度 | OOV 問題 |
|------|-----------|---------|---------|
| 字符 | ~256 | 很長 | 無 |
| 單詞 | ~170K+ | 最短 | 嚴重 |
| **子詞 (BPE)** | **~50K** | **適中** | **極少** |

BPE (Byte Pair Encoding) 通過統計方式自動發現最優的子詞切分。

---

### 概念 1.2：Byte Pair Encoding (BPE)

#### 演算法

BPE 的核心思想驚人地簡單：**反覆合併最頻繁的相鄰 token 對**。

```
初始詞彙表（所有單個字符）：
  {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', ...}

訓練文本中的詞頻：
  "low"    : 5次
  "lower"  : 2次
  "newest" : 6次
  "widest" : 3次

Step 1: 最頻繁的相鄰對 = ('e', 's') 出現 9次
  合併 → 新 token 'es'

Step 2: 最頻繁的相鄰對 = ('es', 't') 出現 9次
  合併 → 新 token 'est'

Step 3: 最頻繁的相鄰對 = ('l', 'o') 出現 7次
  合併 → 新 token 'lo'

... 持續直到達到目標詞彙表大小
```

#### 數據結構

```python
class Tokenizer:
    def __init__(self):
        self.vocab = {}           # token_str -> token_id
        self.merges = []          # [(pair1, pair2, merged), ...] 按順序
        self.inverse_vocab = {}   # token_id -> token_str
```

#### 編碼過程

```
輸入: "lowest"
初始 tokens: ['l', 'o', 'w', 'e', 's', 't']

套用 merge ('e', 's') → 'es':
  ['l', 'o', 'w', 'es', 't']

套用 merge ('es', 't') → 'est':
  ['l', 'o', 'w', 'est']

套用 merge ('l', 'o') → 'lo':
  ['lo', 'w', 'est']

查找 IDs: [12, 8, 45]
```

---

### 概念 1.3：Token Embedding

Token ID 只是一個整數索引。模型需要的是連續的向量表示：

```
Token ID: 15496 ("Hello")
    │
    ▼
┌──────────────────────────────────────────┐
│         Embedding Matrix                  │
│      (vocab_size × d_model)               │
│                                           │
│  ID 0:    [0.12, -0.34, 0.56, ...]       │
│  ID 1:    [0.78, 0.23, -0.91, ...]       │
│  ...                                      │
│  ID 15496:[0.45, -0.12, 0.89, ...]  ◄── │
│  ...                                      │
└──────────────────────────────────────────┘
    │
    ▼
向量: [0.45, -0.12, 0.89, ...]  (維度 = d_model)
```

Embedding 其實就是一個查找表 — `nn.Embedding(vocab_size, d_model)` 本質上等價於一個 `(vocab_size, d_model)` 的矩陣，用 token ID 作為行索引。

---

### 概念 1.4：位置編碼 (Positional Encoding)

Transformer 本身不知道 token 的順序（它看到的只是一個集合）。我們需要注入位置信息：

```
Token Embedding:   [0.45, -0.12, 0.89, ...]   "Hello" 在什麼位置？不知道！
                        +
Position Embedding:[0.01, 0.02, -0.01, ...]    位置 0 的向量
                        =
最終 Embedding:    [0.46, -0.10, 0.88, ...]    "Hello" 在位置 0
```

GPT 使用可學習的位置嵌入：`nn.Embedding(block_size, d_model)`

---

### 概念 1.5：滑動窗口 DataLoader

LLM 的訓練目標是**預測下一個 token**。DataLoader 負責把長文本切成固定長度的窗口：

```
原始 token 序列: [t0, t1, t2, t3, t4, t5, t6, t7, t8, ...]
block_size = 4

窗口 0:  x = [t0, t1, t2, t3]    y = [t1, t2, t3, t4]
窗口 1:  x = [t1, t2, t3, t4]    y = [t2, t3, t4, t5]
窗口 2:  x = [t2, t3, t4, t5]    y = [t3, t4, t5, t6]
                ↑                       ↑
            模型輸入              預測目標（偏移 1 位）
```

注意 x 和 y 的關係：**y 就是 x 向右偏移一個位置**。這讓模型學會：給定前面的 tokens，預測下一個。

---

### Lab 1: 構建 BPE Tokenizer 和 DataLoader

**難度**: ⭐⭐ (中等)
**預計時間**: 2-3 小時
**文件**: `labs/phase1_text_and_data/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | tokenizer.py | `_get_pair_counts()` | ⭐ | 統計相鄰 token 對的頻率 |
| 2 | tokenizer.py | `_merge_pair()` | ⭐ | 合併最頻繁的 token 對 |
| 3 | tokenizer.py | `train()` | ⭐⭐ | 完整 BPE 訓練迴圈 |
| 4 | tokenizer.py | `encode()` | ⭐⭐ | 用已學習的 merges 編碼文本 |
| 5 | tokenizer.py | `decode()` | ⭐ | Token IDs → 文本 |
| 6 | dataloader.py | `TextDataset` | ⭐⭐ | 滑動窗口數據集 |
| 7 | dataloader.py | `create_dataloaders()` | ⭐ | 訓練/驗證 DataLoader |

#### TODO 1: `_get_pair_counts` 偽代碼

```
函數 _get_pair_counts(token_ids):
    counts = 空字典
    對於 i 從 0 到 len(token_ids) - 2:
        pair = (token_ids[i], token_ids[i+1])
        counts[pair] += 1
    返回 counts
```

#### TODO 3: `train` 偽代碼

```
函數 train(text, vocab_size):
    初始化 vocab = 所有單個字節 (0-255)
    token_ids = 把 text 轉成字節列表

    當 len(vocab) < vocab_size:
        pair_counts = _get_pair_counts(token_ids)
        best_pair = 頻率最高的 pair
        new_id = len(vocab)
        token_ids = _merge_pair(token_ids, best_pair, new_id)
        記錄 merge: best_pair → new_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
```

#### 常見錯誤

1. **忘記處理 UTF-8 多字節字符**：中文字符佔 3 個字節，要用 `text.encode('utf-8')` 而不是 `list(text)`
2. **Merge 順序很重要**：encode 時必須按照訓練時的 merge 順序依次套用
3. **DataLoader 的 x/y 偏移**：y = tokens[i+1:i+block_size+1]，不是 tokens[i:i+block_size]

---

### 整合測試

完成 Lab 後，你應該能執行：

```bash
cd labs/phase1_text_and_data
python grade.py
```

預期輸出：
```
=== Phase 1 Grading ===
[PASS] test_pair_counting .................... 10/10
[PASS] test_merge_pair ....................... 10/10
[PASS] test_bpe_training .................... 15/15
[PASS] test_encode_decode_roundtrip ......... 15/15
[PASS] test_dataset_shapes .................. 10/10
[PASS] test_xy_offset ....................... 10/10
[PASS] test_train_val_split ................. 10/10
[PASS] test_batching ........................ 10/10

Total: 90/90 (100%)
```

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| BPE Tokenizer | 子詞切分的統計方法 | 決定模型的「解析度」，影響所有下游任務 |
| Token Embedding | 離散 ID → 連續向量 | 模型只能處理向量，這是橋樑 |
| Positional Encoding | 注入序列位置信息 | 沒有它，Transformer 就是一個集合模型 |
| Sliding Window | 長文本 → 固定窗口 | 訓練效率 + 內存管理的關鍵 |

```
Phase 1 完成的模組：

┌──────────┐     ┌───────────┐     ┌──────────┐
│ 原始文本  │ ──→ │ Tokenizer │ ──→ │DataLoader│ ──→ 準備好的批次
└──────────┘     └───────────┘     └──────────┘

下一步：Phase 2 將構建注意力機制 — LLM 的核心引擎。
```

### 參考資料

**必讀**：
- [Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [OpenAI tiktoken 源碼](https://github.com/openai/tiktoken)

**深入閱讀**：
- [Hugging Face Tokenizer 教程](https://huggingface.co/learn/nlp-course/chapter6/1)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)

**擴展思考**：
- 如果詞彙表太大會怎樣？太小又會怎樣？（提示：想想 embedding 矩陣的大小和序列長度的權衡）
- BPE 對中文和英文的效果有什麼不同？（提示：想想字節表示的差異）

---

## Phase 2: 注意力機制

### 概覽

> **核心問題**：模型如何知道「哪些詞與哪些詞相關」？

在 Phase 1 中，每個 token 都是獨立的向量。但語言是有上下文的 — 「bank」在「river bank」和「bank account」中意義完全不同。注意力機制讓每個 token 能「看到」其他 token，並根據相關性動態調整自己的表示。

```
                    Phase 2 架構
┌──────────────────────────────────────────────┐
│                                              │
│   輸入序列: [x₁, x₂, x₃, x₄]               │
│                    │                         │
│                    ▼                         │
│        ┌─────────────────────┐               │
│        │ Q = W_q · x         │               │
│        │ K = W_k · x         │ 線性投影      │
│        │ V = W_v · x         │               │
│        └─────────┬───────────┘               │
│                  │                           │
│                  ▼                           │
│     ┌──────────────────────┐                 │
│     │ Attention Weights     │                │
│     │                       │                │
│     │ softmax(Q·K^T/√d_k)  │                │
│     │                       │                │
│     │  x₁   x₂   x₃   x₄  │                │
│     │  ┌────┬────┬────┬───┐ │                │
│     │ x₁│0.5 │0.3 │0.1 │0.1│ │               │
│     │ x₂│    │0.6 │0.2 │0.2│ │  (因果遮罩)   │
│     │ x₃│    │    │0.7 │0.3│ │               │
│     │ x₄│    │    │    │1.0│ │               │
│     │  └────┴────┴────┴───┘ │                │
│     └──────────┬───────────┘                 │
│                │                             │
│                ▼                             │
│     Output = Weights · V                     │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：注意力 = 「軟查找」。Query 是問題，Key 是索引，Value 是內容。注意力權重告訴模型「從哪裡取多少信息」。

---

### 概念 2.1：點積注意力

#### 直覺

想像你在一個圖書館：
- **Query (Q)**: 你想找什麼？「機器學習的入門書」
- **Key (K)**: 每本書的標籤/索引
- **Value (V)**: 書的實際內容

注意力做的事：用 Q 和每個 K 做比較（點積），得到相關性分數，然後按相關性加權取 V。

#### 數學

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V

其中:
  Q: (seq_len, d_k) — 查詢矩陣
  K: (seq_len, d_k) — 鍵矩陣
  V: (seq_len, d_v) — 值矩陣
  d_k: key 的維度（用於縮放）
```

#### 為什麼要除以 √d_k？

```
d_k = 64 時：
  點積的期望值 ∝ d_k
  點積的方差 ∝ d_k

不縮放：點積值可能很大 → softmax 趨近 one-hot → 梯度消失
縮放後：點積值穩定在合理範圍 → softmax 更平滑 → 學習更穩定
```

---

### 概念 2.2：因果遮罩 (Causal Mask)

GPT 是**自回歸**模型 — 生成第 t 個 token 時，只能看到前 t-1 個 token，不能偷看未來。

```
                未遮罩                    因果遮罩（下三角）
           t₁   t₂   t₃   t₄         t₁   t₂   t₃   t₄
    t₁  [ 0.5  0.3  0.1  0.1 ]  t₁  [ 0.5  -∞   -∞   -∞  ]
    t₂  [ 0.2  0.6  0.1  0.1 ]  t₂  [ 0.2  0.6  -∞   -∞  ]
    t₃  [ 0.1  0.2  0.5  0.2 ]  t₃  [ 0.1  0.2  0.5  -∞  ]
    t₄  [ 0.1  0.1  0.2  0.6 ]  t₄  [ 0.1  0.1  0.2  0.6 ]

    遮罩位置設為 -∞ → softmax 後變成 0 → 不會注意到未來 token
```

```python
# 實現：
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
```

---

### 概念 2.3：多頭注意力 (Multi-Head Attention)

一個注意力頭只能學到一種「關注模式」。多頭注意力讓模型同時學習多種模式：

```
┌──────────── Multi-Head Attention ────────────┐
│                                              │
│  Head 1: 關注語法關係  ("The cat" → "sat")   │
│  Head 2: 關注指代關係  ("it" → "the cat")    │
│  Head 3: 關注位置鄰近  (相鄰的詞)            │
│  Head 4: 關注語義關係  ("cat" → "animal")    │
│                                              │
│  每個 head:                                  │
│    d_model = 256, n_heads = 4               │
│    head_dim = d_model / n_heads = 64        │
│                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │Head 1│ │Head 2│ │Head 3│ │Head 4│       │
│  │(64d) │ │(64d) │ │(64d) │ │(64d) │       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘       │
│     └────────┴────────┴────────┘             │
│                  │ Concatenate               │
│                  ▼                            │
│           (256d = 4 × 64d)                   │
│                  │                            │
│           ┌──────┴──────┐                    │
│           │  W_o 投影    │                    │
│           │ (256 → 256)  │                    │
│           └─────────────┘                    │
│                                              │
└──────────────────────────────────────────────┘
```

---

### Lab 2: 實現 Multi-Head Causal Attention

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 2-3 小時
**文件**: `labs/phase2_attention/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | attention.py | `scaled_dot_product_attention()` | ⭐⭐ | 基礎注意力計算 |
| 2 | attention.py | `CausalSelfAttention.__init__()` | ⭐ | Q/K/V 投影層 |
| 3 | attention.py | `CausalSelfAttention.forward()` | ⭐⭐ | 單頭因果注意力 |
| 4 | attention.py | `MultiHeadAttention.__init__()` | ⭐⭐ | 多頭初始化 |
| 5 | attention.py | `MultiHeadAttention.forward()` | ⭐⭐⭐ | reshape + 並行計算 |

#### TODO 1: `scaled_dot_product_attention` 偽代碼

```
函數 scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q 的最後一個維度
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    如果 mask 不為 None:
        scores = scores.masked_fill(mask == 0, -inf)
    weights = softmax(scores, dim=-1)
    output = weights @ V
    返回 output
```

#### TODO 5: `MultiHeadAttention.forward` 偽代碼

```
函數 forward(x):
    batch, seq_len, d_model = x.shape

    # 投影
    Q = self.W_q(x)  # (batch, seq_len, d_model)
    K = self.W_k(x)
    V = self.W_v(x)

    # 分頭: (batch, seq_len, d_model) → (batch, n_heads, seq_len, head_dim)
    Q = Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

    # 因果遮罩
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # 注意力
    attn_output = scaled_dot_product_attention(Q, K, V, mask)

    # 合併頭: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, d_model)
    output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

    返回 self.W_o(output)
```

#### 常見錯誤

1. **忘記縮放**：不除以 √d_k 會導致 softmax 飽和，梯度消失
2. **遮罩方向錯誤**：`torch.tril` 是下三角（允許看過去），`torch.triu` 是上三角（標記未來位置）
3. **reshape 順序**：必須先 view 再 transpose，不能直接 reshape

---

### 整合測試

```bash
cd labs/phase2_attention
python grade.py
```

預期輸出：
```
=== Phase 2 Grading ===
[PASS] test_attention_output_shape ........... 10/10
[PASS] test_causal_mask ...................... 15/15
[PASS] test_attention_weights_sum_to_one ..... 10/10
[PASS] test_multi_head_dimensions ........... 15/15
[PASS] test_multi_head_output_shape ......... 10/10

Total: 60/60 (100%)
```

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| 點積注意力 | QKV 的查找語義 | 注意力機制的數學基礎 |
| 縮放因子 | 除以 √d_k 穩定梯度 | 沒有它，訓練會崩潰 |
| 因果遮罩 | 防止偷看未來 | 自回歸生成的前提條件 |
| 多頭注意力 | 並行學習多種模式 | 大幅增加模型表達能力 |

```
Phase 2 完成的模組：

┌──────────┐     ┌───────────────┐     ┌──────────────┐
│ Tokenizer│ ──→ │ DataLoader    │ ──→ │ Multi-Head   │
│ (Phase 1)│     │ (Phase 1)     │     │ Attention    │
└──────────┘     └───────────────┘     │ (Phase 2) ✓  │
                                       └──────────────┘

下一步：Phase 3 將把注意力模組裝進完整的 Transformer 架構。
```

### 參考資料

**必讀**：
- [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

**深入閱讀**：
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

**擴展思考**：
- 如果沒有因果遮罩，這個模型能用來做什麼？（提示：想想 BERT 的做法）
- 注意力的 O(n²) 複雜度對長文本有什麼影響？有哪些解決方案？（提示：sparse attention, linear attention）

---

## Phase 3: Transformer 架構

### 概覽

> **核心問題**：如何把注意力機制組裝成完整的語言模型？

Phase 2 的多頭注意力只是一個組件。要構建完整的 GPT，我們還需要：前饋網路（FFN）、層歸一化（LayerNorm）、殘差連接（Residual）。這些看似簡單的組件，每一個都有深刻的設計考量。

```
                    Phase 3 架構
┌──────────────────────────────────────────────┐
│            GPT Model 完整結構                 │
│                                              │
│  輸入 token IDs                              │
│       │                                      │
│  ┌────▼──────────────────────────────────┐   │
│  │ Token Embedding + Position Embedding  │   │
│  └────┬──────────────────────────────────┘   │
│       │                                      │
│  ┌────▼──────────────────────────────────┐   │
│  │         Transformer Block × N          │   │
│  │  ┌─────────────────────────────────┐  │   │
│  │  │ LayerNorm                       │  │   │
│  │  │    ↓                            │  │   │
│  │  │ Multi-Head Attention            │  │   │
│  │  │    ↓  (+殘差連接)               │  │   │
│  │  │ LayerNorm                       │  │   │
│  │  │    ↓                            │  │   │
│  │  │ Feed-Forward (GELU)             │  │   │
│  │  │    ↓  (+殘差連接)               │  │   │
│  │  └─────────────────────────────────┘  │   │
│  └────┬──────────────────────────────────┘   │
│       │                                      │
│  ┌────▼──────────────┐                       │
│  │ Final LayerNorm   │                       │
│  └────┬──────────────┘                       │
│       │                                      │
│  ┌────▼──────────────┐                       │
│  │ Linear (lm_head)  │  → logits (vocab_size)│
│  └───────────────────┘                       │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：Transformer Block = 注意力（混合信息）+ FFN（處理信息）+ 殘差（保持梯度流）+ LayerNorm（穩定訓練）。

---

### 概念 3.1：Layer Normalization

#### 為什麼需要歸一化？

深度網路中，每層的輸出分布會隨訓練不斷變化（internal covariate shift）。LayerNorm 把每個樣本的特徵歸一化到零均值、單位方差：

```
輸入 x = [2.0, 4.0, 6.0, 8.0]

mean = (2+4+6+8)/4 = 5.0
var  = ((2-5)²+(4-5)²+(6-5)²+(8-5)²)/4 = 5.0
std  = √5 ≈ 2.236

歸一化: [(2-5)/2.236, (4-5)/2.236, (6-5)/2.236, (8-5)/2.236]
       = [-1.34, -0.45, 0.45, 1.34]

仿射變換: y = γ · x_norm + β  （γ, β 是可學習參數）
```

#### LayerNorm vs BatchNorm

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 歸一化維度 | 跨批次的同一特徵 | 同一樣本的所有特徵 |
| 依賴批次大小 | 是 | 否 |
| 適合序列模型 | 否（可變長度） | 是 |
| 推理一致性 | 需要 running stats | 不需要 |

---

### 概念 3.2：GELU 激活函數

GPT 使用 GELU 而不是 ReLU：

```
ReLU(x)  = max(0, x)           ← 在 0 處不可微，負值完全截斷
GELU(x)  = x · Φ(x)           ← 平滑版本，保留小的負值信號
         ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

         GELU vs ReLU
    1.0 │         ╱ ReLU
        │        ╱
        │       ╱
    0.0 │──────╱────── ─ ─ ─
        │    ╱╱    GELU
   -0.2 │  ╱╱
        │─╱─────────────────
        -3  -2  -1   0   1   2   3
```

GELU 的優勢：在零點附近更平滑，允許小的負信號通過，有助於訓練穩定性。

---

### 概念 3.3：殘差連接 (Residual Connection)

```
沒有殘差連接：                    有殘差連接：
x → [Layer] → y                 x → [Layer] → y
                                      ↗          ↘
                                x ────────────────→ x + y

深度增加時：                     深度增加時：
梯度 = ∂L/∂x₁                  梯度 = ∂L/∂x₁
     = ∂L/∂xₙ · ∏ᵢ ∂xᵢ₊₁/∂xᵢ      = ∂L/∂xₙ · (1 + ...)
     → 指數衰減（梯度消失）             → 至少有一個直通路徑
```

殘差連接的本質：**梯度高速公路**。即使中間的層完全失效，梯度仍然能通過恆等映射（identity）流回去。

---

### 概念 3.4：Pre-Norm vs Post-Norm

GPT-2 使用 Pre-Norm（先歸一化再做注意力/FFN）：

```
Post-Norm (原始 Transformer):       Pre-Norm (GPT-2):
x → MHA → Add → LN → FFN → Add → LN   x → LN → MHA → Add → LN → FFN → Add
                                         ↗              ↘  ↗              ↘
                                    x ──────────────────→  x ─────────────→

Pre-Norm 更穩定：歸一化在子層之前，防止輸入值爆炸
```

---

### 概念 3.5：GPT 配置

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257      # BPE 詞彙表大小
    block_size: int = 1024       # 最大序列長度
    d_model: int = 768           # 嵌入維度
    n_heads: int = 12            # 注意力頭數
    n_layers: int = 12           # Transformer 層數
    dropout: float = 0.1         # Dropout 比例
```

各模型規模比較：

| 模型 | d_model | n_heads | n_layers | 參數量 |
|------|---------|---------|----------|-------|
| GPT-2 Small | 768 | 12 | 12 | 124M |
| GPT-2 Medium | 1024 | 16 | 24 | 355M |
| GPT-2 Large | 1280 | 20 | 36 | 774M |
| GPT-2 XL | 1600 | 25 | 48 | 1.5B |

注意 d_model 必須能被 n_heads 整除（head_dim = d_model / n_heads）。

---

### Lab 3: 組裝完整的 GPT 模型

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 3-4 小時
**文件**: `labs/phase3_transformer/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | transformer.py | `LayerNorm` | ⭐⭐ | 從零實現層歸一化 |
| 2 | transformer.py | `GELU` | ⭐ | GELU 激活函數 |
| 3 | transformer.py | `FeedForward` | ⭐ | 兩層 FFN |
| 4 | transformer.py | `TransformerBlock` | ⭐⭐ | Pre-norm + 殘差 |
| 5 | gpt_model.py | `GPTConfig` | ⭐ | 配置數據類 |
| 6 | gpt_model.py | `GPT.__init__()` | ⭐⭐ | 組裝所有模組 |
| 7 | gpt_model.py | `GPT.forward()` | ⭐⭐⭐ | 前向傳播 + Loss |

#### TODO 4: `TransformerBlock.forward` 偽代碼

```
函數 forward(x):
    # 注意力子層（Pre-Norm + 殘差）
    x = x + self.attn(self.ln1(x))

    # FFN 子層（Pre-Norm + 殘差）
    x = x + self.ffn(self.ln2(x))

    返回 x
```

#### TODO 7: `GPT.forward` 偽代碼

```
函數 forward(idx, targets=None):
    batch, seq_len = idx.shape

    # Embedding
    tok_emb = self.token_embedding(idx)       # (B, T, d_model)
    pos_emb = self.position_embedding(位置索引)  # (T, d_model)
    x = tok_emb + pos_emb
    x = self.dropout(x)

    # Transformer blocks
    對於每個 block in self.blocks:
        x = block(x)

    # 最後的 LayerNorm
    x = self.ln_f(x)

    # 投影到詞彙表
    logits = self.lm_head(x)  # (B, T, vocab_size)

    # 計算 Loss（如果提供了 targets）
    如果 targets 不為 None:
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        返回 logits, loss

    返回 logits, None
```

#### 常見錯誤

1. **位置 embedding 的索引**：要用 `torch.arange(seq_len, device=idx.device)`，不能硬編碼
2. **lm_head 和 token_embedding 權重共享**：GPT-2 把這兩個層的權重綁定（weight tying），可以減少參數量
3. **Dropout 的位置**：在 embedding 後、attention 後、FFN 後都要加

---

### 整合測試

```bash
cd labs/phase3_transformer
python grade.py
```

預期輸出：
```
=== Phase 3 Grading ===
[PASS] test_layer_norm_mean_var .............. 10/10
[PASS] test_gelu_activation .................. 5/5
[PASS] test_feedforward_shape ................ 10/10
[PASS] test_transformer_block_shape .......... 10/10
[PASS] test_gpt_output_shape ................ 15/15
[PASS] test_gpt_loss_computation ............. 15/15
[PASS] test_parameter_count .................. 10/10

Total: 75/75 (100%)
```

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| LayerNorm | 穩定每層的輸出分布 | 沒有它，深度 Transformer 無法訓練 |
| GELU | 平滑的激活函數 | 比 ReLU 更適合 Transformer |
| 殘差連接 | 梯度直通路徑 | 允許訓練幾十甚至上百層 |
| Pre-Norm | 歸一化在子層前 | 更穩定的訓練動態 |
| GPT 組裝 | 所有組件的整合 | 從零件到完整引擎 |

```
Phase 3 完成的模組：

┌──────────┐     ┌──────────┐     ┌──────────────┐
│ Tokenizer│ ──→ │DataLoader│ ──→ │   完整 GPT    │
│ (Ph.1)   │     │ (Ph.1)   │     │   模型 ✓     │
└──────────┘     └──────────┘     └──────────────┘

下一步：Phase 4 將在真實數據上訓練這個模型。
```

### 參考資料

**必讀**：
- [Radford et al., 2019 — Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Ba et al., 2016 — Layer Normalization](https://arxiv.org/abs/1607.06450)

**深入閱讀**：
- [Xiong et al., 2020 — On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [He et al., 2016 — Deep Residual Learning](https://arxiv.org/abs/1512.03385)

**擴展思考**：
- 如果移除所有殘差連接，一個 12 層的 Transformer 能訓練嗎？（提示：實驗看看梯度的大小）
- Weight tying（lm_head 和 embedding 共享權重）為什麼有效？（提示：想想 embedding 空間的對稱性）

---

## Phase 4: 預訓練

### 概覽

> **核心問題**：如何讓一個隨機初始化的模型學會語言？

Phase 3 的 GPT 模型權重是隨機的 — 它輸出的是亂碼。預訓練的目標是讓模型在大量文本上學習語言的統計規律。

```
                    Phase 4 架構
┌──────────────────────────────────────────────┐
│                                              │
│  訓練數據 (文本) ──→ DataLoader ──→ 批次     │
│                                    │         │
│                              ┌─────▼─────┐   │
│                              │  GPT 模型  │   │
│                              └─────┬─────┘   │
│                                    │         │
│                              ┌─────▼─────┐   │
│                              │  Logits    │   │
│                              └─────┬─────┘   │
│                                    │         │
│  ┌─────────────────────────────────▼───────┐ │
│  │         Cross-Entropy Loss              │ │
│  │  L = -Σ log P(next_token | context)     │ │
│  └─────────────────────────────┬───────────┘ │
│                                │             │
│                          ┌─────▼─────┐       │
│                          │ Backward  │       │
│                          │ + AdamW   │       │
│                          └─────┬─────┘       │
│                                │             │
│                    ┌───────────▼───────────┐ │
│                    │  更新後的 GPT 模型     │ │
│                    └───────────────────────┘ │
│                                              │
│  Learning Rate Schedule:                     │
│  ┌───────────────────────────────────┐       │
│  │  lr                               │       │
│  │  ↑  ╱╲                            │       │
│  │  │ ╱  ╲                           │       │
│  │  │╱    ╲                          │       │
│  │  │      ╲     cosine decay        │       │
│  │  │       ╲────────────────────    │       │
│  │  └──────────────────────────→ step│       │
│  │  warmup                           │       │
│  └───────────────────────────────────┘       │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：預訓練 = 大量閱讀。模型通過預測下一個詞，被迫理解語法、語義、事實知識和推理模式。

---

### 概念 4.1：交叉熵損失

模型輸出 logits（每個 token 位置對詞彙表的未歸一化分數）。交叉熵損失衡量模型的預測與真實下一個 token 的差距：

```
預測 logits:  [2.1, 0.5, -1.3, 0.8, ...]  (vocab_size 維)
真實 token:   token_id = 0

softmax:     [0.65, 0.13, 0.02, 0.18, ...]
loss = -log(P(正確token)) = -log(0.65) = 0.43

loss 越小 → 模型越確信正確答案 → 預測越準
```

---

### 概念 4.2：AdamW 優化器

```
Adam:   θ = θ - lr × m/(√v + ε)
AdamW:  θ = θ - lr × m/(√v + ε) - lr × λ × θ
                                    ↑ weight decay
                                    直接衰減權重（正則化）

其中:
  m = 一階動量（梯度的指數移動平均）
  v = 二階動量（梯度平方的指數移動平均）
  λ = weight decay 係數（通常 0.01-0.1）
```

AdamW 的 weight decay 不同於 L2 正則化 — 它直接在參數更新時衰減權重，而不是加到損失函數裡。

---

### 概念 4.3：學習率排程

```
Cosine Schedule with Warmup:

lr(step) =
  如果 step < warmup_steps:
    max_lr × step / warmup_steps          (線性預熱)
  否則:
    min_lr + 0.5 × (max_lr - min_lr) ×    (餘弦衰減)
      (1 + cos(π × (step - warmup_steps) / (max_steps - warmup_steps)))

典型值:
  max_lr = 6e-4
  min_lr = 6e-5  (max_lr 的 1/10)
  warmup_steps = max_steps 的 5-10%
```

為什麼需要 warmup？訓練初期，模型參數是隨機的，梯度估計不穩定。小學習率讓 Adam 的動量估計先穩定下來。

---

### 概念 4.4：梯度裁剪

```
如果 ||g|| > max_norm:
    g = g × max_norm / ||g||

作用：防止某個異常批次產生巨大梯度，導致參數跳到很差的位置。
典型值：max_norm = 1.0
```

---

### Lab 4: 預訓練 GPT 模型

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 3-4 小時
**文件**: `labs/phase4_pretraining/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | utils.py | `get_lr()` | ⭐⭐ | 餘弦學習率排程 |
| 2 | utils.py | `estimate_loss()` | ⭐ | 評估平均損失 |
| 3 | utils.py | `save/load_checkpoint()` | ⭐ | 檢查點管理 |
| 4 | train.py | `TrainConfig` | ⭐ | 訓練配置 |
| 5 | train.py | `train()` | ⭐⭐⭐ | 完整訓練迴圈 |

#### TODO 5: `train` 偽代碼

```
函數 train(model, train_loader, val_loader, config):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    對於 step 從 0 到 config.max_steps:
        # 學習率排程
        lr = get_lr(step, config.warmup_steps, config.max_steps, ...)
        設置 optimizer 的 lr = lr

        # 前向傳播
        batch = 從 train_loader 取一個批次
        logits, loss = model(batch.x, batch.y)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 定期評估
        如果 step % config.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, config.eval_steps)
            打印(step, train_loss, val_loss, lr)
```

#### 常見錯誤

1. **忘記 model.train() / model.eval()**：Dropout 在訓練和推理時行為不同
2. **學習率排程的邊界條件**：warmup 結束時 lr 應該等於 max_lr
3. **忘記 zero_grad()**：不清零會累積梯度

---

### 整合測試

```bash
cd labs/phase4_pretraining
python grade.py
```

預期輸出：
```
=== Phase 4 Grading ===
[PASS] test_cosine_lr_warmup ................ 10/10
[PASS] test_cosine_lr_decay ................. 10/10
[PASS] test_checkpoint_roundtrip ............ 10/10
[PASS] test_training_step ................... 15/15
[PASS] test_gradient_clipping ............... 10/10

Total: 55/55 (100%)
```

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| Cross-Entropy Loss | 預測分布與真實分布的差距 | LLM 訓練的核心目標函數 |
| AdamW | 帶 weight decay 的自適應優化器 | 所有現代 LLM 的標準優化器 |
| LR Schedule | Warmup + Cosine Decay | 穩定訓練、避免過擬合 |
| Gradient Clipping | 限制梯度最大範數 | 防止訓練爆炸 |

### 參考資料

**必讀**：
- [Loshchilov & Hutter, 2019 — Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [Chinchilla: Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)

**深入閱讀**：
- [GPT-3 Training Details (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [How to Train Really Large Models on Many GPUs (Lillian Weng)](https://lilianweng.github.io/posts/2021-09-25-train-large/)

**擴展思考**：
- 如果你只有 100MB 的訓練數據，應該用多大的模型？（提示：Chinchilla scaling laws）
- Weight decay 對 embedding 層應該生效嗎？（提示：想想 1D bias 和 LayerNorm 參數）

---

## Phase 5: 文本生成

### 概覽

> **核心問題**：訓練好的模型如何一個一個地「吐出」文字？

預訓練完成後，模型能給每個位置輸出下一個 token 的概率分布。但從概率到實際文字，需要一個**解碼策略** — 這個選擇會極大影響生成文本的品質。

```
                    Phase 5 架構
┌──────────────────────────────────────────────┐
│                                              │
│  Prompt: "The meaning of life is"            │
│       │                                      │
│       ▼                                      │
│  ┌──────────┐                                │
│  │ GPT 模型 │ → logits: [0.1, 2.3, -1.5, ...]│
│  └──────────┘                                │
│       │                                      │
│       ▼                                      │
│  ┌──────────────────────────────────┐        │
│  │        解碼策略選擇               │        │
│  │                                  │        │
│  │  Greedy    │ 永遠選最高分         │        │
│  │  Temp      │ 調整分布「銳利度」    │        │
│  │  Top-k     │ 只從前 k 個候選取    │        │
│  │  Top-p     │ 從累積概率 ≥ p 的取  │        │
│  └──────────────┬───────────────────┘        │
│                 │                            │
│                 ▼                            │
│         選中的 token                         │
│                 │                            │
│                 ▼                            │
│     追加到序列，重複直到結束                    │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：生成 = 自回歸循環。每次只生成一個 token，然後把它加到輸入中，再預測下一個。解碼策略決定了「確定性 vs 創造性」的權衡。

---

### 概念 5.1：Greedy Decoding

```
logits = [1.2, 3.5, 0.8, 2.1, ...]
token  = argmax(logits) = 1    ← 永遠選概率最高的

優點：確定性，可重現
缺點：重複、無聊、容易卡在循環裡
  "The cat sat on the mat on the mat on the mat..."
```

---

### 概念 5.2：Temperature Scaling

```
temperature = 1.0（原始分布）:
  probs = softmax([1.2, 3.5, 0.8]) = [0.14, 0.75, 0.10]

temperature = 0.5（更銳利 → 更確定）:
  probs = softmax([2.4, 7.0, 1.6]) = [0.01, 0.99, 0.005]

temperature = 2.0（更平坦 → 更隨機）:
  probs = softmax([0.6, 1.75, 0.4]) = [0.24, 0.41, 0.20]

temperature → 0: 等同 Greedy
temperature → ∞: 等同均勻分布
```

---

### 概念 5.3：Top-k 和 Top-p Sampling

```
Top-k (k=3):
  所有 token 概率: [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]
  只保留前 3:      [0.35, 0.25, 0.15,  0,    0,    0,    0,    0  ]
  重新歸一化:      [0.47, 0.33, 0.20]
  從這 3 個中採樣

Top-p / Nucleus (p=0.9):
  排序後概率:      [0.35, 0.25, 0.15, 0.10, 0.08, ...]
  累積概率:        [0.35, 0.60, 0.75, 0.85, 0.93, ...]
                                                ↑ 超過 0.9
  保留前 5 個 token，丟棄剩餘
```

Top-p 的優勢：自適應候選數量。概率集中時只保留少量 token，概率分散時保留更多。

---

### Lab 5: 實現文本生成引擎

**難度**: ⭐⭐ (中等)
**預計時間**: 2-3 小時
**文件**: `labs/phase5_generation/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | generate.py | `greedy_decode()` | ⭐ | 最簡單的策略 |
| 2 | generate.py | `temperature_sample()` | ⭐ | 溫度調節 |
| 3 | generate.py | `top_k_sample()` | ⭐⭐ | Top-k 過濾 |
| 4 | generate.py | `top_p_sample()` | ⭐⭐ | Nucleus sampling |
| 5 | generate.py | `generate()` | ⭐⭐ | 統一介面 |

#### TODO 4: `top_p_sample` 偽代碼

```
函數 top_p_sample(model, idx, max_new_tokens, p, temperature):
    重複 max_new_tokens 次:
        logits = model(idx)[:, -1, :]   # 最後一個位置的 logits
        logits = logits / temperature

        # 排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)

        # 找到累積概率超過 p 的位置
        sorted_mask = cumulative_probs - softmax(sorted_logits) >= p
        sorted_logits[sorted_mask] = -inf

        # 恢復原始順序並採樣
        probs = softmax(sorted_logits)
        next_token = 從 sorted_indices 按 probs 採樣
        idx = concat(idx, next_token)

    返回 idx
```

#### 常見錯誤

1. **Temperature = 0 導致除零**：要特殊處理，直接用 greedy
2. **忘記只取最後一個位置的 logits**：`logits[:, -1, :]` 不是 `logits`
3. **Top-p 的邊界條件**：至少保留一個 token（避免空候選集）

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| Greedy | 最簡單但最無聊 | 基準方法，deterministic |
| Temperature | 控制隨機性 | 創造性 vs 準確性的旋鈕 |
| Top-k | 固定候選數 | 避免採樣到極低概率的 token |
| Top-p | 自適應候選數 | 更靈活的控制 |

### 參考資料

**必讀**：
- [Holtzman et al., 2020 — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [How to generate text (Hugging Face)](https://huggingface.co/blog/how-to-generate)

**深入閱讀**：
- [Sampling strategies comparison (Hugging Face)](https://huggingface.co/docs/transformers/generation_strategies)

**擴展思考**：
- 什麼任務適合低 temperature（如 0.2），什麼適合高 temperature（如 1.0）？（提示：代碼生成 vs 創意寫作）
- KV-Cache 如何加速自回歸生成？（提示：想想哪些計算是重複的）

---

## Phase 6: 微調分類

### 概覽

> **核心問題**：如何把一個「會寫文章」的模型變成「會分類」的模型？

預訓練的 GPT 學到了豐富的語言知識，但它只會做一件事：預測下一個 token。微調（Fine-Tuning）讓我們用少量標注數據，把這些通用知識遷移到特定任務上。

```
                    Phase 6 架構
┌──────────────────────────────────────────────┐
│                                              │
│  預訓練的 GPT                                │
│  ┌──────────────────────────────────┐        │
│  │ Embedding + Transformer Blocks   │ 凍結   │
│  │ （豐富的語言知識）                │ 🔒    │
│  └─────────────────┬────────────────┘        │
│                    │                         │
│                    ▼ 最後一個 token 的隱藏狀態 │
│                                              │
│  ┌─────────────────────────────┐             │
│  │ Classification Head (新加)   │ 可訓練 🔓  │
│  │ Linear(d_model → n_classes) │             │
│  └─────────────────┬───────────┘             │
│                    │                         │
│                    ▼                         │
│              [Ham, Spam]                     │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：微調 = 站在巨人的肩膀上。預訓練模型已經「理解」了語言，我們只需要教它做分類決策。

---

### 概念 6.1：遷移學習 (Transfer Learning)

```
策略 1: 特徵提取（凍結骨幹）
  優點：快、需要的數據少、不會破壞預訓練知識
  缺點：表達能力有限
  適用：數據量少（< 1000 樣本）

策略 2: 全模型微調
  優點：任務適應性最強
  缺點：慢、需要更多數據、可能過擬合、可能遺忘預訓練知識
  適用：數據量大（> 10000 樣本）

策略 3: 漸進解凍
  先凍結全部 → 訓練分類頭 → 逐層解凍 → 微調
  適用：中等數據量
```

---

### 概念 6.2：為什麼用最後一個 Token？

GPT 是因果模型 — 只有最後一個 token 「看過」整個序列：

```
"This email is spam"
  t₁    t₂  t₃  t₄

t₁ 看到: [t₁]              ← 只看到自己
t₂ 看到: [t₁, t₂]          ← 看到前兩個
t₃ 看到: [t₁, t₂, t₃]     ← 看到前三個
t₄ 看到: [t₁, t₂, t₃, t₄] ← 看到整個序列 ✓

所以用 t₄ 的隱藏狀態做分類最合理
```

---

### Lab 6: 微調垃圾郵件分類器

**難度**: ⭐⭐ (中等)
**預計時間**: 2-3 小時
**文件**: `labs/phase6_classification/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | dataset.py | `SpamDataset` | ⭐⭐ | 數據處理 + Padding |
| 2 | classifier.py | `GPTClassifier` | ⭐⭐ | 分類頭設計 |
| 3 | classifier.py | `freeze/unfreeze` | ⭐ | 參數凍結控制 |
| 4 | train_classifier.py | `train_epoch()` | ⭐⭐ | 分類訓練迴圈 |
| 5 | train_classifier.py | `evaluate()` | ⭐ | 評估指標計算 |

#### 常見錯誤

1. **忘記用最後一個有效 token**：如果有 padding，最後一個 token 不是序列末尾
2. **類別不平衡**：垃圾郵件數據通常不平衡，考慮用 weighted loss
3. **學習率太大**：微調的學習率通常是預訓練的 1/10 到 1/100

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| 遷移學習 | 用預訓練知識做新任務 | 大幅減少訓練數據和時間 |
| 特徵提取 | 凍結骨幹只訓練頭 | 最安全、最快的微調方式 |
| 分類頭 | 最後 token → Linear → 類別 | 把生成模型變成分類模型 |

### 參考資料

**必讀**：
- [Howard & Ruder, 2018 — Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146)
- [Radford et al., 2018 — Improving Language Understanding by Generative Pre-Training (GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

**深入閱讀**：
- [A Survey of Transfer Learning in NLP](https://arxiv.org/abs/2007.04239)

**擴展思考**：
- 如果分類任務和預訓練數據差異很大（如醫學文本），凍結骨幹還有效嗎？
- BERT 做分類時用 [CLS] token，GPT 用最後一個 token。為什麼不同？

---

## Phase 7: LoRA — 參數高效微調

### 概覽

> **核心問題**：能不能不用更新所有參數，就能有效微調模型？

Phase 6 的全量微調需要更新所有參數，對大模型來說代價太高。LoRA (Low-Rank Adaptation) 提出了一個優雅的解決方案：用低秩矩陣分解，只訓練極少量的新參數。

```
                    Phase 7 架構
┌──────────────────────────────────────────────┐
│                                              │
│  原始權重矩陣 W (d × d)                      │
│  ┌────────────────────────┐                  │
│  │                        │                  │
│  │    W (凍結 🔒)          │  參數量: d²      │
│  │                        │                  │
│  └────────────────────────┘                  │
│              +                               │
│  LoRA 低秩更新 ΔW = B × A                    │
│  ┌────┐   ┌────────────────────────┐         │
│  │    │   │                        │         │
│  │ B  │ × │        A               │         │
│  │d×r │   │       r×d              │         │
│  │    │   │                        │         │
│  └────┘   └────────────────────────┘         │
│  參數量: d×r + r×d = 2dr                     │
│                                              │
│  當 d=768, r=4:                              │
│    全量微調: 768² = 589,824                   │
│    LoRA:    2×768×4 = 6,144                  │
│    壓縮比:  ~96x !!!                         │
│                                              │
│  輸出 = W·x + (α/r) · B·A·x                 │
│         ↑ 原始    ↑ LoRA 修正                │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：LoRA 的洞察是 — 微調時的權重變化 ΔW 是**低秩的**。也就是說，你不需要改動整個矩陣，只需要在一個低維子空間中做調整。

---

### 概念 7.1：低秩分解的直覺

```
一個 768×768 的矩陣有 ~59 萬個參數
但微調時的變化 ΔW 通常只在一個很小的子空間裡

ΔW ≈ B × A

其中 B: (768, 4), A: (4, 768)
只有 ~6000 個參數

    d=768
    ┌─────────────────────────────┐
    │                             │
    │         ΔW (768×768)        │  ← 實際上低秩
d=768│         ≈ B × A             │
    │                             │
    │                             │
    └─────────────────────────────┘

    等價於：

    ┌──┐     ┌─────────────────────────────┐
    │  │     │                             │
    │B │  ×  │            A                │
    │768│     │          (4×768)             │
    │×4 │     │                             │
    │  │     └─────────────────────────────┘
    └──┘
```

---

### 概念 7.2：LoRA 的關鍵設計

```
1. 初始化：
   A ~ Kaiming Uniform（隨機）
   B = 0（零矩陣）

   為什麼 B=0？ → 訓練開始時 ΔW = B×A = 0
   → 模型行為和原始模型完全相同
   → 訓練是漸進式的，不會突然改變模型

2. 縮放因子 α/r：
   output = W·x + (α/r) · B·A·x

   α 是一個超參數（通常設為 r 或 2r）
   α/r 讓不同 rank 的 LoRA 有類似的學習動態

3. 目標模組：
   通常只對注意力的 Q 和 V 投影加 LoRA
   （實驗顯示這最有效，K 投影影響較小）
```

---

### 概念 7.3：合併與部署

```
訓練後合併：
  W' = W + (α/r) · B × A

合併後的模型和全量微調的模型結構完全相同！
→ 推理時沒有額外開銷
→ 可以為不同任務保存不同的 LoRA 權重，共享同一個底座模型

┌──────────────┐
│  Base Model   │ ← 一份，所有任務共享
│  (frozen)     │
└──────┬───────┘
       │
  ┌────┼────────────┐
  │    │             │
  ▼    ▼             ▼
┌───┐ ┌───┐       ┌───┐
│LoRA│ │LoRA│       │LoRA│
│任務A│ │任務B│       │任務C│  ← 每個只有幾 MB
└───┘ └───┘       └───┘
```

---

### Lab 7: 實現 LoRA 微調

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 2-3 小時
**文件**: `labs/phase7_lora/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | lora.py | `LoRALinear` | ⭐⭐⭐ | 核心 LoRA 層 |
| 2 | lora.py | `apply_lora_to_model()` | ⭐⭐ | 替換模型中的層 |
| 3 | lora.py | `merge_lora_weights()` | ⭐⭐ | 合併 LoRA 權重 |
| 4 | lora.py | `count_trainable_params()` | ⭐ | 統計可訓練參數 |
| 5 | train_lora.py | `train_with_lora()` | ⭐⭐ | LoRA 訓練迴圈 |

#### TODO 1: `LoRALinear` 偽代碼

```
class LoRALinear(nn.Module):
    __init__(in_features, out_features, rank, alpha):
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False  # 凍結原始權重

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A)

        self.scaling = alpha / rank

    forward(x):
        original = self.W(x)
        lora = (x @ self.A.T @ self.B.T) * self.scaling
        return original + lora
```

#### 常見錯誤

1. **B 的初始化**：B 必須是零矩陣，不能用隨機初始化
2. **忘記凍結原始權重**：`requires_grad = False`
3. **矩陣乘法順序**：ΔW = B × A，不是 A × B

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| 低秩假設 | 微調變化在低維子空間 | LoRA 的理論基礎 |
| LoRA 層 | W + (α/r)BA | 96%+ 的參數壓縮 |
| 合併部署 | W' = W + ΔW | 推理零開銷 |
| 多任務適配 | 同一底座 + 不同 LoRA | 存儲和部署效率 |

### 參考資料

**必讀**：
- [Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Practical Tips for Finetuning LLMs Using LoRA (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

**深入閱讀**：
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

**擴展思考**：
- Rank r 設太大或太小會怎樣？（提示：太大 → 接近全量微調，太小 → 容量不足）
- 為什麼 LoRA 通常只加在 Q 和 V 上，不加在 K 上？（提示：實驗結果，參見原論文 Table 6）

---

## Phase 8: 指令微調

### 概覽

> **核心問題**：如何把一個「會補全文字」的模型變成「會回答問題」的助手？

預訓練的 GPT 只會接龍 — 給它「What is Python?」，它可能接著寫「What is Java? What is...」。我們需要教它：看到問題就回答，看到指令就執行。

```
                    Phase 8 架構
┌──────────────────────────────────────────────┐
│                                              │
│  Stage 1: Supervised Fine-Tuning (SFT)       │
│  ┌────────────────────────────────────────┐  │
│  │ Instruction: "Summarize this text."    │  │
│  │ Input: "The quick brown fox..."        │ ←不計算 loss │
│  │ Response: "A fox jumps over a dog."    │ ←計算 loss  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Stage 2: DPO Alignment                      │
│  ┌────────────────────────────────────────┐  │
│  │ Prompt: "Tell me a joke"               │  │
│  │ Chosen:  "Why did the chicken..."  ✓   │  │
│  │ Rejected: "I can't tell jokes"     ✗   │  │
│  │                                        │  │
│  │ Loss = -log σ(β(log π/π_ref(chosen)    │  │
│  │              - log π/π_ref(rejected)))  │  │
│  └────────────────────────────────────────┘  │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：SFT 教模型「如何回答」，DPO 教模型「如何回答得更好」。

---

### 概念 8.1：指令數據格式

```
Alpaca 格式：
{
    "instruction": "Translate the following to French.",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}

模板化成文本：
"### Instruction:
Translate the following to French.

### Input:
Hello, how are you?

### Response:
Bonjour, comment allez-vous?"
```

---

### 概念 8.2：SFT 中的 Label Masking

```
整個序列: "### Instruction:\nSummarize\n\n### Response:\nShort version."
Labels:   [-100, -100, -100, -100, -100, -100, token₁, token₂, token₃]
                     ↑ 不計算 loss              ↑ 只對 Response 計算 loss

為什麼？ 我們不希望模型「學會生成指令」，只希望它「學會給出回答」。
```

---

### 概念 8.3：DPO (Direct Preference Optimization)

RLHF (PPO) 很複雜：需要訓練 reward model + 用 RL 訓練 policy。
DPO 簡化了這個流程 — 直接從偏好數據學習，不需要顯式的 reward model。

```
DPO Loss:
  L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

其中:
  π     = 正在訓練的模型（policy）
  π_ref = 凍結的參考模型（SFT 後的模型）
  y_w   = 偏好的回答（chosen/winner）
  y_l   = 不偏好的回答（rejected/loser）
  β     = 溫度參數（控制偏離參考模型的程度）

直覺：
  讓 π(chosen) 的概率增加
  讓 π(rejected) 的概率減少
  但不要偏離 π_ref 太遠（通過 KL 約束）
```

---

### Lab 8: 指令微調與 DPO 對齊

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 3-4 小時
**文件**: `labs/phase8_instruction_tuning/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | sft.py | `InstructionDataset` | ⭐⭐ | 指令數據處理 + Label Masking |
| 2 | sft.py | `train_sft()` | ⭐⭐ | SFT 訓練迴圈 |
| 3 | dpo.py | `PreferenceDataset` | ⭐⭐ | 偏好數據處理 |
| 4 | dpo.py | `dpo_loss()` | ⭐⭐⭐ | DPO 損失計算 |
| 5 | dpo.py | `train_dpo()` | ⭐⭐ | DPO 訓練迴圈 |
| 6 | evaluate.py | `generate_responses()` | ⭐ | 批量生成回答 |

#### TODO 4: `dpo_loss` 偽代碼

```
函數 dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, beta):
    # 計算 policy model 的 log prob
    policy_chosen_logprobs = 計算 log P_policy(chosen_ids)
    policy_rejected_logprobs = 計算 log P_policy(rejected_ids)

    # 計算 reference model 的 log prob (不需梯度)
    with torch.no_grad():
        ref_chosen_logprobs = 計算 log P_ref(chosen_ids)
        ref_rejected_logprobs = 計算 log P_ref(rejected_ids)

    # DPO loss
    chosen_rewards = beta * (policy_chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (policy_rejected_logprobs - ref_rejected_logprobs)

    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()

    返回 loss
```

#### 常見錯誤

1. **Label masking 的邊界**：確保 Response 的第一個 token 也包含在 loss 計算中
2. **DPO 的 reference model 必須凍結**：不能有梯度流過
3. **Log probability 的計算**：需要用 log_softmax + gather，不是直接用 logits

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| SFT | 用指令-回答對監督微調 | 讓模型學會「對話」格式 |
| Label Masking | 只對回答部分計算 loss | 避免學習生成指令 |
| DPO | 從偏好數據直接優化 | 比 RLHF 簡單得多 |
| 對齊 | 讓模型產出人類偏好的回答 | 安全性和有用性的關鍵 |

### 參考資料

**必讀**：
- [Rafailov et al., 2023 — Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Ouyang et al., 2022 — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

**深入閱讀**：
- [Taori et al., 2023 — Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Tunstall et al., 2023 — Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)

**擴展思考**：
- DPO 相比 PPO 有什麼局限性？（提示：DPO 假設 reward 可以用 optimal policy 表示）
- 指令數據的品質和數量，哪個更重要？（提示：參考 LIMA 論文 — Less Is More for Alignment）

---

## Phase 9: Mixture of Experts (MoE)

### 概覽

> **核心問題**：如何在不增加推理成本的前提下，大幅增加模型容量？

傳統的 Dense Transformer 在前向傳播時激活所有參數。MoE 的核心洞察是：**不是所有參數都需要同時工作**。

```
                    Phase 9 架構
┌──────────────────────────────────────────────┐
│                                              │
│  Dense FFN:                                  │
│  每個 token → 同一個 FFN → 輸出              │
│                                              │
│  MoE FFN:                                    │
│  每個 token → Router → 選 Top-k 專家         │
│                                              │
│  ┌──────────────────────────────────┐        │
│  │          Router (Gate)           │        │
│  │   softmax(W_gate · x)           │        │
│  │   [0.05, 0.02, 0.45, 0.03,     │        │
│  │    0.01, 0.38, 0.04, 0.02]     │        │
│  └───────────┬──────────────────────┘        │
│              │ 選 Top-2                      │
│         ┌────┴────┐                          │
│         ▼         ▼                          │
│  ┌──────────┐ ┌──────────┐                   │
│  │ Expert 3 │ │ Expert 6 │  ← 只激活 2/8     │
│  │ (0.45)   │ │ (0.38)   │    的專家         │
│  └────┬─────┘ └────┬─────┘                   │
│       │            │                          │
│       ▼            ▼                          │
│  加權組合: 0.54 × E3(x) + 0.46 × E6(x)      │
│                                              │
│  總參數: 8× ↑   活躍參數: 2× ≈ Dense         │
│                                              │
└──────────────────────────────────────────────┘
```

**核心心智模型**：MoE = 專家會診。Router 是分診台，決定每個 token 應該由哪些專家處理。不同的 token 可能需要不同類型的「專家知識」。

---

### 概念 9.1：稠密 vs 稀疏

```
Dense Transformer (GPT-2 124M):
  每個 token 激活 124M 個參數
  計算量 ∝ 124M

MoE Transformer (8 experts, top-2):
  總參數 ~= 124M × 4 ≈ 500M  （FFN 部分 ×8，其餘共享）
  每個 token 只激活 ~150M 個參數
  計算量 ∝ 150M（和 Dense 差不多！）

同樣的計算量，4× 的模型容量！
```

| 指標 | Dense (124M) | MoE (500M, top-2/8) |
|------|-------------|---------------------|
| 總參數 | 124M | 500M |
| 活躍參數/token | 124M | ~150M |
| 推理 FLOPs | 1× | ~1.2× |
| 模型容量 | 1× | ~4× |
| 內存需求 | 1× | ~4× |

---

### 概念 9.2：Router / Gate Network

```python
class Router(nn.Module):
    def __init__(self, d_model, n_experts):
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        logits = self.gate(x)                    # (batch, seq_len, n_experts)
        probs = F.softmax(logits, dim=-1)        # 路由概率
        top_k_probs, top_k_indices = probs.topk(k)  # 選 top-k
        return top_k_probs, top_k_indices
```

---

### 概念 9.3：負載均衡損失

沒有均衡約束的話，Router 可能把所有 token 都送給同一個專家（「贏者通吃」）：

```
不均衡的路由：
  Expert 1: 90% tokens  ← 過載！
  Expert 2: 5% tokens
  ...
  Expert 8: 0.5% tokens ← 浪費！

均衡損失 = N × Σᵢ fᵢ · Pᵢ

其中:
  N = 專家數量
  fᵢ = 分配給專家 i 的 token 比例
  Pᵢ = 路由到專家 i 的平均概率

當路由完全均勻時：fᵢ = Pᵢ = 1/N → Loss 最小
```

---

### 概念 9.4：MoE in Practice

現實中的 MoE 架構（Mixtral, Switch Transformer）：

```
┌──────────────────────────────────────────────┐
│         MoE GPT 模型                         │
│                                              │
│  Layer 1:  標準 TransformerBlock              │
│  Layer 2:  MoE TransformerBlock              │
│  Layer 3:  標準 TransformerBlock              │
│  Layer 4:  MoE TransformerBlock              │
│  ...                                         │
│                                              │
│  不是每層都用 MoE — 交替使用                   │
│  這是 Mixtral 的做法                          │
└──────────────────────────────────────────────┘
```

---

### Lab 9: 實現 MoE Transformer

**難度**: ⭐⭐⭐ (較難)
**預計時間**: 3-4 小時
**文件**: `labs/phase9_moe/`

#### TODO 清單

| # | 文件 | 函數 | 難度 | 說明 |
|---|------|------|------|------|
| 1 | moe.py | `Expert` | ⭐ | 單個專家網路 |
| 2 | moe.py | `Router` | ⭐⭐ | 門控路由 |
| 3 | moe.py | `MoELayer` | ⭐⭐⭐ | 稀疏前向計算 |
| 4 | moe.py | `load_balancing_loss()` | ⭐⭐ | 均衡損失 |
| 5 | moe_transformer.py | `MoETransformerBlock` | ⭐⭐ | MoE 版 Transformer Block |
| 6 | moe_transformer.py | `MoEGPT` | ⭐⭐ | 完整 MoE 模型 |

#### TODO 3: `MoELayer.forward` 偽代碼

```
函數 forward(x):
    batch, seq_len, d_model = x.shape
    x_flat = x.reshape(-1, d_model)  # (batch*seq_len, d_model)

    # 路由
    top_k_probs, top_k_indices = self.router(x_flat)  # (N, k), (N, k)

    # 計算每個專家的輸出
    output = torch.zeros_like(x_flat)

    對於每個 expert_idx 從 0 到 n_experts:
        # 找到被路由到這個專家的 token
        mask = (top_k_indices == expert_idx 的任一列)
        如果沒有 token 被路由到這個專家: 跳過

        expert_input = x_flat[mask]
        expert_output = self.experts[expert_idx](expert_input)

        # 用路由權重加權
        weight = 對應的 top_k_probs
        output[mask] += weight * expert_output

    返回 output.reshape(batch, seq_len, d_model)
```

#### 常見錯誤

1. **忘記對 top-k 權重重新歸一化**：選出 top-k 後，權重之和不再是 1
2. **Expert 的梯度**：只有被路由到的專家才有梯度
3. **空專家處理**：某些 batch 中某些專家可能沒有收到任何 token

---

### 回顧

| 概念 | 你學到的 | 為什麼重要 |
|------|---------|-----------|
| 稀疏激活 | 每個 token 只用部分專家 | 擴展容量不增加計算 |
| Router | 門控網路做 token 分配 | MoE 的核心決策機制 |
| 負載均衡 | 防止贏者通吃 | 確保所有專家都被利用 |
| 交替 MoE | 不是每層都用 MoE | 平衡效率和效果 |

```
Phase 9 完成 — 課程完結！

你構建的完整系統：

  文本 → Tokenizer → Embedding → Multi-Head Attention
       → Transformer Block (+ MoE) → Pre-Training
       → Text Generation → Classification
       → LoRA Fine-Tuning → SFT + DPO
```

### 參考資料

**必讀**：
- [Fedus et al., 2022 — Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [Jiang et al., 2024 — Mixtral of Experts](https://arxiv.org/abs/2401.04088)

**深入閱讀**：
- [Shazeer et al., 2017 — Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

**擴展思考**：
- MoE 的專家真的會「專門化」嗎？不同的專家學到了什麼？（提示：研究 Mixtral 的專家分析論文）
- MoE 和 Model Parallelism 如何結合？（提示：Expert Parallelism — 不同的專家放在不同的 GPU 上）
