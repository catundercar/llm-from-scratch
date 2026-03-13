# LLM From Scratch

从零开始构建大语言模型 — 一门手把手的实战课程，带你用 Python + PyTorch 从 Tokenizer 一路搭建到 Mixture of Experts。

**[线上课程网站](https://catundercar.github.io/llm-from-scratch/)** · [English](README.md)

## 课程概览

| Phase | 主题 | 关键概念 |
|-------|------|----------|
| 0 | 全景图：Transformer 与 LLM | 整体架构、数据流、核心组件 |
| 1 | 文本与数据 | Tokenizer (BPE)、DataLoader、Embedding |
| 2 | 注意力机制 | Self-Attention、Multi-Head Attention、因果遮罩 |
| 3 | Transformer 架构 | GPT Block、LayerNorm、残差连接 |
| 4 | 预训练 | 语言建模损失、训练循环、学习率调度 |
| 5 | 文本生成 | Temperature、Top-k、Top-p 采样 |
| 6 | 分类微调 | 迁移学习、分类头、冻结策略 |
| 7 | LoRA | 低秩适配、参数高效微调 |
| 8 | 指令微调 | SFT、DPO、对话格式 |
| 9 | 混合专家 | MoE 路由、稀疏激活、负载均衡 |

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/catundercar/llm-from-scratch.git
cd llm-from-scratch

# 安装 Python 依赖
pip install -r requirements.txt
```

### 运行 Lab

每个 Phase 都是独立的 Python 包，包含骨架代码（带 TODO）、测试和评分脚本：

```bash
cd labs/phase1_text_and_data

# 运行测试
pytest tests/ -v

# 查看成绩
python scripts/grade.py
```

### 本地运行课程网站

```bash
cd website
npm install
npm run dev
```

## 项目结构

```
llm-from-scratch/
├── labs/                        # 实战 Lab（Python + PyTorch）
│   ├── shared/                  # 跨 Phase 共用工具
│   ├── phase1_text_and_data/    # Phase 1: Tokenizer + DataLoader
│   ├── phase2_attention/        # Phase 2: Self-Attention
│   ├── phase3_transformer/      # Phase 3: GPT Block
│   ├── phase4_pretraining/      # Phase 4: 预训练
│   ├── phase5_generation/       # Phase 5: 文本生成
│   ├── phase6_classification/   # Phase 6: 分类微调
│   ├── phase7_lora/             # Phase 7: LoRA
│   ├── phase8_instruction_tuning/ # Phase 8: SFT + DPO
│   └── phase9_moe/              # Phase 9: MoE
├── website/                     # 课程网站（Vite + React + TypeScript）
├── COURSE.md                    # 课程完整内容
├── course-roadmap.jsx           # 课程路线图原型
└── requirements.txt             # Python 依赖
```

## 技术栈

- **Lab**: Python 3.10+, PyTorch 2.0+, tiktoken
- **Website**: Vite, React 18, TypeScript, React Router
- **部署**: GitHub Pages (自动部署)

## 课程设计理念

1. **由底向上** — 从最底层的 Tokenizer 开始，逐层搭建到完整的 LLM
2. **概念先行** — 每个 Lab 前先理解 WHY，再动手 HOW
3. **骨架填空** — 框架和测试已写好，你只需实现核心算法（TODO 标记处）
4. **即时反馈** — 每个 Phase 完成后都有可运行的成果

## 参考资料

- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) — Sebastian Raschka
- [LLM Course](https://github.com/mlabonne/llm-course) — Maxime Labonne
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.

## License

MIT
