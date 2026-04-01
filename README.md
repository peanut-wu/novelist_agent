# RLWriter

> 用强化学习模仿任意作家文风，无需微调模型

## 一句话介绍

RLWriter 是一个**纯提示工程**的小说生成框架。它通过分析目标作品，自动学习作者的写作风格（句式、节奏、用词偏好等），然后用学到的风格生成新内容。

**核心特点：**
- ✅ 不需要训练模型（零参数更新）
- ✅ 支持任意中文/英文作家风格模仿
- ✅ 内置搜索增强，自动查证历史/地理细节
- ✅ 章节串联生成，保持角色和剧情一致性

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/peanut-wu/claw_rl_writer.git
cd claw_rl_writer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置 API Key

```bash
# 创建 .env 文件
cat > .env << EOF
DEEPSEEK_API_KEY=sk-your-key-here
EOF
```

> 获取 DeepSeek API Key：https://platform.deepseek.com/

### 3. 训练（学习文风）

```bash
# 用《水浒传》训练 5 步，学习其文风
python3 -m rlwriter.cli train \
  --novel data/novels/水浒传.txt \
  --config config/deepseek.yaml \
  --steps 5
```

训练完成后，会在 `output/` 目录生成 checkpoint 文件。

### 4. 生成小说

```bash
# 初始化项目
python3 gen_series.py --project novels/my_story \
  --idea "主角穿越回南宋末年，拥有超凡武力，拯救华夏的武侠故事" \
  --checkpoint output/checkpoints_final \
  --config config/deepseek.yaml \
  --prepare

# 生成前 3 章
python3 gen_series.py --project novels/my_story --chapters 1-3
```

生成结果在 `novels/my_story/chapters/` 目录。

---

## 工作原理

```
原文小说
    ↓
[预处理] 分割场景 → 提取情节摘要（剥离具体文字）
    ↓
[生成] 用当前风格提示词 + 摘要 → 生成正文
    ↓
[评分] 对比原文：文体特征 / 作者身份 / 质量
    ↓
[优化] LLM 分析差异 → 改进提示词
    ↓
[循环] 重复直到收敛
```

简单说：**不看原文怎么写，只看原文写了什么，然后用自己的话复述，再对比差距改进。**

---

## 项目结构

```
claw_rl_writer/
├── data/novels/          # 训练用小说（放这里）
├── config/               # 配置文件
│   └── deepseek.yaml     # DeepSeek 推荐配置
├── novels/               # 你的生成项目
│   └── my_story/         # 每个小说一个文件夹
│       ├── outline.json      # 大纲
│       ├── characters.json   # 角色设定
│       └── chapters/         # 生成的章节
├── output/               # 训练输出（checkpoint）
└── src/rlwriter/         # 源码
```

---

## 配置说明

### 搜索功能（可选）

默认使用 DuckDuckGo（免费），如需更高质量搜索可配置：

```yaml
# config/deepseek.yaml
search:
  enabled: true
  mode: "llm_select"     # 自动判断是否需要搜索
  api_keys:
    tavily: ""           # 高质量检索（需申请）
    exa: ""              # 语义搜索（需申请）
```

### 推理参数

```yaml
inference:
  chapter:
    max_tokens: 4096          # 单章最大长度
    target_words: "3000-5000" # 目标字数
    temperature: 0.8          # 创造性（0-1）
```

---

## 示例

本项目包含一个完整示例：

- **训练集**：《水浒传》（四大名著之一，公版）
- **生成主题**：穿越武侠（南宋末年 + 超凡武力）
- **输出**：`demo/` 目录下的前 3 章

```bash
# 查看示例
ls demo/
# outline.md      - 故事大纲
# ch001.txt       - 第一章
# ch002.txt       - 第二章
# ch003.txt       - 第三章
```

---

## 常见问题

**Q: 需要多少训练数据？**  
A: 10 万字以上效果较好，但 5 万字也能出基本风格。

**Q: 训练要多久？**  
A: 取决于步数和小说长度。5 步约 10-30 分钟，20 步约 1-2 小时。

**Q: 支持英文吗？**  
A: 支持，但分词和特征提取针对中文优化，英文效果可能略差。

**Q: 生成的内容会侵权吗？**  
A: 不会。训练只学习**风格**（句式、节奏等），不复制原文内容。请使用公版书训练。

---

## 技术细节

- **风格学习**：计算文体学 + LLM 裁判 + 作者身份验证
- **记忆机制**：MemGPT 式分层记忆，防止文风漂移
- **优化器**：APO（Automatic Prompt Optimization）文本梯度优化
- **搜索集成**：DuckDuckGo / Tavily / Exa / Wikipedia

---

## License

MIT License - 自由使用和修改。

**注意**：请遵守当地版权法规，仅使用公版书或自有版权作品进行训练。
