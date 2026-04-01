# RLWriter 开发计划

## 当前状态（2026-03-29）

### 已完成 ✅

**03-29 新增：**

| 模块 | 文件 | 说明 |
|------|------|------|
| 项目文件夹管理 | `project.py` | ProjectManager：project.yaml 驱动，自动管理大纲/角色/章节/状态文件 |
| 推理引擎 | `inference.py` | 统一推理入口，支持 outline/characters/chapter 等模式 |
| 推理配置 | `config/deepseek.yaml` inference 节 | 各环节独立配置 max_tokens/temperature/target_words 等 |
| CLI 参数重构 | `cli.py` | `--idea` 改为 outline 模式专用；characters/chapter 模式通过 `--outline-file` / `--characters-file` 传入 |
| gen_series 重构 | `gen_series.py` | 改为 `--project` 项目文件夹模式，逐章独立状态（state/chXXX/） |
| 自动续写 | inference 配置 | `auto_continue` + `max_continues`：输出截断时自动续写拼接 |

**03-28 及之前：**

| 模块 | 文件 | 测试 | 说明 |
|------|------|------|------|
| 伏笔追踪 | `foreshadow_tracker.py` | 22 tests | add/resolve/format/stats/load/save |
| NexusSum 分层压缩 | `nexus_sum.py` | 15 tests | 超6000字自动LLM压缩，checkpoint/rollback |
| 结构化模式提取 | `plot_design/analyzer.py` | 13 tests | 4维JSON输出（结构/弧线/节奏/伏笔） |
| 角色冷热分离 | `gen_series.py` + `series/utils.py` | 17 tests | active/inactive 两态，节省token |
| 章节后处理 | `series/utils.py` | 10 tests | clean_chapter_output() 去除元分析文字 |
| 搜索集成 | `search/__init__.py` | 18 tests | DuckDuckGo/Wikipedia/Tavily/Exa + llm_select |

### 架构概览

```
训练: 原文 → 场景分割 → 风格剥离摘要 → RL迭代(文体学+作者验证+LLM裁判) → checkpoint
推理: checkpoint + 大纲 + 角色 → 串联生成(伏笔追踪+NexusSum+搜索) → 章节正文
```

### 推理 Prompt 结构

```
CHAPTER_PROMPT:
  {style_rules}        ← 文风规则（训练）
  {plot_rules}         ← 情节架构规则（结构化JSON）
  {memory_warnings}    ← 写作禁忌（训练，LLM预处理缓存）
  {outline}            ← 大纲（当前章±1章）
  {characters}         ← 角色设定
  {chapter_beats}      ← 本章情节点（自动提取）
  {previous_summary}   ← NexusSum压缩后的历史
  {pending_hooks}      ← 未回收伏笔列表
  {search_context}     ← 搜索背景资料（llm_select自动决策）
```

### 串联生成流程

```
第1章 → LLM摘要 → NexusSum → 更新角色状态(冷热) → 伏笔追踪 →
第2章 → LLM摘要 → NexusSum → 更新角色状态 → 伏笔追踪 →
第3章 → ...
```

---

## 待优化

| # | 问题 | 优先级 | 说明 |
|---|------|--------|------|
| 1 | 训练产出的鹿鼎记 checkpoint plot_prompt 仍偏泛 | 中 | 需重新训练以提取更结构化的模式 |
| 2 | 长篇（>20章）连通性待验证 | 中 | NexusSum 阈值可能需调优 |
| 3 | 搜索的 llm_select 每章多一次 LLM 调用 | 低 | 可改为每N章判断一次或缓存决策 |
| 4 | quick_train.py / extract_patterns.py / preprocess_memory.py 散落在根目录 | 低 | 可移入 scripts/ |

---

## 后续方向

- [ ] 更多训练语料（不同体裁：科幻、都市、悬疑）
- [ ] Web UI 集成
- [ ] 多模型对比评估
- [ ] 实时文风迁移（输入目标作者风格描述，即时生成）
