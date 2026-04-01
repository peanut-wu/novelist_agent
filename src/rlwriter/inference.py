"""推理引擎：加载 Checkpoint → 生成大纲/人设/章节

推理流程：
  engine = InferenceEngine.load("output/checkpoints/best")
  outline = engine.generate_outline("一个穿越到古代的程序员的故事")
  characters = engine.generate_characters(outline)
  chapter = engine.generate_chapter(outline, characters, chapter_num=1)

推理阶段支持搜索引擎接口，用于背景知识调研。
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from .checkpoint import CheckpointManager, CheckpointState
from .utils import LLMConfig, InferenceConfig, get_llm_client, llm_call, extract_json_from_text


class SearchProvider(Protocol):
    """搜索引擎接口协议

    实现此协议即可接入任意搜索引擎。
    """
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """搜索

        Returns:
            列表，每个元素 {"title": str, "url": str, "snippet": str}
        """
        ...


class NoOpSearchProvider:
    """空搜索实现（默认，不搜索）"""
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return []


class WebSearchProvider:
    """Web 搜索实现（封装 Brave Search / 其他 API）

    用法：
        provider = WebSearchProvider(api_key="BSA...")
        engine = InferenceEngine.load("ckpts/best", search_provider=provider)
    """

    def __init__(self, api_key: str, api_url: str = "https://api.search.brave.com/res/v1/web/search"):
        self.api_key = api_key
        self.api_url = api_url

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """调用 Brave Search API"""
        try:
            import httpx
            resp = httpx.get(
                self.api_url,
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": self.api_key},
                timeout=10,
            )
            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
            return results
        except Exception:
            return []


@dataclass
class GenerationResult:
    """生成结果"""
    mode: str           # outline / characters / chapter
    content: str        # 生成的文本内容
    metadata: dict = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> dict:
        return {"mode": self.mode, "content": self.content, "metadata": self.metadata}


class InferenceEngine:
    """推理引擎

    从 checkpoint 加载训练好的提示词，用于推理阶段的创作。
    """

    OUTLINE_PROMPT = """你是一位专业的小说大纲设计师。请根据以下创意和要求，设计一个完整的小说大纲。

**要求：**
1. 设计 8-15 个章节的大纲
2. 每章包含 3-5 个关键情节点
3. 设计 2-4 个主要角色的弧线
4. 规划核心冲突的升级曲线
5. 标注转折点和高潮位置

{style_rules}

{search_context}

**创意/指令：**
{idea}

**输出格式：**
请用以下 JSON 格式输出：
{{
  "title": "小说标题",
  "genre": "题材类型",
  "theme": "核心主题",
  "chapters": [
    {{"chapter": 1, "title": "章节标题", "summary": "章节摘要", "beats": ["节拍1", "节拍2", ...]}}
  ],
  "main_characters": [
    {{"name": "角色名", "role": "主角/配角/反派", "arc": "弧线描述"}}
  ],
  "conflict": "核心冲突描述",
  " climax_chapter": 10
}}"""

    CHARACTERS_PROMPT = """你是一位专业的小说角色设计师。请根据以下大纲，为每个主要角色设计详细的人物设定。

**要求：**
1. 每个角色包含：外貌、性格、背景、动机、弱点、口头禅
2. 角色间的关系网络
3. 每个角色的成长弧线（从初始状态到最终状态）

{style_rules}

{search_context}

**小说大纲：**
{outline}

**输出格式：**
[
  {{
    "name": "角色名",
    "appearance": "外貌描述",
    "personality": "性格特点",
    "background": "背景故事",
    "motivation": "核心动机",
    "weakness": "致命弱点",
    "catchphrase": "口头禅",
    "arc": "成长弧线",
    "relationships": ["与其他角色的关系"]
  }}
]"""

    CHAPTER_PROMPT = """你是一位专业的小说续写助手。请根据以下信息，创作第 {chapter_num} 章正文。

**要求：**
1. 严格按照大纲的情节点展开
2. 运用给定的文风规则进行创作
3. 遵循情节架构规则，保持结构完整性
4. 严格遵守写作禁忌，避免训练中已识别的常见问题
5. 保持叙事连贯性和节奏感
6. 字数约 {target_words} 字
7. 本章标题：{chapter_title}

{style_rules}

{plot_rules}

{memory_warnings}

{pending_hooks}

{search_context}

**小说大纲：**
{outline}

**角色设定：**
{characters}

**第 {chapter_num} 章情节点：**
{chapter_beats}

**前文摘要（连贯性）：**
{previous_summary}

---

**第 {chapter_num} 章正文：**"""

    def __init__(self, state: CheckpointState, llm_cfg: LLMConfig,
                 search_provider: Optional[SearchProvider] = None,
                 search_manager=None,
                 inference_cfg=None):
        self.state = state
        self.llm_cfg = llm_cfg
        self.client = get_llm_client(llm_cfg)
        self.search = search_provider or NoOpSearchProvider()
        self.search_manager = search_manager
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.style_prompt = state.style_prompt
        self.plot_prompt = state.plot_prompt or {}
        self.memory_nodes = state.memory_nodes or {}

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """构建 messages 列表，注入 system_instruction 作为 system message（如果有）"""
        messages = []
        sys_instr = self.style_prompt.get("system_instruction", "")
        if sys_instr:
            messages.append({"role": "system", "content": sys_instr})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def ensure_memory_warnings(self, checkpoint_path: str = "") -> str:
        """确保 memory_warnings 已预处理（一次性 LLM 聚合）

        如果 checkpoint 已有缓存，直接返回。否则调 LLM 预处理并可选缓存到 checkpoint。
        """
        if self.state.memory_warnings:
            return self.state.memory_warnings

        warnings = self.preprocess_memory_warnings()
        if warnings and checkpoint_path:
            self.save_memory_warnings_to_checkpoint(warnings, checkpoint_path)

        return warnings

    @classmethod
    def load(cls, checkpoint_path: str,
             llm_cfg: Optional[LLMConfig] = None,
             search_provider: Optional[SearchProvider] = None,
             search_manager=None,
             inference_cfg=None) -> "InferenceEngine":
        """从 checkpoint 加载推理引擎

        Args:
            checkpoint_path: checkpoint 目录路径或 checkpoint 名称
            llm_cfg: LLM 配置（可选，从 checkpoint 读取）
            search_provider: 搜索引擎实现
            search_manager: 搜索管理器（SearchManager 实例）
            inference_cfg: 推理阶段配置（InferenceConfig 实例）
        """
        path = Path(checkpoint_path)
        if path.is_dir():
            ckpt_dir = str(path)
            ckpt_name = "step_" + str(max(
                int(d.name.replace("step_", ""))
                for d in path.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            )) if any(d.name.startswith("step_") for d in path.iterdir()) else "final"
        else:
            ckpt_dir = str(path.parent)
            ckpt_name = path.name

        mgr = CheckpointManager(ckpt_dir)
        state = mgr.load(ckpt_name)

        if llm_cfg is None:
            llm_cfg = LLMConfig()

        return cls(state, llm_cfg, search_provider, search_manager, inference_cfg=inference_cfg)

    def _format_style_rules(self) -> str:
        """格式化文风规则"""
        rules = self.style_prompt.get("style_rules", "")
        if rules and rules != "（待学习）":
            return f"**文风规则（来自训练）：**\n{rules}"
        return ""

    def _format_plot_rules(self) -> str:
        """格式化情节架构规则（来自 plot_design 训练）"""
        pp = self.plot_prompt
        if not pp:
            return ""
        parts = []
        for key, label in [
            ("structure_template", "结构模板"),
            ("character_arc_guide", "角色弧线"),
            ("pacing_rules", "节奏控制"),
            ("foreshadowing_guide", "伏笔设计"),
        ]:
            val = pp.get(key, "")
            if val:
                parts.append(f"- {label}：{val}")
        if parts:
            return "**情节架构规则（来自训练）：**\n" + "\n".join(parts)
        return ""

    def _format_memory_warnings(self) -> str:
        """加载 memory_warnings（优先用 LLM 预处理缓存，否则回退关键词）"""
        # 优先：checkpoint 中缓存的 LLM 预处理结果
        state_warnings = getattr(self.state, 'memory_warnings', None)
        if state_warnings:
            return state_warnings

        # 回退：关键词分类（无 LLM 调用）
        nodes = self.memory_nodes
        if not nodes:
            return ""

        import re

        diffs = []
        for v in nodes.values():
            content = v.get('content', '')
            matches = re.findall(
                r'差异点\d+:\s*(.+?)\s*→\s*建议:\s*(.+?)(?=差异点|$)',
                content, re.DOTALL,
            )
            for problem, suggestion in matches:
                diffs.append((problem.strip(), suggestion.strip()))

        if not diffs:
            return ""

        categories = {
            '环境描写': [], '对话问题': [], '视角问题': [],
            '节奏问题': [], '情节偏离': [], '心理描写': [], '比喻问题': [],
        }
        for p, s in diffs:
            if any(k in p for k in ('环境', '场景', '细节渲染')):
                categories['环境描写'].append(s)
            elif any(k in p for k in ('对话',)):
                categories['对话问题'].append(s)
            elif any(k in p for k in ('视角', '全知', '感知')):
                categories['视角问题'].append(s)
            elif any(k in p for k in ('节奏', '拖沓', '冗余', '铺陈')):
                categories['节奏问题'].append(s)
            elif any(k in p for k in ('情节', '偏离', '额外', '枝蔓')):
                categories['情节偏离'].append(s)
            elif any(k in p for k in ('心理', '内心', '理性化')):
                categories['心理描写'].append(s)
            elif any(k in p for k in ('比喻',)):
                categories['比喻问题'].append(s)

        rules = []
        for cat, suggestions in categories.items():
            if not suggestions:
                continue
            seen = set()
            unique = []
            for s in suggestions:
                key = s[:40]
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            for s in unique[:2]:
                rules.append(f"- **{cat}**：{s[:120]}")

        if rules:
            return "**写作禁忌（训练经验，关键词归纳）：**\n" + "\n".join(rules[:12])
        return ""

    def preprocess_memory_warnings(self) -> str:
        """用 LLM 聚合 memory_nodes → 精简禁忌清单（一次性预处理）"""
        nodes = self.memory_nodes
        if not nodes:
            return ""

        # 提取所有差异
        import re
        diffs = []
        for v in nodes.values():
            content = v.get('content', '')
            matches = re.findall(
                r'差异点\d+:\s*(.+?)\s*→\s*建议:\s*(.+?)(?=差异点|$)',
                content, re.DOTALL,
            )
            for problem, suggestion in matches:
                diffs.append(f"问题: {problem.strip()[:80]}\n建议: {suggestion.strip()[:80]}")

        if not diffs:
            return ""

        # 去重后截断（避免 token 过多）
        seen = set()
        unique_diffs = []
        for d in diffs:
            key = d[:40]
            if key not in seen:
                seen.add(key)
                unique_diffs.append(d)
        diff_text = "\n---\n".join(unique_diffs[:80])

        prompt = f"""以下是小说写作训练中，模型续写与原文的差异分析（共{len(unique_diffs)}条去重后）。

请将这些差异归纳为 8-12 条精简的"写作禁忌"，每条格式为：
- **类别**：具体禁忌描述（控制在一句话内）

要求：
1. 按问题类型分组（如环境描写、对话、视角、节奏、情节偏离等）
2. 每条禁忌要具体可操作，不要泛泛而谈
3. 去掉与具体小说人物/情节相关的细节，只保留通用写作原则
4. 总字数控制在 800 字以内

差异分析：
{diff_text}

写作禁忌清单："""

        messages = [{"role": "user", "content": prompt}]
        mw_cfg = self.inference_cfg.memory_warnings
        response = llm_call(self.client, self.llm_cfg, messages,
                            temperature=mw_cfg.temperature, max_tokens=mw_cfg.max_tokens)
        result = f"**写作禁忌（训练经验，LLM 归纳）：**\n{response.strip()}"
        return result

    def save_memory_warnings_to_checkpoint(self, warnings: str, ckpt_path: str):
        """将预处理结果直接写入 checkpoint state.json"""
        import json as _json
        state_file = Path(ckpt_path) / "state.json"
        if state_file.exists():
            data = _json.loads(state_file.read_text(encoding="utf-8"))
        else:
            data = self.state.to_dict()
        data["memory_warnings"] = warnings
        state_file.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.state.memory_warnings = warnings

    def _search_context(self, topic: str) -> str:
        """搜索并格式化搜索结果"""
        results = self.search.search(topic, max_results=3)
        if not results:
            return ""
        lines = ["**背景调研（搜索引擎）：**"]
        for r in results:
            lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')[:200]}")
        return "\n".join(lines)

    def generate_outline(self, idea: str, search_query: Optional[str] = None) -> GenerationResult:
        """根据创意生成小说大纲

        Args:
            idea: 创意/指令文本
            search_query: 可选的搜索查询（用于背景调研）

        Returns:
            GenerationResult (mode=outline)
        """
        search_ctx = ""
        if search_query:
            search_ctx = self._search_context(search_query)
        elif self.search_manager and self.search_manager.enabled and self.search_manager.mode == "llm_select":
            source, auto_query = self.search_manager.llm_select_source(
                self.client, idea, task_type="大纲", llm_cfg=self.llm_cfg
            )
            if auto_query:
                results = self.search_manager.search(auto_query, source=source)
                if results:
                    from .search import format_search_results
                    search_ctx = f"\n**背景调研资料：**\n{format_search_results(results)}\n"

        prompt = self.OUTLINE_PROMPT.format(
            style_rules=self._format_style_rules(),
            search_context=search_ctx,
            idea=idea,
        )
        messages = self._build_messages(prompt)
        icfg = self.inference_cfg.outline
        response = llm_call(self.client, self.llm_cfg, messages,
                            temperature=icfg.temperature, max_tokens=icfg.max_tokens,
                            auto_continue=self.inference_cfg.auto_continue,
                            max_continues=self.inference_cfg.max_continues)

        # 解析大纲
        outline_data = {}
        try:
            outline_data = extract_json_from_text(response, expect_list=False) or {}
        except Exception:
            pass

        return GenerationResult(
            mode="outline",
            content=response,
            metadata={"parsed": outline_data, "idea": idea},
        )

    def generate_characters(self, outline: str,
                            search_query: Optional[str] = None) -> GenerationResult:
        """根据大纲设计角色

        Args:
            outline: 小说大纲文本
            search_query: 可选的搜索查询

        Returns:
            GenerationResult (mode=characters)
        """
        search_ctx = ""
        if search_query:
            search_ctx = self._search_context(search_query)
        elif self.search_manager and self.search_manager.enabled and self.search_manager.mode == "llm_select":
            source, auto_query = self.search_manager.llm_select_source(
                self.client, outline[:500], task_type="角色设定", llm_cfg=self.llm_cfg
            )
            if auto_query:
                results = self.search_manager.search(auto_query, source=source)
                if results:
                    from .search import format_search_results
                    search_ctx = f"\n**背景调研资料：**\n{format_search_results(results)}\n"

        prompt = self.CHARACTERS_PROMPT.format(
            style_rules=self._format_style_rules(),
            search_context=search_ctx,
            outline=outline[:3000],
        )
        messages = self._build_messages(prompt)
        icfg = self.inference_cfg.characters
        response = llm_call(self.client, self.llm_cfg, messages,
                            temperature=icfg.temperature, max_tokens=icfg.max_tokens,
                            auto_continue=self.inference_cfg.auto_continue,
                            max_continues=self.inference_cfg.max_continues)

        # 解析角色列表
        characters_data = []
        try:
            characters_data = extract_json_from_text(response, expect_list=True) or []
        except Exception:
            pass

        return GenerationResult(
            mode="characters",
            content=response,
            metadata={"parsed": characters_data},
        )

    def generate_chapter(self, outline: str, characters: str,
                         chapter_num: int, chapter_title: str = "",
                         chapter_beats: str = "",
                         previous_summary: str = "",
                         search_query: Optional[str] = None,
                         outline_parsed: Optional[dict] = None,
                         pending_hooks: str = "") -> GenerationResult:
        """创作指定章节

        Args:
            outline: 小说大纲文本（完整）
            characters: 角色设定
            chapter_num: 章节号
            chapter_title: 章节标题
            chapter_beats: 本章情节点（优先使用）
            previous_summary: 前文摘要（连贯性）
            search_query: 可选的搜索查询
            outline_parsed: 已解析的大纲 JSON（用于智能截断）

        Returns:
            GenerationResult (mode=chapter)
        """
        search_ctx = ""
        if search_query:
            search_ctx = self._search_context(search_query)
        elif self.search_manager and self.search_manager.enabled and self.search_manager.mode == "llm_select":
            context_text = f"第{chapter_num}章: {chapter_title}\n{chapter_beats}"
            source, auto_query = self.search_manager.llm_select_source(
                self.client, context_text, task_type="章节", llm_cfg=self.llm_cfg
            )
            if auto_query:
                results = self.search_manager.search(auto_query, source=source)
                if results:
                    from .search import format_search_results
                    search_ctx = f"\n**背景调研资料：**\n{format_search_results(results)}\n"

        if not chapter_title:
            chapter_title = f"第{chapter_num}章"

        # 智能截断 outline：保留当前章+相邻章的情节点
        outline_text = self._smart_outline_truncate(outline, chapter_num, outline_parsed)

        # 自动生成 chapter_beats（如果没传且有解析数据）
        if not chapter_beats and outline_parsed:
            beats = self._extract_chapter_beats(outline_parsed, chapter_num)
            chapter_beats = beats or "（无特定情节点，按大纲自由发挥）"

        prompt = self.CHAPTER_PROMPT.format(
            style_rules=self._format_style_rules(),
            plot_rules=self._format_plot_rules(),
            memory_warnings=self._format_memory_warnings(),
            pending_hooks=pending_hooks or "",
            search_context=search_ctx,
            outline=outline_text,
            characters=characters[:3000],  # 放宽到 3000
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            chapter_beats=chapter_beats or "（无特定情节点，按大纲自由发挥）",
            previous_summary=previous_summary or "（无前文，本章为开篇）",
            target_words=self.inference_cfg.chapter.target_words or "3000-5000",
        )
        messages = self._build_messages(prompt)
        icfg = self.inference_cfg.chapter
        response = llm_call(self.client, self.llm_cfg, messages,
                            temperature=icfg.temperature, max_tokens=icfg.max_tokens,
                            auto_continue=self.inference_cfg.auto_continue,
                            max_continues=self.inference_cfg.max_continues)

        return GenerationResult(
            mode="chapter",
            content=response,
            metadata={"chapter_num": chapter_num, "title": chapter_title},
        )

    def _smart_outline_truncate(self, outline: str, chapter_num: int,
                                 outline_parsed: Optional[dict] = None) -> str:
        """智能截断大纲：优先保留当前章+相邻章"""
        if not outline_parsed or "chapters" not in outline_parsed:
            return outline[:3000]

        chapters = outline_parsed["chapters"]
        # 提取当前章 ± 1 章的内容
        relevant = []
        for ch in chapters:
            ch_num = ch.get("chapter", 0)
            if abs(ch_num - chapter_num) <= 1:
                relevant.append(ch)

        if not relevant:
            return outline[:3000]

        # 构建精简版大纲
        parts = []
        theme = outline_parsed.get("theme", "")
        if theme:
            parts.append(f"主题：{theme}")

        for ch in relevant:
            title = ch.get("title", f"第{ch.get('chapter', '?')}章")
            summary = ch.get("summary", "")
            beats = ch.get("beats", [])
            beat_str = "\n".join(f"  - {b}" for b in beats)
            parts.append(f"**第{ch.get('chapter', '?')}章「{title}」**：{summary}\n{beat_str}")

        # 也加上角色弧线
        main_chars = outline_parsed.get("main_characters", [])
        if main_chars:
            char_str = "\n".join(f"  - {c.get('name', '?')}：{c.get('arc', '')}" for c in main_chars[:4])
            parts.append(f"角色弧线：\n{char_str}")

        result = "\n\n".join(parts)
        return result[:3000] if len(result) > 3000 else result

    def _extract_chapter_beats(self, outline_parsed: dict, chapter_num: int) -> str:
        """从解析的大纲中提取指定章节的 beats"""
        for ch in outline_parsed.get("chapters", []):
            if ch.get("chapter") == chapter_num:
                beats = ch.get("beats", [])
                if beats:
                    title = ch.get("title", "")
                    header = f"「{title}」" if title else ""
                    return "\n".join(f"- {b}" for b in beats)
        return ""
