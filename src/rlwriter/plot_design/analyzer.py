"""情节结构分析器：从摘要树中提取作者的情节设计模式"""

import json
import re
from typing import Optional

from ..utils import LLMConfig, llm_call
from .structure import PlotBeat, CharacterArc, Foreshadowing, PlotStructure


class PlotStructureAnalyzer:
    """从摘要树中提取情节结构模式"""

    BEAT_PROMPT = """从小说章节摘要中提取情节节拍。
节拍类型: setup/complication/climax/resolution/twist/revelation
标注张力(0-1)、涉及角色、情感基调。

章节摘要：
{chapter_summary}

输出 JSON 数组：
[{{"beat_type":"xxx","summary":"一句话","tension":0.x,"characters":["A"],"tone":"xxx"}}]"""

    ARC_PROMPT = """从情节摘要中提取角色弧线变化。
追踪动机变化、状态转变、关系变化。

全书摘要：{book_summary}
章节摘要：{chapter_summaries}

输出 JSON 数组：
[{{"character":"名字","scene_index":0,"motivation":"动机","state":"状态变化","relationship_changes":["xxx"]}}]"""

    FORESHADOWING_PROMPT = """识别伏笔（铺垫→回收）对。
未回收的伏笔 payoff_scene 设为 -1。

场景摘要：
{scene_summaries}

输出 JSON 数组：
[{{"setup_scene":0,"payoff_scene":5,"element":"描述","subtlety":0.x}}]"""

    TENSION_PROMPT = """为每个章节评估叙事张力和信息密度(0-1)。

章节摘要：
{chapter_summaries}

输出 JSON 数组：
[{{"tension":0.x,"pacing":0.x}}]"""

    def __init__(self, client, llm_cfg: LLMConfig):
        self.client = client
        self.llm_cfg = llm_cfg

    def analyze(self, summary_tree: dict, book_anchors: str = "") -> PlotStructure:
        """从摘要树提取完整的情节结构"""
        structure = PlotStructure()
        chapters = summary_tree.get("chapter", [])
        scenes = summary_tree.get("scene", [])
        book_nodes = summary_tree.get("book", [])

        book_summary = ""
        if book_nodes:
            book_summary = book_nodes[0].get("text", "") if isinstance(book_nodes[0], dict) else str(book_nodes[0])

        structure.beats = self._extract_beats(chapters)
        structure.character_arcs = self._extract_arcs(book_summary or book_anchors, chapters)
        structure.foreshadowings = self._extract_foreshadowings(scenes)
        curve = self._extract_tension_curve(chapters)
        structure.tension_curve = [c["tension"] for c in curve]
        structure.pacing_profile = [c["pacing"] for c in curve]
        return structure

    def _extract_beats(self, chapters: list[dict]) -> list[PlotBeat]:
        beats = []
        idx = 0
        for ch in chapters:
            prompt = self.BEAT_PROMPT.format(chapter_summary=ch.get("text", "")[:500])
            resp = llm_call(self.client, self.llm_cfg, [{"role": "user", "content": prompt}],
                            temperature=0.3, max_tokens=2048)
            for item in self._parse_json(resp):
                beats.append(PlotBeat(
                    index=idx, beat_type=item.get("beat_type", "unknown"),
                    summary=item.get("summary", ""), tension_level=float(item.get("tension", 0.5)),
                    characters=item.get("characters", []), emotional_tone=item.get("tone", ""),
                ))
                idx += 1
        return beats

    def _extract_arcs(self, book_summary: str, chapters: list[dict]) -> list[CharacterArc]:
        ch_text = "\n\n".join(f"**章节{i}：**\n{ch.get('text','')[:300]}" for i, ch in enumerate(chapters))
        prompt = self.ARC_PROMPT.format(book_summary=book_summary[:500], chapter_summaries=ch_text)
        resp = llm_call(self.client, self.llm_cfg, [{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=2048)
        return [CharacterArc(
            character=i.get("character", ""), scene_index=int(i.get("scene_index", 0)),
            motivation=i.get("motivation", ""), state=i.get("state", ""),
            relationship_changes=i.get("relationship_changes", []),
        ) for i in self._parse_json(resp)]

    def _extract_foreshadowings(self, scenes: list[dict]) -> list[Foreshadowing]:
        s_text = "\n\n".join(f"**场景{i}：**\n{s.get('summary', s.get('text_preview',''))[:200]}"
                             for i, s in enumerate(scenes))
        prompt = self.FORESHADOWING_PROMPT.format(scene_summaries=s_text)
        resp = llm_call(self.client, self.llm_cfg, [{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=2048)
        return [Foreshadowing(
            setup_scene=int(i.get("setup_scene", 0)), payoff_scene=int(i.get("payoff_scene", -1)),
            element=i.get("element", ""), subtlety=float(i.get("subtlety", 0.5)),
        ) for i in self._parse_json(resp)]

    def _extract_tension_curve(self, chapters: list[dict]) -> list[dict]:
        ch_text = "\n\n".join(f"**章节{i}：**\n{ch.get('text','')[:300]}" for i, ch in enumerate(chapters))
        prompt = self.TENSION_PROMPT.format(chapter_summaries=ch_text)
        resp = llm_call(self.client, self.llm_cfg, [{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=1024)
        return [{"tension": float(i.get("tension", 0.5)), "pacing": float(i.get("pacing", 0.5))}
                for i in self._parse_json(resp)]

    @staticmethod
    def _parse_json(response: str) -> list[dict]:
        try:
            m = re.search(r'\[[\s\S]*\]', response)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    # ========== 结构化模式提取（推理用） ==========

    EXTRACT_PATTERNS_PROMPT = """以下是一部小说的情节结构分析数据（节拍、角色弧线、伏笔、张力曲线）。
请从中提取可操作的写作模式，输出为结构化 JSON。

**节拍数据：**
{beats_text}

**角色弧线：**
{arcs_text}

**伏笔数据：**
{foreshadowings_text}

**张力曲线：**
{tension_text}

请输出严格 JSON 格式（不要其他内容）：
{{
  "structure_template": {{
    "acts": [{{"chapters": "1-3", "phase": "建置", "avg_tension": 0.3}}],
    "climax_positions": [6, 10],
    "scene_transition_style": "描述转场手法"
  }},
  "character_arc_guide": {{
    "typical_transitions": ["从X到Y的典型转变"],
    "trigger_patterns": ["触发状态转变的典型事件"],
    "relationship_evolution": "关系动态如何变化"
  }},
  "pacing_rules": {{
    "tension_curve_template": [0.3, 0.5, 0.8, 0.4, 0.9],
    "dialogue_action_ratio": "对话占比描述",
    "scene_length_pattern": "场景长度模式",
    "breathing_points": "缓和节奏出现的位置模式"
  }},
  "foreshadowing_guide": {{
    "techniques": ["具体的伏笔手法1", "手法2"],
    "avg_payoff_chapters": 3,
    "density": "伏笔密度描述",
    "subtlety_pattern": "隐蔽度描述"
  }}
}}"""

    def extract_structured_patterns(self, structure: PlotStructure) -> dict:
        """从 PlotStructure 中提取结构化写作模式"""
        # 格式化分析数据
        beats_text = "\n".join(
            f"- [{b.beat_type}] 张力={b.tension_level:.1f} 角色={b.characters} 基调={b.emotional_tone}: {b.summary}"
            for b in structure.beats[:30]
        ) or "（无节拍数据）"

        arcs_text = "\n".join(
            f"- {a.character}: 动机={a.motivation} 状态={a.state} 关系变化={a.relationship_changes}"
            for a in structure.character_arcs[:20]
        ) or "（无弧线数据）"

        foreshadowings_text = "\n".join(
            f"- 场景{f.setup_scene}→场景{f.payoff_scene}: {f.element} (隐蔽度={f.subtlety:.1f})"
            for f in structure.foreshadowings[:15]
        ) or "（无伏笔数据）"

        tension_pairs = list(zip(structure.tension_curve, structure.pacing_profile))[:15]
        tension_text = "\n".join(
            f"- 章节{i}: 张力={t:.2f} 节奏={p:.2f}"
            for i, (t, p) in enumerate(tension_pairs)
        ) or "（无张力数据）"

        prompt = self.EXTRACT_PATTERNS_PROMPT.format(
            beats_text=beats_text,
            arcs_text=arcs_text,
            foreshadowings_text=foreshadowings_text,
            tension_text=tension_text,
        )

        resp = llm_call(self.client, self.llm_cfg,
                        [{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=3000)

        try:
            m = re.search(r'\{[\s\S]*\}', resp)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
        return {}
