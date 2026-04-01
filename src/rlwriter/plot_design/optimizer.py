"""情节设计提示词优化器"""

import json
import re
import time
from dataclasses import dataclass, field

from ..utils import LLMConfig, llm_call
from .structure import PlotStructure


@dataclass
class PlotPromptState:
    structure_template: str = ""
    character_arc_guide: str = ""
    pacing_rules: str = ""
    foreshadowing_guide: str = ""
    iteration: int = 0
    history: list[dict] = field(default_factory=list)

    def to_prompt_components(self) -> dict:
        return {
            "structure_template": self.structure_template,
            "character_arc_guide": self.character_arc_guide,
            "pacing_rules": self.pacing_rules,
            "foreshadowing_guide": self.foreshadowing_guide,
        }

    def render(self) -> str:
        parts = []
        if self.structure_template:
            parts.append(f"**情节结构模板：**\n{self.structure_template}")
        if self.character_arc_guide:
            parts.append(f"\n**角色弧线指引：**\n{self.character_arc_guide}")
        if self.pacing_rules:
            parts.append(f"\n**节奏控制规则：**\n{self.pacing_rules}")
        if self.foreshadowing_guide:
            parts.append(f"\n**伏笔设计指引：**\n{self.foreshadowing_guide}")
        return "\n\n".join(parts) if parts else "（待学习）"


INITIAL_STRUCTURE = "按目标作品的节拍分布模式设计情节，匹配冲突升级曲线，保持结构完整性。"
INITIAL_ARC = "每个主要角色有清晰的动机变化轨迹，状态转变需有合理触发事件，关系动态有起伏。"
INITIAL_PACING = "匹配目标作品的张力曲线形态，控制紧张/缓和交替频率，信息密度一致。"
INITIAL_FORESHADOWING = "铺垫密度匹配目标作品，回收时间间隔合理，隐蔽程度符合原文风格。"


class PlotDesignOptimizer:
    """情节设计提示词优化器"""

    OUTLINE_PROMPT = """根据以下情节设计规则和章节摘要，生成完整情节大纲。
当前情节设计规则：
{plot_prompt}
全书摘要：{book_summary}
章节摘要：{chapter_summaries}
输出 JSON：{{"book_summary":"一句话","chapters":["章节1大纲","章节2大纲",...],"character_arcs":["角色A:从X到Y"],"foreshadowing_plan":["伏笔1"]}}"""

    GRADIENT_PROMPT = """对比【原文情节结构】和【生成情节结构】，分析差异并给出改进建议。
原文结构：{original_structure}
生成结构：{generated_structure}
奖励评分：{reward_details}
输出格式：差异点1: [描述] → 建议: [改进方法]"""

    OPTIMIZE_PROMPT = """根据文本梯度优化情节设计提示词。
当前结构模板：{structure_template}
当前弧线指引：{character_arc_guide}
当前节奏规则：{pacing_rules}
当前伏笔指引：{foreshadowing_guide}
文本梯度：{text_gradient}
输出 JSON：{{"structure_template":"优化后","character_arc_guide":"优化后","pacing_rules":"优化后","foreshadowing_guide":"优化后"}}"""

    def __init__(self, llm_cfg: LLMConfig, convergence_threshold: float = 0.02):
        self.llm_cfg = llm_cfg
        self.client = None
        self.convergence_threshold = convergence_threshold
        self.state = PlotPromptState(
            structure_template=INITIAL_STRUCTURE,
            character_arc_guide=INITIAL_ARC,
            pacing_rules=INITIAL_PACING,
            foreshadowing_guide=INITIAL_FORESHADOWING,
        )

    def _get_client(self):
        if self.client is None:
            from ..utils import get_llm_client
            self.client = get_llm_client(self.llm_cfg)
        return self.client

    def generate_outline(self, summary_tree: dict) -> dict:
        book_summary = ""
        book_nodes = summary_tree.get("book", [])
        if book_nodes:
            book_summary = book_nodes[0].get("text", "") if isinstance(book_nodes[0], dict) else str(book_nodes[0])
        chapters = summary_tree.get("chapter", [])
        ch_text = "\n\n".join(f"**章节{i}：**\n{ch.get('text','')[:300]}" for i, ch in enumerate(chapters))

        prompt = self.OUTLINE_PROMPT.format(
            plot_prompt=self.state.render(),
            book_summary=book_summary[:500] or "（无）",
            chapter_summaries=ch_text or "（无）",
        )
        resp = llm_call(self._get_client(), self.llm_cfg,
                        [{"role": "user", "content": prompt}], temperature=0.7, max_tokens=4096)
        try:
            m = re.search(r'\{[\s\S]*\}', resp)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
        return {"book_summary": "", "chapters": [], "character_arcs": [], "foreshadowing_plan": []}

    def compute_gradient(self, original: PlotStructure, generated: PlotStructure, reward: dict) -> str:
        prompt = self.GRADIENT_PROMPT.format(
            original_structure=json.dumps(original.to_dict(), ensure_ascii=False)[:1500],
            generated_structure=json.dumps(generated.to_dict(), ensure_ascii=False)[:1500],
            reward_details=json.dumps(reward, ensure_ascii=False),
        )
        return llm_call(self._get_client(), self.llm_cfg,
                        [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1000)

    def optimize(self, gradient: str):
        prompt = self.OPTIMIZE_PROMPT.format(
            structure_template=self.state.structure_template,
            character_arc_guide=self.state.character_arc_guide,
            pacing_rules=self.state.pacing_rules,
            foreshadowing_guide=self.state.foreshadowing_guide,
            text_gradient=gradient,
        )
        resp = llm_call(self._get_client(), self.llm_cfg,
                        [{"role": "user", "content": prompt}], temperature=0.4, max_tokens=2000)
        try:
            m = re.search(r'\{[\s\S]*\}', resp)
            if m:
                data = json.loads(m.group())
                self.state.structure_template = data.get("structure_template", self.state.structure_template)
                self.state.character_arc_guide = data.get("character_arc_guide", self.state.character_arc_guide)
                self.state.pacing_rules = data.get("pacing_rules", self.state.pacing_rules)
                self.state.foreshadowing_guide = data.get("foreshadowing_guide", self.state.foreshadowing_guide)
        except (json.JSONDecodeError, ValueError):
            pass
        self.state.iteration += 1

    def record(self, reward: float, gradient: str):
        self.state.history.append({"iteration": self.state.iteration, "reward": reward,
                                    "gradient": gradient[:200], "timestamp": time.time()})

    def is_converged(self) -> bool:
        if len(self.state.history) < 2:
            return False
        return abs(self.state.history[-1]["reward"] - self.state.history[-2]["reward"]) < self.convergence_threshold
