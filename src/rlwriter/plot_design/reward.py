"""情节设计奖励模型：四维奖励"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..utils import LLMConfig, llm_call
from .structure import PlotStructure


@dataclass
class PlotRewardWeights:
    structure: float = 0.30
    character_arc: float = 0.25
    foreshadowing: float = 0.20
    tension_curve: float = 0.25


class PlotDesignReward:
    """四维情节奖励"""

    STRUCTURE_PROMPT = """评估【生成大纲】的情节结构是否符合【原文结构模式】。
维度：节拍分布匹配/节奏一致性/冲突升级相似/结构完整性，每项0-10分。
原文节拍：{original_beats}
生成节拍：{generated_beats}
输出 JSON：{{"节拍分布匹配":x,"节奏一致性":x,"冲突升级相似":x,"结构完整性":x}}"""

    ARC_PROMPT = """评估【生成情节】的角色弧线是否符合【原文模式】。
维度：动机变化匹配/状态转变合理/关系动态一致，每项0-10分。
原文弧线：{original_arcs}
生成弧线：{generated_arcs}
输出 JSON：{{"动机变化匹配":x,"状态转变合理":x,"关系动态一致":x}}"""

    FORESHADOWING_PROMPT = """评估【生成情节】的伏笔设计是否符合【原文模式】。
维度：铺垫密度匹配/回收节奏一致/隐蔽程度符合/元素多样性，每项0-10分。
原文伏笔：{original_foreshadowings}
生成伏笔：{generated_foreshadowings}
输出 JSON：{{"铺垫密度匹配":x,"回收节奏一致":x,"隐蔽程度符合":x,"元素多样性":x}}"""

    TENSION_PROMPT = """评估【生成情节】的张力曲线是否符合【原文模式】。
维度：曲线形态相似/峰值位置一致/动态范围匹配/节奏感吻合，每项0-10分。
原文张力：{original_tension}
生成张力：{generated_tension}
输出 JSON：{{"曲线形态相似":x,"峰值位置一致":x,"动态范围匹配":x,"节奏感吻合":x}}"""

    def __init__(self, llm_cfg: LLMConfig, weights: Optional[PlotRewardWeights] = None):
        self.llm_cfg = llm_cfg
        self.client = None
        self.weights = weights or PlotRewardWeights()

    def _get_client(self):
        if self.client is None:
            from ..utils import get_llm_client
            self.client = get_llm_client(self.llm_cfg)
        return self.client

    def _judge(self, prompt: str) -> tuple[float, dict]:
        resp = llm_call(self._get_client(), self.llm_cfg,
                        [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500)
        scores = {}
        try:
            m = re.search(r'\{[^}]+\}', resp, re.DOTALL)
            if m:
                scores = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
        return (sum(scores.values()) / (len(scores) * 10.0)) if scores else 0.5, scores

    def compute_reward(self, original: PlotStructure, generated: PlotStructure) -> dict:
        w = self.weights

        s_score, s_details = self._judge(self.STRUCTURE_PROMPT.format(
            original_beats=json.dumps(original.to_dict()["beats"], ensure_ascii=False)[:1000],
            generated_beats=json.dumps(generated.to_dict()["beats"], ensure_ascii=False)[:1000],
        ))
        a_score, a_details = self._judge(self.ARC_PROMPT.format(
            original_arcs=json.dumps(original.to_dict()["character_arcs"], ensure_ascii=False)[:1000],
            generated_arcs=json.dumps(generated.to_dict()["character_arcs"], ensure_ascii=False)[:1000],
        ))
        f_score, f_details = self._judge(self.FORESHADOWING_PROMPT.format(
            original_foreshadowings=json.dumps(original.to_dict()["foreshadowings"], ensure_ascii=False)[:1000],
            generated_foreshadowings=json.dumps(generated.to_dict()["foreshadowings"], ensure_ascii=False)[:1000],
        ))
        t_score, t_details = self._judge(self.TENSION_PROMPT.format(
            original_tension=json.dumps(original.tension_curve),
            generated_tension=json.dumps(generated.tension_curve),
        ))

        total = w.structure * s_score + w.character_arc * a_score + w.foreshadowing * f_score + w.tension_curve * t_score
        return {
            "total": round(total, 4),
            "structure": {"score": s_score, "details": s_details},
            "character_arc": {"score": a_score, "details": a_details},
            "foreshadowing": {"score": f_score, "details": f_details},
            "tension_curve": {"score": t_score, "details": t_details},
        }
