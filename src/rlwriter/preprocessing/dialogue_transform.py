"""多智能体对话转结构：将复杂对话降维为客观情节描述

解决风格剥离中的一个难点：
- 如果摘要保留了原文对话的修辞风格，后续生成退化为复写
- 需要将"对话驱动"转为"叙述驱动"，消除修辞模糊性
"""

from dataclasses import dataclass
from typing import Optional

from ..utils import LLMConfig, get_llm_client, llm_call


@dataclass
class DialogueBlock:
    """对话块"""
    speaker: str
    content: str
    emotion: str = ""        # 推断的情感
    action_intent: str = ""  # 行为意图


@dataclass
class SceneStructure:
    """场景结构化描述"""
    setting: str             # 环境/场景设定
    characters: list[str]    # 出场人物
    plot_points: list[str]   # 情节要点
    emotions: list[str]      # 情感曲线
    dialogue_intent: list[DialogueBlock] = None  # 对话意图（非原文对话）


class DialogueTransformer:
    """对话 → 结构化描述转换器"""

    TRANSFORM_PROMPT = """你是一位小说情节分析师。请将以下小说片段转换为结构化的情节描述。

**核心要求：**
1. **不要保留对话原文**，而是将每段对话转换为"谁对谁说了什么意图的话"
2. **剥离修辞手法**，用平实语言描述事件
3. **保留因果关系**：谁做了什么→导致什么→影响了谁
4. **提取情感线索**：用 [情感:xxx] 标注关键情感转折

**原文：**

{text}

---

**结构化描述：**

场景设定：[环境、时间、氛围]
出场人物：[人物列表]
情节要点：
1. [事件1]
2. [事件2]
...
情感曲线：[情感1] → [情感2] → ...
对话意图：
- [人物A]对[人物B]表达了[意图]，[人物B]回应了[意图]"""

    def __init__(self, client, llm_cfg: LLMConfig):
        self.client = client
        self.llm_cfg = llm_cfg

    def transform(self, scene_text: str) -> str:
        """将对话驱动的场景转为结构化描述"""
        prompt = self.TRANSFORM_PROMPT.format(text=scene_text)
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.3, max_tokens=2048)

    def batch_transform(self, scenes: list[str]) -> list[str]:
        """批量转换"""
        return [self.transform(scene) for scene in scenes]
