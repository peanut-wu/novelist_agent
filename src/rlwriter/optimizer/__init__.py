"""基于文本梯度的提示词优化器（APO）

核心机制：
- LLM 同时作为生成器、评估裁判、提示词优化器
- 自然语言反馈作为"文本梯度"
- 在上下文窗口内完成纯文本空间的 RL 闭环
- 无参数更新，仅修改提示词组件
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..utils import OptimizerConfig, LLMConfig, get_llm_client, llm_call
from ..memory import MemoryManager


@dataclass
class PromptState:
    """当前提示词状态"""
    system_instruction: str = ""    # 系统指令
    style_rules: str = ""           # 文风规则列表
    examples: str = ""              # 示例演示
    iteration: int = 0
    history: list[dict] = field(default_factory=list)  # 历史奖励记录

    def to_prompt_components(self) -> dict:
        return {
            "system_instruction": self.system_instruction,
            "style_rules": self.style_rules,
            "examples": self.examples,
        }

    def render_system_prompt(self) -> str:
        """渲染为完整的系统提示词"""
        parts = []
        if self.system_instruction:
            parts.append(self.system_instruction)
        if self.style_rules:
            parts.append(f"\n**文风规则：**\n{self.style_rules}")
        if self.examples:
            parts.append(f"\n**参考示例：**\n{self.examples}")
        return "\n".join(parts)


class TextGradientOptimizer:
    """文本梯度优化器

    优化流程：
    1. 用当前提示词生成文本
    2. 奖励模型评分
    3. LLM 分析"损失"（原文 vs 生成文本的差异）
    4. LLM 生成"梯度"（自然语言反馈）
    5. LLM 根据梯度修改提示词组件
    """

    GRADIENT_PROMPT = """你是一位文风分析专家。请对比【原文】和【续写文本】，分析续写在文风模仿上的具体不足。

**分析要求：**
1. 找出 3-5 个最显著的文风差异点
2. 每个差异点用一句话描述
3. 对每个差异点，给出具体的改进建议

**【原文】**
{original}

**【续写文本】**
{generated}

**【当前奖励评分】**
{reward_details}

**【输出格式】**
差异点1: [描述] → 建议: [具体改进方法]
差异点2: [描述] → 建议: [具体改进方法]
..."""

    OPTIMIZE_PROMPT = """你是一位提示词工程专家。请根据【文本梯度】（文风差异分析）来优化【当前提示词组件】。

**优化规则：**
1. 保留当前提示词中有效的部分
2. 根据梯度建议添加新的文风规则
3. 删除或修改效果不佳的规则
4. 确保提示词清晰、具体、可操作
5. 文风规则不超过 10 条

**【当前系统指令】**
{system_instruction}

**【当前文风规则】**
{style_rules}

**【当前示例】**
{examples}

**【文本梯度（差异分析与建议）】**
{text_gradient}

**【已学习的文风记忆】**
{memory_context}

**【输出格式】**
请用 JSON 格式输出优化后的组件：
{{"system_instruction": "优化后的系统指令", "style_rules": "优化后的文风规则", "examples": "优化后的示例"}}"""

    INITIAL_SYSTEM_INSTRUCTION = """你是一位专业的小说续写助手。你的任务是根据情节摘要，用指定的文风续写小说正文。

核心要求：
1. 严格遵循给定的文风规则
2. 保持叙事连贯性和节奏感
3. 仅基于摘要内容展开，不添加额外情节
4. 模拟目标作者的写作习惯"""

    def __init__(self, cfg: OptimizerConfig, llm_cfg: LLMConfig, memory: MemoryManager):
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.client = get_llm_client(llm_cfg)
        self.memory = memory
        self.state = PromptState(
            system_instruction=self.INITIAL_SYSTEM_INSTRUCTION,
            style_rules="（待学习）",
            examples="（待补充）",
        )

    def generate_text(self, summary: str) -> str:
        """用当前提示词根据摘要生成正文"""
        system_prompt = self.state.render_system_prompt()
        memory_ctx = self.memory.inject_context(summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请根据以下情节摘要续写小说正文：\n\n{summary}\n\n{memory_ctx}"},
        ]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.8, max_tokens=4096)

    def compute_gradient(self, original: str, generated: str, reward_details: dict) -> str:
        """计算文本梯度（自然语言差异分析）"""
        prompt = self.GRADIENT_PROMPT.format(
            original=original[:1000],
            generated=generated[:1000],
            reward_details=json.dumps(reward_details, ensure_ascii=False, indent=2),
        )
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.3, max_tokens=1000)

    def optimize_prompt(self, text_gradient: str) -> PromptState:
        """根据文本梯度优化提示词组件"""
        memory_ctx = self.memory.style_memory.get_context_string(text_gradient)

        prompt = self.OPTIMIZE_PROMPT.format(
            system_instruction=self.state.system_instruction,
            style_rules=self.state.style_rules,
            examples=self.state.examples,
            text_gradient=text_gradient,
            memory_context=memory_ctx,
        )
        messages = [{"role": "user", "content": prompt}]
        response = llm_call(self.client, self.llm_cfg, messages, temperature=0.4, max_tokens=2000)

        # 解析 JSON
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                self.state.system_instruction = data.get("system_instruction", self.state.system_instruction)
                self.state.style_rules = data.get("style_rules", self.state.style_rules)
                self.state.examples = data.get("examples", self.state.examples)
        except (json.JSONDecodeError, ValueError):
            pass  # 解析失败，保持原状

        self.state.iteration += 1
        return self.state

    def record_iteration(self, reward: float, reward_details: dict, gradient: str):
        """记录本轮迭代结果"""
        self.state.history.append({
            "iteration": self.state.iteration,
            "reward": reward,
            "reward_details": reward_details,
            "gradient": gradient,
            "timestamp": time.time(),
        })

    def is_converged(self) -> bool:
        """检查是否收敛"""
        if len(self.state.history) < 2:
            return False
        last_two = [h["reward"] for h in self.state.history[-2:]]
        return abs(last_two[1] - last_two[0]) < self.cfg.convergence_threshold

    def get_prompt_state(self) -> PromptState:
        return self.state
