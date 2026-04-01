"""文本预处理：语义场景分割 + 风格剥离摘要"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..utils import (
    LLMConfig, SegmentationConfig, SummarizationConfig,
    get_llm_client, llm_call, llm_embed,
)

from .hierarchical import HierarchicalSummarizer, SummaryNode
from .dialogue_transform import DialogueTransformer, DialogueBlock, SceneStructure


@dataclass
class Scene:
    """一个语义场景单元"""
    index: int
    text: str
    char_count: int
    summary: Optional[str] = None
    embedding: Optional[list[float]] = None


class SceneSegmenter:
    """基于嵌入向量相似度的语义场景分割

    核心思路：
    1. 按段落初步切分
    2. 对每个段落计算嵌入向量
    3. 滑动窗口计算相邻段落的余弦相似度
    4. 相似度低于阈值处标记为场景边界
    5. 合并过短/过长的场景
    """

    def __init__(self, client, cfg: SegmentationConfig, llm_cfg: LLMConfig):
        self.client = client
        self.cfg = cfg
        self.llm_cfg = llm_cfg

    def _split_paragraphs(self, text: str) -> list[str]:
        """按段落切分，过滤空段落"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 10]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _merge_scenes(self, raw_scenes: list[Scene]) -> list[Scene]:
        """合并过短场景，拆分过长场景"""
        merged = []
        buffer = ""
        buffer_idx = 0

        for scene in raw_scenes:
            if len(buffer) + scene.char_count < self.cfg.min_scene_length:
                buffer += "\n\n" + scene.text if buffer else scene.text
            else:
                if buffer:
                    merged.append(Scene(
                        index=buffer_idx,
                        text=buffer.strip(),
                        char_count=len(buffer.strip()),
                    ))
                    buffer_idx += 1
                buffer = scene.text

        if buffer:
            merged.append(Scene(
                index=buffer_idx,
                text=buffer.strip(),
                char_count=len(buffer.strip()),
            ))

        # 拆分过长场景
        final = []
        idx = 0
        for scene in merged:
            if scene.char_count > self.cfg.max_scene_length:
                chunks = self._force_split(scene.text)
                for chunk in chunks:
                    final.append(Scene(index=idx, text=chunk, char_count=len(chunk)))
                    idx += 1
            else:
                scene.index = idx
                final.append(scene)
                idx += 1

        return final

    def _force_split(self, text: str) -> list[str]:
        """强制按长度拆分，尽量在句号处"""
        sentences = re.split(r'([。！？])', text)
        # 重新组合句子（保留标点）
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            reconstructed.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed.append(sentences[-1])

        chunks = []
        current = ""
        for s in reconstructed:
            if len(current) + len(s) > self.cfg.max_scene_length and current:
                chunks.append(current.strip())
                current = s
            else:
                current += s
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def segment(self, text: str) -> list[Scene]:
        """主入口：对全文进行语义场景分割"""
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return [Scene(0, text, len(text))]

        # 如果段落数量少，直接返回
        if len(paragraphs) <= 3:
            return [Scene(0, text, len(text))]

        # 计算嵌入向量
        embeddings = llm_embed(self.client, self.llm_cfg, paragraphs)

        # 计算相邻段落的相似度，找场景边界
        boundaries = [0]  # 始终在第一个段落开始
        for i in range(1, len(paragraphs)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.cfg.similarity_threshold:
                boundaries.append(i)
        boundaries.append(len(paragraphs))

        # 按边界构建场景
        raw_scenes = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            scene_text = "\n\n".join(paragraphs[start:end])
            raw_scenes.append(Scene(
                index=i,
                text=scene_text,
                char_count=len(scene_text),
                embedding=embeddings[start] if start < len(embeddings) else None,
            ))

        return self._merge_scenes(raw_scenes)


class StyleDecoupledSummarizer:
    """风格剥离式摘要抽取

    核心思路：
    - 用 LLM 将场景文本压缩为纯情节摘要
    - 明确要求剥离修辞、句式、词汇风格
    - 保留人物、事件、因果关系等情节骨架
    """

    STYLE_DECOUPLING_PROMPT = """你是一位专业的小说情节分析师。请对以下小说文本进行情节摘要。

**核心要求：**
1. **剥离风格**：不要保留原文的修辞手法、句式结构、特定词汇
2. **保留骨架**：记录所有关键情节点、人物行为、对话要点、环境要素
3. **客观叙述**：用平实的白话文描述，如同向他人转述故事情节
4. **标注情感**：用 [情感:xxx] 标记场景中的关键情感转折

**输出格式：**
- 场景概述（1句话）
- 关键情节点（编号列表）
- 人物动态
- 情感曲线

---

**原文：**

{text}

---

**情节摘要：**"""

    SUMMARY_EXPANSION_PROMPT = """你是一位小说续写助手。请根据以下情节摘要，续写小说正文。

**当前文风规则：**
{style_rules}

**续写要求：**
1. 严格按照情节摘要的内容展开
2. 运用上述文风规则进行创作
3. 保持叙事连贯性和节奏感
4. 不要添加摘要中没有的情节

**情节摘要：**

{summary}

---

**续写正文：**"""

    def __init__(self, client, cfg: SummarizationConfig, llm_cfg: LLMConfig):
        self.client = client
        self.cfg = cfg
        self.llm_cfg = llm_cfg

    def summarize_scene(self, scene: Scene) -> str:
        """对单个场景进行风格剥离摘要"""
        prompt = self.STYLE_DECOUPLING_PROMPT.format(text=scene.text)
        messages = [{"role": "user", "content": prompt}]
        summary = llm_call(self.client, self.llm_cfg, messages, temperature=0.3, max_tokens=2048)
        scene.summary = summary.strip()
        return scene.summary

    def expand_summary(self, summary: str, style_rules: str) -> str:
        """根据摘要 + 文风规则扩展为正文"""
        prompt = self.SUMMARY_EXPANSION_PROMPT.format(
            summary=summary,
            style_rules=style_rules,
        )
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.8, max_tokens=4096)

    def process_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """批量处理场景摘要"""
        for scene in scenes:
            self.summarize_scene(scene)
        return scenes
