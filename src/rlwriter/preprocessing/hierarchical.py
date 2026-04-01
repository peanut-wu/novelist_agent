"""递归层次化合并：将底层场景摘要聚合为章节摘要、卷宗摘要

解决两个问题：
1. 长篇小说的全书尺度情节锚定（防情节幻觉）
2. 多粒度摘要供不同生成粒度使用
"""

from dataclasses import dataclass, field
from typing import Optional

from ..utils import LLMConfig, get_llm_client, llm_call


@dataclass
class SummaryNode:
    """摘要树节点"""
    level: str           # scene / chapter / volume / book
    index: int
    text: str            # 该层级的摘要内容
    children: list[int] = field(default_factory=list)  # 子节点索引
    char_count: int = 0


class HierarchicalSummarizer:
    """递归层次化合并器

    层级结构：
    - scene:   场景级摘要（已有）
    - chapter: 章节级摘要（合并 N 个场景）
    - volume:  卷宗级摘要（合并 N 个章节）
    - book:    全书级摘要（合并所有卷宗）
    """

    MERGE_PROMPT = """你是一位小说情节分析师。请将以下 {count} 段场景摘要合并为一段更高层级的摘要。

**要求：**
1. 合并为连贯的整体叙事，不是简单拼接
2. 保留所有关键情节点和人物发展
3. 剥离具体场景细节，只保留宏观情节走向
4. 如有情感主线变化，标注出来
5. 控制在 {max_words} 字以内

**场景摘要：**

{summaries}

---

**合并摘要："""

    BOOK_ANCHOR_PROMPT = """你是一位小说大纲设计师。请基于以下全书摘要，提取核心情节锚点。

**要求：**
1. 提取 5-10 个不可更改的核心情节节点
2. 每个节点包含：事件描述、涉及人物、情感基调
3. 这些锚点将用于约束后续生成，防止情节幻觉

**全书摘要：**

{book_summary}

---

**核心情节锚点："""

    def __init__(self, client, llm_cfg: LLMConfig,
                 scenes_per_chapter: int = 5,
                 chapters_per_volume: int = 8):
        self.client = client
        self.llm_cfg = llm_cfg
        self.scenes_per_chapter = scenes_per_chapter
        self.chapters_per_volume = chapters_per_volume
        self.summary_tree: dict[str, list[SummaryNode]] = {
            "scene": [],
            "chapter": [],
            "volume": [],
            "book": [],
        }

    def build_tree(self, scene_summaries: list[str]) -> dict[str, list[SummaryNode]]:
        """从场景摘要构建完整的摘要树"""
        # Scene level
        for i, summary in enumerate(scene_summaries):
            self.summary_tree["scene"].append(SummaryNode(
                level="scene", index=i, text=summary, char_count=len(summary)
            ))

        # Chapter level
        chapters = self._merge_level(
            self.summary_tree["scene"],
            self.scenes_per_chapter,
            "chapter",
            max_words=500
        )
        self.summary_tree["chapter"] = chapters

        # Volume level
        volumes = self._merge_level(
            self.summary_tree["chapter"],
            self.chapters_per_volume,
            "volume",
            max_words=800
        )
        self.summary_tree["volume"] = volumes

        # Book level (single node)
        if volumes:
            all_volume_text = "\n\n".join(v.text for v in volumes)
            book_summary = self._merge_single(all_volume_text, max_words=1000)
            self.summary_tree["book"] = [SummaryNode(
                level="book", index=0, text=book_summary,
                children=[v.index for v in volumes],
                char_count=len(book_summary)
            )]

        return self.summary_tree

    def _merge_level(self, child_nodes: list[SummaryNode],
                     group_size: int, level_name: str,
                     max_words: int = 500) -> list[SummaryNode]:
        """将子节点按组合并为上层节点"""
        merged = []
        for i in range(0, len(child_nodes), group_size):
            group = child_nodes[i:i + group_size]
            if not group:
                continue

            summaries_text = "\n\n---\n\n".join(
                f"**场景 {node.index}：**\n{node.text}" for node in group
            )

            merged_text = self._merge_single(summaries_text, max_words)
            merged.append(SummaryNode(
                level=level_name,
                index=len(merged),
                text=merged_text,
                children=[node.index for node in group],
                char_count=len(merged_text),
            ))

        return merged

    def _merge_single(self, summaries_text: str, max_words: int = 500) -> str:
        """调用 LLM 合并摘要"""
        count = summaries_text.count("**场景") or summaries_text.count("**章节") or "多段"
        prompt = self.MERGE_PROMPT.format(
            count=count,
            max_words=max_words,
            summaries=summaries_text,
        )
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.3, max_tokens=2048)

    def get_book_anchors(self) -> str:
        """提取全书核心情节锚点（防幻觉）"""
        book_nodes = self.summary_tree.get("book", [])
        if not book_nodes:
            return ""

        prompt = self.BOOK_ANCHOR_PROMPT.format(book_summary=book_nodes[0].text)
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.llm_cfg, messages, temperature=0.3, max_tokens=2048)

    def get_context_for_scene(self, scene_index: int) -> str:
        """为指定场景获取多层级上下文"""
        parts = []

        # 全书锚点
        book_nodes = self.summary_tree.get("book", [])
        if book_nodes:
            parts.append(f"**全书主线：**\n{book_nodes[0].text[:300]}")

        # 所属章节摘要
        for ch in self.summary_tree.get("chapter", []):
            if scene_index in ch.children:
                parts.append(f"**当前章节摘要：**\n{ch.text}")
                break

        # 前一个场景的摘要（连贯性）
        scenes = self.summary_tree.get("scene", [])
        if scene_index > 0 and scene_index - 1 < len(scenes):
            parts.append(f"**前一场景：**\n{scenes[scene_index - 1].text[:200]}")

        return "\n\n".join(parts)
