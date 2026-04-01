"""分层图谱记忆架构（MemGPT 灵感 + Zettelkasten 原则）

记忆分层：
- 快速内存 (RAM)：当前上下文窗口中的文风规则
- 慢速内存 (Disk)：持久化的文风节点知识图谱
- 会话缓冲：本轮迭代的中间结果
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from ..utils import MemoryConfig, LLMConfig, get_llm_client, llm_call


@dataclass
class StyleNode:
    """文风记忆节点（Zettelkasten 卡片）"""
    node_id: str
    content: str                          # 文风规则/发现
    context_tags: list[str]               # 适用场景标签
    trigger_keywords: list[str]           # 触发关键词
    weight: float = 1.0                   # 动态权重
    scene_indices: list[int] = field(default_factory=list)  # 关联的场景编号
    created_at: float = field(default_factory=time.time)
    iteration: int = 0                    # 首次发现的迭代轮次
    reward_score: float = 0.0             # 该规则带来的奖励增益

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StyleMemory:
    """文风记忆库

    支持两种后端：
    - file: JSON 文件持久化
    - sqlite: SQLite 数据库（待实现）
    """

    def __init__(self, cfg: MemoryConfig, persistent: bool = True):
        self.cfg = cfg
        self.nodes: dict[str, StyleNode] = {}
        self._persistent = persistent
        if persistent:
            self._load()

    def _get_storage_path(self) -> Path:
        path = Path(self.cfg.path)
        path.mkdir(parents=True, exist_ok=True)
        return path / "style_nodes.json"

    def _load(self):
        path = self._get_storage_path()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.nodes = {k: StyleNode.from_dict(v) for k, v in data.items()}

    def _save(self):
        if not self._persistent:
            return
        path = self._get_storage_path()
        data = {k: v.to_dict() for k, v in self.nodes.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_node(self, node: StyleNode):
        """添加或更新文风节点"""
        self.nodes[node.node_id] = node
        self._save()

    def get_node(self, node_id: str) -> Optional[StyleNode]:
        return self.nodes.get(node_id)

    def search(self, query: str, top_k: int = 5) -> list[StyleNode]:
        """基于关键词匹配检索最相关的文风节点"""
        query_lower = query.lower()
        scored = []
        for node in self.nodes.values():
            score = 0.0
            # 关键词匹配
            for kw in node.trigger_keywords:
                if kw.lower() in query_lower:
                    score += 2.0
            # 标签匹配
            for tag in node.context_tags:
                if tag.lower() in query_lower:
                    score += 1.5
            # 内容模糊匹配
            for word in query_lower.split():
                if word in node.content.lower():
                    score += 0.5
            # 加权
            score *= node.weight
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_k]]

    def update_weights(self, node_rewards: dict[str, float], decay: float = 0.95):
        """根据本轮迭代的奖励更新节点权重

        node_rewards: {node_id: reward_delta}
        decay: 衰减因子，未参与的节点权重衰减
        """
        for node_id, node in self.nodes.items():
            if node_id in node_rewards:
                # 正向奖励提升权重，负向惩罚降低权重
                node.weight = max(0.1, node.weight + node_rewards[node_id] * 0.1)
                node.reward_score = node_rewards[node_id]
            else:
                # 未使用的节点权重衰减
                node.weight *= decay
        self._save()

    def get_context_string(self, query: str) -> str:
        """获取用于注入提示词的文风规则上下文"""
        nodes = self.search(query, self.cfg.retrieval_top_k)
        if not nodes:
            return "（暂无已学习的文风规则）"

        lines = ["**已学习的文风规则：**"]
        for i, node in enumerate(nodes, 1):
            lines.append(f"{i}. [{', '.join(node.context_tags)}] {node.content}")
            lines.append(f"   (权重: {node.weight:.2f}, 关键词: {', '.join(node.trigger_keywords)})")

        return "\n".join(lines)

    def generate_node_id(self, content: str) -> str:
        """基于内容生成唯一节点 ID"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"<StyleMemory: {len(self.nodes)} nodes>"


class MemoryManager:
    """记忆管理器：协调快速内存、慢速内存、会话缓冲"""

    def __init__(self, cfg: MemoryConfig, persistent: bool = True):
        self.cfg = cfg
        self.style_memory = StyleMemory(cfg, persistent=persistent)
        self.session_buffer: list[dict] = []  # 本轮迭代的中间发现

    def inject_context(self, scene_text: str) -> str:
        """将相关的文风规则注入当前上下文"""
        return self.style_memory.get_context_string(scene_text)

    def record_discovery(self, rule_content: str, tags: list[str],
                         keywords: list[str], scene_idx: int, reward: float):
        """记录新发现的文风规则"""
        node_id = self.style_memory.generate_node_id(rule_content)
        node = StyleNode(
            node_id=node_id,
            content=rule_content,
            context_tags=tags,
            trigger_keywords=keywords,
            scene_indices=[scene_idx],
            reward_score=reward,
        )
        self.style_memory.add_node(node)
        self.session_buffer.append({
            "type": "discovery",
            "node_id": node_id,
            "reward": reward,
            "timestamp": time.time(),
        })

    def finalize_iteration(self):
        """迭代结束时：更新权重、清空会话缓冲"""
        # 统计各节点在本轮的奖励
        node_rewards = {}
        for record in self.session_buffer:
            nid = record["node_id"]
            node_rewards[nid] = node_rewards.get(nid, 0) + record["reward"]

        self.style_memory.update_weights(node_rewards)
        self.session_buffer.clear()

    def get_stats(self) -> dict:
        """获取记忆库统计信息"""
        nodes = list(self.style_memory.nodes.values())
        return {
            "total_nodes": len(nodes),
            "avg_weight": sum(n.weight for n in nodes) / max(len(nodes), 1),
            "top_weighted": sorted(nodes, key=lambda n: n.weight, reverse=True)[:5],
        }
