"""伏笔追踪器 - 追踪伏笔的埋设与回收"""

import json
from pathlib import Path
from typing import Optional


class ForeshadowTracker:
    """
    伏笔追踪管理器。
    - add_hook: 新增伏笔
    - resolve_hook: 标记回收
    - get_open_hooks: 获取未回收伏笔
    - save/load: 持久化
    """

    def __init__(self):
        self.hooks: list[dict] = []

    def add_hook(self, description: str, chapter: int,
                 importance: str = "medium", technique: str = "") -> dict:
        """新增伏笔"""
        hook = {
            "hook_id": f"hook_{len(self.hooks) + 1:03d}",
            "description": description,
            "chapter_introduced": chapter,
            "importance": importance,  # high / medium / low
            "technique": technique,
            "status": "open",
            "chapter_resolved": None,
        }
        self.hooks.append(hook)
        return hook

    def resolve_hook(self, keyword: str, chapter: int) -> Optional[dict]:
        """根据关键词匹配，标记伏笔回收"""
        for hook in self.hooks:
            if (hook["status"] == "open"
                    and keyword in hook["description"]
                    and hook["chapter_introduced"] < chapter):
                hook["status"] = "resolved"
                hook["chapter_resolved"] = chapter
                return hook
        return None

    def get_open_hooks(self) -> list[dict]:
        """获取所有未回收伏笔"""
        return [h for h in self.hooks if h["status"] == "open"]

    def get_hooks_for_chapter(self, chapter: int) -> list[dict]:
        """获取在指定章节之前引入且未回收的伏笔"""
        return [h for h in self.hooks
                if h["status"] == "open" and h["chapter_introduced"] < chapter]

    def format_open_hooks(self, chapter: int) -> str:
        """格式化未回收伏笔，用于注入 prompt"""
        open_hooks = self.get_hooks_for_chapter(chapter)
        if not open_hooks:
            return ""
        lines = ["**未回收伏笔（需要在后续章节中适时回收）：**"]
        for h in open_hooks:
            imp = "⭐" if h["importance"] == "high" else ""
            tech = f"（{h['technique']}）" if h["technique"] else ""
            lines.append(f"- {imp} [CH{h['chapter_introduced']}] {h['description']}{tech}")
        return "\n".join(lines)

    def save(self, path: str):
        """持久化到 JSON 文件"""
        Path(path).write_text(
            json.dumps(self.hooks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "ForeshadowTracker":
        """从 JSON 文件加载"""
        tracker = cls()
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                tracker.hooks = data
        return tracker

    def get_stats(self) -> dict:
        """统计信息"""
        total = len(self.hooks)
        open_count = len([h for h in self.hooks if h["status"] == "open"])
        resolved_count = total - open_count
        return {
            "total": total,
            "open": open_count,
            "resolved": resolved_count,
            "resolution_rate": resolved_count / total if total > 0 else 0,
        }
