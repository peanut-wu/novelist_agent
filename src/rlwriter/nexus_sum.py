"""NexusSum 分层历史压缩 - 模拟人类记忆遗忘曲线"""

import shutil
from pathlib import Path
from typing import Optional

from .utils import LLMConfig, get_llm_client, llm_call


class NexusSum:
    """
    分层压缩历史记忆：
    - 早期折叠：前半部分极度浓缩为宏观骨架
    - 近期保真：后四分之一保留高精度事件流

    用法：
        ns = NexusSum("output/novel_project", llm_cfg=cfg)
        ns.append_chapter(1, summary, emotional_shift="从平静到紧张")
        ns.append_chapter(2, summary)
        history = ns.get_history()  # 自动压缩
    """

    def __init__(self, project_dir: str, max_length: int = 6000,
                 llm_cfg: Optional[LLMConfig] = None,
                 compress_max_tokens: int = 3000):
        self.project_dir = Path(project_dir)
        self.memory_dir = self.project_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.memory_dir / "chapter_summaries.md"
        self.max_length = max_length
        self.llm_cfg = llm_cfg
        self.compress_max_tokens = compress_max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None and self.llm_cfg is not None:
            self._client = get_llm_client(self.llm_cfg)
        return self._client

    @property
    def history(self) -> str:
        """直接访问历史文本（支持读写）"""
        return self.get_history()

    @history.setter
    def history(self, value: str):
        """直接设置历史文本（用于从外部状态恢复）"""
        if value:
            self.history_path.write_text(value, encoding="utf-8")
        elif self.history_path.exists():
            self.history_path.unlink()

    def get_history(self) -> str:
        """获取当前压缩后的历史"""
        if not self.history_path.exists():
            return ""
        return self.history_path.read_text(encoding="utf-8")

    def append_chapter(self, chapter_num: int, summary: str,
                       emotional_shift: str = ""):
        """追加新章节到历史，超限时自动压缩"""
        current = self.get_history()
        entry = f"\n### 第 {chapter_num} 章\n- **因果流**: {summary}\n"
        if emotional_shift:
            entry += f"- **状态偏移**: {emotional_shift}\n"
        new_history = current + entry

        if len(new_history) > self.max_length:
            new_history = self._compress(new_history)

        self.history_path.write_text(new_history, encoding="utf-8")

    def _compress(self, text: str) -> str:
        """执行分层压缩。有 LLM 时调用真实压缩，否则截断。"""
        if self.client is not None:
            return self._compress_with_llm(text)
        else:
            return self._compress_fallback(text)

    def _compress_with_llm(self, text: str) -> str:
        """调用 LLM 执行分层压缩（带超时降级）"""
        prompt = self._compression_prompt(text)
        messages = [{"role": "user", "content": prompt}]

        try:
            compressed = llm_call(self.client, self.llm_cfg, messages,
                                  temperature=0.1, max_tokens=self.compress_max_tokens, timeout=60)
            if not compressed:
                raise ValueError("LLM 返回空内容")
            # 验证压缩效果：至少压缩到原长 90% 以下
            if len(compressed) < len(text) * 0.9:
                return compressed
            else:
                print(f"  ⚠️ NexusSum 压缩不够激进（{len(text)}→{len(compressed)}），降级截断")
                return self._compress_fallback(text)
        except Exception as e:
            print(f"  ⚠️ NexusSum LLM 压缩失败（{e}），降级截断")
            return self._compress_fallback(text)

    def _compress_fallback(self, text: str) -> str:
        """降级策略：截断前 25%，保留后 75%"""
        cutoff = len(text) // 4
        return (
            "\n### [早期剧情骨架 — 已截断]\n"
            "（前半部分章节已压缩，仅保留近期高精度事件流）\n\n"
            + text[cutoff:]
        )

    def _compression_prompt(self, history_text: str) -> str:
        """生成分层压缩的 LLM 指令"""
        return f"""以下剧情历史库已超过容量限制，请执行分层压缩：

**压缩规则：**
1. 【早期折叠】前半部分：极度浓缩为宏观骨架，只保留重大转折点（谁做了什么、关键转折），删除对话和细节
2. 【近期保真】后四分之一：保留详尽的事件流，包括关键对话意图、伏笔线索、角色状态变化

**输出要求：**
- 保持 Markdown 格式（### 第 X 章）
- 早期部分每章压缩到 1-2 句话
- 近期部分保持原有详细程度
- 总字数压缩到原文的 60% 以下

**待压缩历史：**
{history_text}

请直接输出压缩后的文本，不要加任何解释。"""

    def get_checkpoints_dir(self) -> Path:
        cp_dir = self.memory_dir / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        return cp_dir

    def save_checkpoint(self, chapter_num: int):
        """保存检查点快照"""
        cp_path = self.get_checkpoints_dir() / f"history_CH{chapter_num:03d}.md"
        if self.history_path.exists():
            shutil.copy2(self.history_path, cp_path)

    def rollback_to(self, chapter_num: int) -> bool:
        """回滚到指定章节的检查点"""
        cp_path = self.get_checkpoints_dir() / f"history_CH{chapter_num:03d}.md"
        if not cp_path.exists():
            return False
        shutil.copy2(cp_path, self.history_path)
        # 清理未来检查点
        for f in self.get_checkpoints_dir().iterdir():
            try:
                num = int(f.stem.split("CH")[1])
                if num > chapter_num:
                    f.unlink()
            except (ValueError, IndexError):
                continue
        return True

    def load_from_latest_checkpoint(self) -> bool:
        """从最新的 checkpoint 恢复历史内容

        Returns:
            True if restored, False if no checkpoint found
        """
        cp_dir = self.get_checkpoints_dir()
        if not cp_dir.exists():
            return False

        checkpoints = []
        for f in cp_dir.iterdir():
            if f.name.startswith("history_CH") and f.name.endswith(".md"):
                try:
                    num = int(f.stem.split("CH")[1])
                    checkpoints.append((num, f))
                except (ValueError, IndexError):
                    continue

        if not checkpoints:
            return False

        # 选择最新的 checkpoint
        checkpoints.sort(key=lambda x: x[0])
        latest_num, latest_path = checkpoints[-1]
        shutil.copy2(latest_path, self.history_path)
        print(f"NexusSum 已从 checkpoint 恢复 (CH{latest_num:03d}, {latest_path.stat().st_size} bytes)")
        return True

    def get_length(self) -> int:
        """当前历史文本长度"""
        return len(self.get_history())
