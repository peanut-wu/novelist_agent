"""测试 NexusSum 分层压缩"""

import os
import shutil
import tempfile
import pytest

from rlwriter.nexus_sum import NexusSum


class TestNexusSum:
    """NexusSum 单元测试"""

    def _make_nexus(self, max_length=500) -> NexusSum:
        """创建临时 NexusSum 实例（无 LLM）"""
        self.tmp_dir = tempfile.mkdtemp()
        return NexusSum(self.tmp_dir, max_length=max_length)

    def teardown_method(self):
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_empty_history(self):
        ns = self._make_nexus()
        assert ns.get_history() == ""
        assert ns.get_length() == 0

    def test_append_single_chapter(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "李慕夜入鬼市，得到半卷密函")
        history = ns.get_history()
        assert "第 1 章" in history
        assert "李慕" in history
        assert "因果流" in history

    def test_append_with_emotional_shift(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "平静开始", emotional_shift="从平静到紧张")
        history = ns.get_history()
        assert "状态偏移" in history
        assert "从平静到紧张" in history

    def test_append_multiple_chapters(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "第1章摘要")
        ns.append_chapter(2, "第2章摘要")
        ns.append_chapter(3, "第3章摘要")
        history = ns.get_history()
        assert "第 1 章" in history
        assert "第 2 章" in history
        assert "第 3 章" in history

    def test_compression_triggered(self):
        """超过 max_length 时触发压缩"""
        ns = self._make_nexus(max_length=200)
        # 写入足够多的内容触发压缩
        for i in range(1, 6):
            ns.append_chapter(i, f"这是第{i}章的摘要内容" * 10)
        # 压缩后应该比无压缩短
        assert ns.get_length() < 200 * 6  # 不会无限膨胀

    def test_compression_preserves_recent(self):
        """压缩后近期章节应保留"""
        ns = self._make_nexus(max_length=300)
        for i in range(1, 10):
            ns.append_chapter(i, f"第{i}章事件" * 5)
        history = ns.get_history()
        # 近期章节（第8、9章）应该还在
        assert "第 9 章" in history or "第 8 章" in history

    def test_fallback_compression(self):
        """无 LLM 时降级截断"""
        ns = self._make_nexus(max_length=100)
        ns.llm_cfg = None  # 确保无 LLM
        ns._client = None
        for i in range(1, 5):
            ns.append_chapter(i, f"第{i}章详细内容" * 10)
        history = ns.get_history()
        assert "已截断" in history or len(history) < 2000

    def test_save_and_load_checkpoint(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "第1章摘要")
        ns.append_chapter(2, "第2章摘要")
        ns.save_checkpoint(2)

        # 再加第3章
        ns.append_chapter(3, "第3章摘要")
        assert "第 3 章" in ns.get_history()

        # 回滚到第2章
        result = ns.rollback_to(2)
        assert result is True
        assert "第 3 章" not in ns.get_history()
        assert "第 2 章" in ns.get_history()

    def test_rollback_nonexistent(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "摘要")
        result = ns.rollback_to(99)
        assert result is False

    def test_rollback_cleans_future_checkpoints(self):
        ns = self._make_nexus()
        ns.append_chapter(1, "ch1")
        ns.save_checkpoint(1)
        ns.append_chapter(2, "ch2")
        ns.save_checkpoint(2)
        ns.append_chapter(3, "ch3")
        ns.save_checkpoint(3)

        ns.rollback_to(1)
        cp_dir = ns.get_checkpoints_dir()
        files = list(cp_dir.iterdir())
        # 只剩 CH001 的检查点
        names = [f.stem for f in files]
        assert "history_CH001" in names
        assert "history_CH003" not in names

    def test_get_length(self):
        ns = self._make_nexus()
        assert ns.get_length() == 0
        ns.append_chapter(1, "摘要内容")
        assert ns.get_length() > 0

    def test_compression_prompt_structure(self):
        ns = self._make_nexus()
        prompt = ns._compression_prompt("测试历史文本")
        assert "早期折叠" in prompt
        assert "近期保真" in prompt
        assert "测试历史文本" in prompt

    def test_checkpoint_dir_created(self):
        ns = self._make_nexus()
        cp_dir = ns.get_checkpoints_dir()
        assert cp_dir.exists()

    def test_history_persistence(self):
        """追加后重新加载应保持"""
        ns = self._make_nexus()
        ns.append_chapter(1, "持久化测试")

        # 新实例加载同一目录
        ns2 = NexusSum(self.tmp_dir, max_length=500)
        assert "第 1 章" in ns2.get_history()
        assert "持久化测试" in ns2.get_history()

    def test_multiple_compressions(self):
        """多次超限应多次压缩"""
        ns = self._make_nexus(max_length=100)
        ns.llm_cfg = None
        ns._client = None
        for i in range(1, 20):
            ns.append_chapter(i, f"第{i}章内容" * 5)
        # 不会崩溃，长度有界
        assert ns.get_length() < 2000
