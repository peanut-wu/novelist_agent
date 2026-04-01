"""测试伏笔追踪器"""

import json
import os
import tempfile
import pytest

from rlwriter.foreshadow_tracker import ForeshadowTracker


class TestForeshadowTracker:
    """ForeshadowTracker 单元测试"""

    def test_add_hook_basic(self):
        tracker = ForeshadowTracker()
        hook = tracker.add_hook("铁牌上的云雷纹", chapter=1, importance="high")
        assert hook["hook_id"] == "hook_001"
        assert hook["description"] == "铁牌上的云雷纹"
        assert hook["chapter_introduced"] == 1
        assert hook["importance"] == "high"
        assert hook["status"] == "open"
        assert hook["chapter_resolved"] is None

    def test_add_hook_default_values(self):
        tracker = ForeshadowTracker()
        hook = tracker.add_hook("神秘信件", chapter=3)
        assert hook["importance"] == "medium"
        assert hook["technique"] == ""

    def test_add_hook_with_technique(self):
        tracker = ForeshadowTracker()
        hook = tracker.add_hook("芙蓉图", chapter=1, technique="对话双关法")
        assert hook["technique"] == "对话双关法"

    def test_add_multiple_hooks(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("伏笔A", chapter=1)
        tracker.add_hook("伏笔B", chapter=2)
        tracker.add_hook("伏笔C", chapter=3)
        assert len(tracker.hooks) == 3
        assert tracker.hooks[0]["hook_id"] == "hook_001"
        assert tracker.hooks[2]["hook_id"] == "hook_003"

    def test_resolve_hook_basic(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("铁牌上的云雷纹", chapter=1, importance="high")
        result = tracker.resolve_hook("铁牌", chapter=5)
        assert result is not None
        assert result["status"] == "resolved"
        assert result["chapter_resolved"] == 5

    def test_resolve_hook_no_match(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("铁牌", chapter=1)
        result = tracker.resolve_hook("不存在的关键词", chapter=5)
        assert result is None

    def test_resolve_hook_same_chapter_rejected(self):
        """伏笔不能在同一章引入并回收"""
        tracker = ForeshadowTracker()
        tracker.add_hook("铁牌", chapter=1)
        result = tracker.resolve_hook("铁牌", chapter=1)
        assert result is None

    def test_resolve_hook_already_resolved(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("铁牌", chapter=1)
        tracker.resolve_hook("铁牌", chapter=3)
        result = tracker.resolve_hook("铁牌", chapter=5)
        assert result is None

    def test_get_open_hooks_empty(self):
        tracker = ForeshadowTracker()
        assert tracker.get_open_hooks() == []

    def test_get_open_hooks_mixed(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("伏笔A", chapter=1)
        tracker.add_hook("伏笔B", chapter=1)
        tracker.add_hook("伏笔C", chapter=2)
        tracker.resolve_hook("伏笔A", chapter=3)
        open_hooks = tracker.get_open_hooks()
        assert len(open_hooks) == 2
        assert all(h["status"] == "open" for h in open_hooks)

    def test_get_hooks_for_chapter(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("CH1伏笔", chapter=1)
        tracker.add_hook("CH3伏笔", chapter=3)
        tracker.add_hook("CH5伏笔", chapter=5)
        # 第4章时，只有CH1和CH3的伏笔可见
        hooks_ch4 = tracker.get_hooks_for_chapter(4)
        assert len(hooks_ch4) == 2
        assert hooks_ch4[0]["description"] == "CH1伏笔"

    def test_format_open_hooks_empty(self):
        tracker = ForeshadowTracker()
        assert tracker.format_open_hooks(5) == ""

    def test_format_open_hooks_with_content(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("铁牌", chapter=1, importance="high", technique="物品传递法")
        result = tracker.format_open_hooks(3)
        assert "未回收伏笔" in result
        assert "⭐" in result
        assert "CH1" in result
        assert "铁牌" in result
        assert "物品传递法" in result

    def test_format_open_hooks_no_high_importance(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("普通伏笔", chapter=1, importance="low")
        result = tracker.format_open_hooks(3)
        assert "⭐" not in result

    def test_save_and_load(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("伏笔A", chapter=1, importance="high")
        tracker.add_hook("伏笔B", chapter=2)
        tracker.resolve_hook("伏笔A", chapter=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            tracker.save(path)
            loaded = ForeshadowTracker.load(path)
            assert len(loaded.hooks) == 2
            assert loaded.hooks[0]["status"] == "resolved"
            assert loaded.hooks[0]["chapter_resolved"] == 3
            assert loaded.hooks[1]["status"] == "open"
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        tracker = ForeshadowTracker.load("/tmp/nonexistent_foreshadow.json")
        assert tracker.hooks == []

    def test_load_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                ForeshadowTracker.load(path)
        finally:
            os.unlink(path)

    def test_get_stats_empty(self):
        tracker = ForeshadowTracker()
        stats = tracker.get_stats()
        assert stats == {"total": 0, "open": 0, "resolved": 0, "resolution_rate": 0}

    def test_get_stats_partial(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("A", chapter=1)
        tracker.add_hook("B", chapter=1)
        tracker.add_hook("C", chapter=1)
        tracker.resolve_hook("A", chapter=3)
        stats = tracker.get_stats()
        assert stats["total"] == 3
        assert stats["open"] == 2
        assert stats["resolved"] == 1
        assert abs(stats["resolution_rate"] - 1 / 3) < 0.001

    def test_get_stats_all_resolved(self):
        tracker = ForeshadowTracker()
        tracker.add_hook("A", chapter=1)
        tracker.resolve_hook("A", chapter=2)
        stats = tracker.get_stats()
        assert stats["resolution_rate"] == 1.0

    def test_keyword_partial_match(self):
        """关键词匹配是子串匹配"""
        tracker = ForeshadowTracker()
        tracker.add_hook("集贤殿书的鲜红官印", chapter=1)
        result = tracker.resolve_hook("集贤殿", chapter=5)
        assert result is not None

    def test_resolve_hook_cannot_resolve_future_hook(self):
        """不能回收在当前章节之后才引入的伏笔"""
        tracker = ForeshadowTracker()
        tracker.add_hook("CH5伏笔", chapter=5)
        result = tracker.resolve_hook("伏笔", chapter=3)
        assert result is None  # chapter_introduced(5) >= chapter(3) 被过滤
