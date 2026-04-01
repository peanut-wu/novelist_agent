"""测试阶段4：角色冷热分离 + 章节后处理"""

import pytest
from rlwriter.series.utils import format_active_characters, clean_chapter_output


class TestFormatActiveCharacters:
    """角色冷热格式化测试"""

    def test_empty_state(self):
        assert format_active_characters({"active": {}, "inactive": {}}) == ""

    def test_no_active(self):
        assert format_active_characters({}) == ""

    def test_single_character(self):
        state = {
            "active": {
                "李慕": {
                    "location": "延康坊",
                    "key_events": ["被金吾卫抓捕", "审讯脱身", "获赠银两"],
                }
            },
            "inactive": {
                "胡商": {"name": "胡商", "last_seen": "CH1 鬼市"}
            }
        }
        result = format_active_characters(state)
        assert "活跃角色状态" in result
        assert "李慕" in result
        assert "延康坊" in result
        assert "被金吾卫抓捕" in result

    def test_multiple_characters(self):
        state = {
            "active": {
                "李慕": {"location": "金吾狱", "key_events": ["入狱"]},
                "高延福": {"location": "金吾狱", "key_events": ["审讯"]},
            }
        }
        result = format_active_characters(state)
        assert "李慕" in result
        assert "高延福" in result

    def test_max_2_events(self):
        state = {
            "active": {
                "李慕": {
                    "location": "长安",
                    "key_events": ["事件1", "事件2", "事件3", "事件4", "事件5"],
                }
            }
        }
        result = format_active_characters(state)
        assert "事件1" in result
        assert "事件2" in result
        assert "事件3" not in result

    def test_no_key_events(self):
        state = {
            "active": {
                "路人甲": {"location": "街边", "key_events": []}
            }
        }
        result = format_active_characters(state)
        assert "路人甲" in result
        assert "无关键事件" in result


class TestCleanChapterOutput:
    """章节后处理测试"""

    def test_no_meta_text(self):
        text = "这是正文内容。\n李慕推开门。"
        assert clean_chapter_output(text) == text

    def test_removes_key_points_marker(self):
        text = "正文结束。\n\n---\n\n**本章关键点收束**：\n1. 测试"
        result = clean_chapter_output(text)
        assert "正文结束" in result
        assert "关键点" not in result

    def test_removes_narrative_rhythm(self):
        text = "最后一段话。\n---\n\n**叙事节奏**：张弛有度"
        result = clean_chapter_output(text)
        assert "最后一段话" in result
        assert "叙事节奏" not in result

    def test_removes_style_section(self):
        text = "李慕走了。\n---\n\n**文风贯彻**：动作简洁"
        result = clean_chapter_output(text)
        assert "李慕走了" in result
        assert "文风贯彻" not in result

    def test_removes_cn_end_marker(self):
        text = "长安城的灯火渐亮。\n\n（本章完）"
        result = clean_chapter_output(text)
        assert result.endswith("长安城的灯火渐亮。")
        assert "本章完" not in result

    def test_removes_en_end_marker(self):
        text = "故事继续。\n\n(本章完)"
        result = clean_chapter_output(text)
        assert result.endswith("故事继续。")

    def test_preserves_legitimate_text(self):
        text = "李慕想起母亲信中那句话：今冬定能凑足银钱。"
        result = clean_chapter_output(text)
        assert result == text

    def test_multiple_markers_first_wins(self):
        """多个标记时，第一个生效"""
        text = "段落1。\n---\n\n**本章关键点**：分析1\n\n**文风贯彻**：分析2"
        result = clean_chapter_output(text)
        assert "段落1" in result
        assert "关键点" not in result
        assert "文风贯彻" not in result

    def test_empty_text(self):
        assert clean_chapter_output("") == ""

    def test_strips_whitespace(self):
        text = "正文。\n\n   "
        result = clean_chapter_output(text)
        assert result == "正文。"
