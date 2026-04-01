"""测试：角色状态更新 - JSON 截断修复 + 超长摘要"""

import json
import pytest
from unittest.mock import MagicMock, patch

from rlwriter.utils import LLMConfig
from rlwriter.series.utils import update_character_state


def _mock_client():
    return MagicMock()


def _llm_cfg():
    return LLMConfig()


class TestUpdateCharacterState:
    """角色状态更新测试"""

    def test_normal_json(self):
        """正常 JSON 返回"""
        response = json.dumps({
            "active": {"李慕": {"location": "长安", "key_events": ["入狱"], "possessions": [], "relationships_update": {}}},
            "inactive": {}
        }, ensure_ascii=False)

        with patch("rlwriter.series.utils.llm_call", return_value=response):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "短文本", {"active": {}, "inactive": {}}
            )
        assert "李慕" in result["active"]
        assert result["active"]["李慕"]["location"] == "长安"

    def test_truncated_json_completes(self):
        """截断 JSON 自动补全闭合括号"""
        # 模拟 LLM 输出被截断：缺少结尾的 }}})
        truncated = '{"active": {"李慕": {"location": "长安", "key_events": ["入狱"], "possessions": [], "relationships_update": {}'
        # 等价于缺失 3 个 }

        with patch("rlwriter.series.utils.llm_call", return_value=truncated):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", {"active": {}, "inactive": {}}
            )
        assert "李慕" in result["active"]
        assert result["active"]["李慕"]["location"] == "长安"

    def test_truncated_nested_json(self):
        """多层嵌套截断也能补全"""
        truncated = '{"active": {"李慕": {"location": "长安", "key_events": ["a", "b"], "possessions": [], "relationships_update": {"张三": "好友"}'
        # 缺失 3 个 }

        with patch("rlwriter.series.utils.llm_call", return_value=truncated):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", {"active": {}, "inactive": {}}
            )
        assert result["active"]["李慕"]["relationships_update"]["张三"] == "好友"

    def test_invalid_json_fallback(self):
        """彻底无效的 JSON，返回原状态"""
        with patch("rlwriter.series.utils.llm_call", return_value="不是JSON"):
            original = {"active": {"X": {"location": "Y"}}, "inactive": {}}
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", original
            )
        assert result == original

    def test_extra_text_around_json(self):
        """LLM 返回带额外文本的 JSON"""
        response = '好的，这是更新后的状态：\n{"active": {"李慕": {"location": "鬼市", "key_events": [], "possessions": [], "relationships_update": {}}}, "inactive": {}}\n希望对你有帮助。'

        with patch("rlwriter.series.utils.llm_call", return_value=response):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", {"active": {}, "inactive": {}}
            )
        assert result["active"]["李慕"]["location"] == "鬼市"

    def test_missing_active_inactive_returns_original(self):
        """返回的 JSON 没有 active/inactive 字段"""
        response = '{"其他字段": "值"}'

        with patch("rlwriter.series.utils.llm_call", return_value=response):
            original = {"active": {}, "inactive": {"胡商": {"name": "胡商"}}}
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", original
            )
        assert result == original

    def test_long_text_triggers_summary(self):
        """超 30000 字时触发摘要"""
        long_text = "章节内容" * 10000  # 40000 字符
        summary_result = "摘要后的文本"

        def mock_call(client, cfg, messages, **kwargs):
            prompt = messages[0]["content"]
            if "概括" in prompt:
                return summary_result
            return json.dumps({"active": {}, "inactive": {}}, ensure_ascii=False)

        call_count = {"count": 0}

        def counting_call(client, cfg, messages, **kwargs):
            call_count["count"] += 1
            return mock_call(client, cfg, messages, **kwargs)

        with patch("rlwriter.series.utils.llm_call", side_effect=counting_call):
            update_character_state(
                _mock_client(), _llm_cfg(), 1, long_text, {"active": {}, "inactive": {}}
            )
        # 应该调用 2 次：1 次摘要 + 1 次状态更新
        assert call_count["count"] == 2

    def test_long_text_summary_fallback(self):
        """超长文本摘要失败时 fallback 到截尾"""
        long_text = "X" * 40000

        def fail_then_succeed(client, cfg, messages, **kwargs):
            prompt = messages[0]["content"]
            if "概括" in prompt:
                raise Exception("摘要失败")
            # 状态更新
            return json.dumps({"active": {}, "inactive": {}}, ensure_ascii=False)

        with patch("rlwriter.series.utils.llm_call", side_effect=fail_then_succeed):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, long_text, {"active": {}, "inactive": {}}
            )
        # fallback 后仍能正常返回
        assert "active" in result

    def test_short_text_no_summary(self):
        """短文本不触发摘要"""
        short_text = "短" * 100

        with patch("rlwriter.series.utils.llm_call",
                   return_value=json.dumps({"active": {}, "inactive": {}})) as mock:
            update_character_state(
                _mock_client(), _llm_cfg(), 1, short_text, {"active": {}, "inactive": {}}
            )
        # 只调用 1 次（状态更新），无摘要
        assert mock.call_count == 1

    def test_preserves_incoming_state(self):
        """传入的角色状态在更新后得到正确合并"""
        incoming = {"active": {"李慕": {"location": "鬼市"}}, "inactive": {"胡商": {"name": "胡商"}}}
        response = json.dumps({
            "active": {"李慕": {"location": "金吾狱", "key_events": ["入狱"], "possessions": [], "relationships_update": {}}},
            "inactive": {"胡商": {"name": "胡商", "last_seen": "CH1"}}
        }, ensure_ascii=False)

        with patch("rlwriter.series.utils.llm_call", return_value=response):
            result = update_character_state(
                _mock_client(), _llm_cfg(), 1, "文本", incoming
            )
        assert result["active"]["李慕"]["location"] == "金吾狱"
        assert "胡商" in result["inactive"]
