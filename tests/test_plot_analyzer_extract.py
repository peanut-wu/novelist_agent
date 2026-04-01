"""测试 PlotStructureAnalyzer 模式提取"""

import json
import pytest
from unittest.mock import MagicMock, patch

from rlwriter.plot_design.analyzer import PlotStructureAnalyzer
from rlwriter.plot_design.structure import PlotStructure, PlotBeat, CharacterArc, Foreshadowing


class TestExtractStructuredPatterns:
    """模式提取单元测试"""

    def _make_structure(self) -> PlotStructure:
        """创建测试用 PlotStructure"""
        return PlotStructure(
            beats=[
                PlotBeat(0, "setup", "李慕入鬼市", 0.3, ["李慕"], "紧张"),
                PlotBeat(1, "complication", "得到密函", 0.6, ["李慕", "王录事"], "悬疑"),
                PlotBeat(2, "climax", "金吾卫抓捕", 0.9, ["李慕", "高延福"], "高压"),
                PlotBeat(3, "resolution", "急智脱身", 0.5, ["李慕"], "释然"),
            ],
            character_arcs=[
                CharacterArc("李慕", 0, "筹科考资费", "从书生到卷入阴谋", ["高延福: 被监视"]),
            ],
            foreshadowings=[
                Foreshadowing(0, 2, "铁牌上的云雷纹", 0.7),
                Foreshadowing(0, -1, "集贤殿书印", 0.8),
            ],
            tension_curve=[0.3, 0.6, 0.9, 0.5],
            pacing_profile=[0.5, 0.7, 0.9, 0.4],
        )

    def test_extract_calls_llm(self):
        """验证提取方法调用 LLM"""
        mock_client = MagicMock()
        mock_llm_cfg = MagicMock()
        mock_llm_cfg.model = "test"
        mock_llm_cfg.temperature = 0.3
        mock_llm_cfg.max_tokens = 3000
        mock_llm_cfg.timeout = 120

        analyzer = PlotStructureAnalyzer(mock_client, mock_llm_cfg)

        # Mock llm_call
        mock_response = json.dumps({
            "structure_template": {"acts": [{"chapters": "1-3", "phase": "建置"}]},
            "foreshadowing_guide": {"techniques": ["物品传递法"]},
        }, ensure_ascii=False)

        with patch("rlwriter.plot_design.analyzer.llm_call", return_value=mock_response):
            structure = self._make_structure()
            result = analyzer.extract_structured_patterns(structure)

        assert "structure_template" in result
        assert "foreshadowing_guide" in result
        assert result["structure_template"]["acts"][0]["phase"] == "建置"

    def test_extract_handles_invalid_json(self):
        """LLM 返回无效 JSON 时返回空 dict"""
        mock_client = MagicMock()
        mock_llm_cfg = MagicMock()

        analyzer = PlotStructureAnalyzer(mock_client, mock_llm_cfg)

        with patch("rlwriter.plot_design.analyzer.llm_call", return_value="这不是JSON"):
            structure = self._make_structure()
            result = analyzer.extract_structured_patterns(structure)

        assert result == {}

    def test_extract_handles_empty_structure(self):
        """空 PlotStructure 也能工作"""
        mock_client = MagicMock()
        mock_llm_cfg = MagicMock()

        analyzer = PlotStructureAnalyzer(mock_client, mock_llm_cfg)

        mock_response = json.dumps({
            "structure_template": {},
            "character_arc_guide": {},
            "pacing_rules": {},
            "foreshadowing_guide": {},
        })

        with patch("rlwriter.plot_design.analyzer.llm_call", return_value=mock_response):
            result = analyzer.extract_structured_patterns(PlotStructure())

        assert isinstance(result, dict)

    def test_extract_prompt_contains_data(self):
        """验证 prompt 包含了分析数据"""
        mock_client = MagicMock()
        mock_llm_cfg = MagicMock()
        captured_prompt = []

        def capture_call(client, cfg, messages, **kwargs):
            captured_prompt.append(messages[0]["content"])
            return '{"structure_template": {}}'

        analyzer = PlotStructureAnalyzer(mock_client, mock_llm_cfg)

        with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=capture_call):
            structure = self._make_structure()
            analyzer.extract_structured_patterns(structure)

        assert len(captured_prompt) == 1
        prompt = captured_prompt[0]
        assert "李慕" in prompt  # 角色名出现在 prompt 中
        assert "setup" in prompt  # beat_type 出现
        assert "铁牌" in prompt  # 伏笔内容出现

    def test_extract_handles_partial_json(self):
        """LLM 返回的 JSON 有额外文本时也能解析"""
        mock_client = MagicMock()
        mock_llm_cfg = MagicMock()

        response_with_extra = '好的，这是结果：\n{"pacing_rules": {"tension_curve_template": [0.3, 0.6, 0.9]}}\n希望对你有帮助。'

        analyzer = PlotStructureAnalyzer(mock_client, mock_llm_cfg)

        with patch("rlwriter.plot_design.analyzer.llm_call", return_value=response_with_extra):
            result = analyzer.extract_structured_patterns(self._make_structure())

        assert "pacing_rules" in result
        assert result["pacing_rules"]["tension_curve_template"] == [0.3, 0.6, 0.9]
