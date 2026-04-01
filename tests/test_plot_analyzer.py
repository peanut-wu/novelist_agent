"""测试：情节结构分析器（mock LLM 调用）"""

import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from rlwriter.plot_design import PlotStructureAnalyzer, PlotStructure
from rlwriter.utils import LLMConfig


def _make_mock_client():
    """创建 mock OpenAI client"""
    client = MagicMock()
    return client


def _mock_llm_call_beats(client, cfg, messages, **kwargs):
    """mock llm_call，根据 prompt 内容返回不同响应"""
    prompt = messages[0]["content"]

    if "情节节拍" in prompt:
        return json.dumps([
            {"beat_type": "setup", "summary": "主角登场", "tension": 0.2, "characters": ["张三"], "tone": "平静"},
            {"beat_type": "complication", "summary": "发现线索", "tension": 0.5, "characters": ["张三"], "tone": "紧张"},
            {"beat_type": "climax", "summary": "最终对决", "tension": 0.9, "characters": ["张三", "反派"], "tone": "激烈"},
        ], ensure_ascii=False)

    if "角色弧线" in prompt or "角色分析" in prompt:
        return json.dumps([
            {"character": "张三", "scene_index": 0, "motivation": "寻找真相", "state": "从迷茫→坚定", "relationship_changes": ["与李四从陌生→信任"]},
        ], ensure_ascii=False)

    if "伏笔" in prompt:
        return json.dumps([
            {"setup_scene": 0, "payoff_scene": 2, "element": "桌上的旧照片", "subtlety": 0.7},
            {"setup_scene": 1, "payoff_scene": -1, "element": "神秘的电话", "subtlety": 0.8},
        ], ensure_ascii=False)

    if "张力" in prompt or "节奏" in prompt:
        return json.dumps([
            {"tension": 0.2, "pacing": 0.3},
            {"tension": 0.5, "pacing": 0.6},
            {"tension": 0.9, "pacing": 0.8},
        ], ensure_ascii=False)

    return "[]"


def test_extract_beats():
    """测试情节节拍提取"""
    client = _make_mock_client()
    cfg = LLMConfig()

    analyzer = PlotStructureAnalyzer(client, cfg)

    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_llm_call_beats):
        chapters = [
            {"index": 0, "text": "第一章摘要...", "children": [0, 1]},
            {"index": 1, "text": "第二章摘要...", "children": [2, 3]},
            {"index": 2, "text": "第三章摘要...", "children": [4]},
        ]
        beats = analyzer._extract_beats(chapters)

    # 每个 chapter 返回 3 个 beat，共 3 个 chapter
    assert len(beats) == 9
    assert beats[0].beat_type == "setup"
    assert beats[0].tension_level == 0.2
    assert beats[2].beat_type == "climax"
    assert beats[2].characters == ["张三", "反派"]
    print("✅ _extract_beats 测试通过")


def test_extract_arcs():
    """测试角色弧线提取"""
    client = _make_mock_client()
    cfg = LLMConfig()
    analyzer = PlotStructureAnalyzer(client, cfg)

    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_llm_call_beats):
        chapters = [{"index": 0, "text": "章节摘要", "children": [0]}]
        arcs = analyzer._extract_arcs("全书摘要", chapters)

    assert len(arcs) == 1
    assert arcs[0].character == "张三"
    assert "→" in arcs[0].state
    assert arcs[0].motivation == "寻找真相"
    print("✅ _extract_arcs 测试通过")


def test_extract_foreshadowings():
    """测试伏笔识别"""
    client = _make_mock_client()
    cfg = LLMConfig()
    analyzer = PlotStructureAnalyzer(client, cfg)

    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_llm_call_beats):
        scenes = [
            {"index": 0, "summary": "场景0摘要"},
            {"index": 1, "summary": "场景1摘要"},
        ]
        fs = analyzer._extract_foreshadowings(scenes)

    assert len(fs) == 2
    assert fs[0].payoff_scene == 2  # 已回收
    assert fs[1].payoff_scene == -1  # 未回收
    assert fs[0].subtlety == 0.7
    print("✅ _extract_foreshadowings 测试通过")


def test_extract_tension_curve():
    """测试张力曲线提取"""
    client = _make_mock_client()
    cfg = LLMConfig()
    analyzer = PlotStructureAnalyzer(client, cfg)

    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_llm_call_beats):
        chapters = [{"index": 0, "text": "章节摘要", "children": [0]}]
        curve = analyzer._extract_tension_curve(chapters)

    assert len(curve) == 3
    assert curve[0]["tension"] == 0.2
    assert curve[2]["tension"] == 0.9
    assert curve[2]["pacing"] == 0.8
    print("✅ _extract_tension_curve 测试通过")


def test_analyze_full():
    """测试完整的 analyze 流程"""
    client = _make_mock_client()
    cfg = LLMConfig()
    analyzer = PlotStructureAnalyzer(client, cfg)

    summary_tree = {
        "book": [{"text": "一本关于寻找真相的小说"}],
        "chapter": [
            {"index": 0, "text": "第一章", "children": [0, 1]},
            {"index": 1, "text": "第二章", "children": [2]},
        ],
        "scene": [
            {"index": 0, "summary": "场景0"},
            {"index": 1, "summary": "场景1"},
            {"index": 2, "summary": "场景2"},
        ],
    }

    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_llm_call_beats):
        structure = analyzer.analyze(summary_tree)

    assert isinstance(structure, PlotStructure)
    assert len(structure.beats) > 0
    assert len(structure.character_arcs) > 0
    assert len(structure.foreshadowings) > 0
    assert len(structure.tension_curve) > 0
    assert len(structure.pacing_profile) > 0

    # 验证 to_dict 也能正常工作
    d = structure.to_dict()
    assert len(d["beats"]) == len(structure.beats)
    print("✅ analyze 完整流程测试通过")


def test_parse_json_array_valid():
    """测试 JSON 解析 - 正常情况"""
    response = '这是分析结果：\n[{"a": 1}, {"b": 2}]\n以上就是'
    result = PlotStructureAnalyzer._parse_json(response)
    assert len(result) == 2
    assert result[0]["a"] == 1
    print("✅ _parse_json 正常解析测试通过")


def test_parse_json_array_invalid():
    """测试 JSON 解析 - 异常情况"""
    assert PlotStructureAnalyzer._parse_json("没有JSON") == []
    assert PlotStructureAnalyzer._parse_json("[broken json") == []
    assert PlotStructureAnalyzer._parse_json("") == []
    print("✅ _parse_json 异常处理测试通过")


def test_analyze_empty_tree():
    """测试空摘要树"""
    client = _make_mock_client()
    cfg = LLMConfig()
    analyzer = PlotStructureAnalyzer(client, cfg)

    with patch("rlwriter.plot_design.analyzer.llm_call", return_value="[]"):
        structure = analyzer.analyze({})

    assert len(structure.beats) == 0
    assert len(structure.character_arcs) == 0
    assert len(structure.foreshadowings) == 0
    assert len(structure.tension_curve) == 0
    print("✅ 空摘要树测试通过")


if __name__ == "__main__":
    test_extract_beats()
    test_extract_arcs()
    test_extract_foreshadowings()
    test_extract_tension_curve()
    test_analyze_full()
    test_parse_json_array_valid()
    test_parse_json_array_invalid()
    test_analyze_empty_tree()
    print("\n🎉 全部测试通过")
