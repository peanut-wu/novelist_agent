"""测试：plot_design 模块（数据结构 + 分析器 + 奖励 + 优化器）"""

import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from rlwriter.plot_design import (
    PlotBeat, CharacterArc, Foreshadowing, PlotStructure,
    PlotStructureAnalyzer, PlotDesignReward, PlotRewardWeights,
    PlotDesignOptimizer, PlotPromptState,
)
from rlwriter.plot_design.optimizer import INITIAL_STRUCTURE
from rlwriter.utils import LLMConfig


# ========== 数据结构测试 ==========

def test_plot_structure():
    ps = PlotStructure(
        beats=[PlotBeat(0, "setup", "开局", 0.3, ["A"], "轻松")],
        character_arcs=[CharacterArc("A", 0, "求生", "被动→主动")],
        foreshadowings=[Foreshadowing(0, 2, "钥匙", 0.7)],
        tension_curve=[0.3, 0.6, 0.9],
        pacing_profile=[0.4, 0.5, 0.8],
    )
    d = ps.to_dict()
    assert len(d["beats"]) == 1
    assert d["beats"][0]["type"] == "setup"
    assert d["character_arcs"][0]["character"] == "A"
    assert d["foreshadowings"][0]["payoff"] == 2
    assert d["tension_curve"] == [0.3, 0.6, 0.9]
    print("✅ PlotStructure 测试通过")


def test_plot_structure_empty():
    ps = PlotStructure()
    d = ps.to_dict()
    assert d["beats"] == []
    assert d["tension_curve"] == []
    print("✅ PlotStructure 空值测试通过")


# ========== 分析器测试 ==========

def _mock_analyzer_llm(client, cfg, messages, **kwargs):
    prompt = messages[0]["content"]
    if "节拍" in prompt:
        return json.dumps([
            {"beat_type": "setup", "summary": "韦小宝出场", "tension": 0.2, "characters": ["韦小宝"], "tone": "轻松"},
            {"beat_type": "climax", "summary": "皇宫冒险", "tension": 0.8, "characters": ["韦小宝"], "tone": "紧张"},
        ], ensure_ascii=False)
    if "弧线" in prompt:
        return json.dumps([{"character": "韦小宝", "scene_index": 0, "motivation": "生存", "state": "狡猾→忠诚", "relationship_changes": ["与康熙从利用→真情"]}])
    if "伏笔" in prompt:
        return json.dumps([{"setup_scene": 0, "payoff_scene": 5, "element": "四十二章经", "subtlety": 0.6}])
    if "张力" in prompt:
        return json.dumps([{"tension": 0.2, "pacing": 0.3}, {"tension": 0.8, "pacing": 0.7}])
    return "[]"


def test_analyzer_extract_beats():
    client = MagicMock()
    analyzer = PlotStructureAnalyzer(client, LLMConfig())
    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_analyzer_llm):
        beats = analyzer._extract_beats([{"text": "第一章", "children": [0]}])
    assert len(beats) == 2
    assert beats[0].beat_type == "setup"
    assert beats[1].tension_level == 0.8
    print("✅ analyzer._extract_beats 测试通过")


def test_analyzer_extract_arcs():
    client = MagicMock()
    analyzer = PlotStructureAnalyzer(client, LLMConfig())
    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_analyzer_llm):
        arcs = analyzer._extract_arcs("全书摘要", [{"text": "章节"}])
    assert len(arcs) == 1
    assert arcs[0].character == "韦小宝"
    assert "→" in arcs[0].state
    print("✅ analyzer._extract_arcs 测试通过")


def test_analyzer_extract_foreshadowings():
    client = MagicMock()
    analyzer = PlotStructureAnalyzer(client, LLMConfig())
    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_analyzer_llm):
        fs = analyzer._extract_foreshadowings([{"summary": "场景0"}])
    assert len(fs) == 1
    assert fs[0].element == "四十二章经"
    print("✅ analyzer._extract_foreshadowings 测试通过")


def test_analyzer_full():
    client = MagicMock()
    analyzer = PlotStructureAnalyzer(client, LLMConfig())
    tree = {
        "book": [{"text": "鹿鼎记"}],
        "chapter": [{"index": 0, "text": "第一章", "children": [0]}],
        "scene": [{"index": 0, "summary": "开场"}],
    }
    with patch("rlwriter.plot_design.analyzer.llm_call", side_effect=_mock_analyzer_llm):
        ps = analyzer.analyze(tree)
    assert isinstance(ps, PlotStructure)
    assert len(ps.beats) > 0
    assert len(ps.character_arcs) > 0
    assert len(ps.tension_curve) > 0
    print("✅ analyzer.analyze 完整测试通过")


def test_analyzer_parse_json():
    assert PlotStructureAnalyzer._parse_json("没有JSON") == []
    assert PlotStructureAnalyzer._parse_json('[{"a":1}]') == [{"a": 1}]
    print("✅ analyzer._parse_json 测试通过")


# ========== 奖励测试 ==========

def _mock_reward_llm(client, cfg, messages, **kwargs):
    prompt = messages[0]["content"]
    if "节拍分布" in prompt:
        return '{"节拍分布匹配": 7, "节奏一致性": 8, "冲突升级相似": 6, "结构完整性": 7}'
    if "动机变化" in prompt:
        return '{"动机变化匹配": 8, "状态转变合理": 7, "关系动态一致": 9}'
    if "铺垫密度" in prompt:
        return '{"铺垫密度匹配": 6, "回收节奏一致": 7, "隐蔽程度符合": 8, "元素多样性": 5}'
    if "曲线形态" in prompt:
        return '{"曲线形态相似": 7, "峰值位置一致": 8, "动态范围匹配": 6, "节奏感吻合": 7}'
    return '{"score": 5}'


def test_plot_reward():
    cfg = LLMConfig(api_key="test")
    reward = PlotDesignReward(cfg)
    orig = PlotStructure(beats=[PlotBeat(0, "setup", "开局", 0.3)])
    gen = PlotStructure(beats=[PlotBeat(0, "setup", "开始", 0.4)])

    with patch("rlwriter.plot_design.reward.llm_call", side_effect=_mock_reward_llm):
        result = reward.compute_reward(orig, gen)

    assert 0 <= result["total"] <= 1
    assert "structure" in result
    assert "character_arc" in result
    assert "foreshadowing" in result
    assert "tension_curve" in result
    print(f"✅ PlotDesignReward 测试通过 (total={result['total']:.4f})")


# ========== 优化器测试 ==========

def _mock_optimizer_llm(client, cfg, messages, **kwargs):
    prompt = messages[0]["content"]
    if "情节大纲" in prompt or "生成" in prompt:
        return json.dumps({"book_summary": "test", "chapters": ["ch1", "ch2"],
                           "character_arcs": ["A:成长"], "foreshadowing_plan": ["伏笔1"]})
    if "差异" in prompt or "梯度" in prompt:
        return "差异点1: 节奏偏慢 → 建议: 加快冲突推进"
    if "优化" in prompt:
        return json.dumps({
            "structure_template": "优化后的结构模板",
            "character_arc_guide": "优化后的弧线指引",
            "pacing_rules": "优化后的节奏规则",
            "foreshadowing_guide": "优化后的伏笔指引",
        })
    return "{}"


def test_plot_optimizer():
    cfg = LLMConfig(api_key="test")
    opt = PlotDesignOptimizer(cfg)
    assert opt.state.structure_template != ""
    assert opt.state.iteration == 0

    def local_mock(client, cfg, messages, **kwargs):
        return json.dumps({"book_summary": "test", "chapters": ["ch1", "ch2"],
                           "character_arcs": ["A:成长"], "foreshadowing_plan": ["伏笔1"]})

    with patch("rlwriter.plot_design.optimizer.llm_call", side_effect=local_mock):
        outline = opt.generate_outline({"chapter": [{"text": "ch1"}]})
    assert outline["book_summary"] == "test"
    assert len(outline["chapters"]) == 2
    print("✅ generate_outline 测试通过")


def test_plot_optimizer_optimize():
    cfg = LLMConfig(api_key="test")
    opt = PlotDesignOptimizer(cfg)

    def local_mock(client, cfg, messages, **kwargs):
        prompt = messages[0]["content"]
        if "优化" in prompt:
            return json.dumps({
                "structure_template": "优化后结构",
                "character_arc_guide": "优化后弧线",
                "pacing_rules": "优化后节奏",
                "foreshadowing_guide": "优化后伏笔",
            })
        return "{}"

    with patch("rlwriter.plot_design.optimizer.llm_call", side_effect=local_mock):
        opt.optimize("差异点1: 节奏偏慢")

    assert opt.state.iteration == 1
    assert "优化后" in opt.state.structure_template
    print("✅ optimize 测试通过")


def test_plot_optimizer_convergence():
    cfg = LLMConfig(api_key="test")
    opt = PlotDesignOptimizer(cfg, convergence_threshold=0.02)

    opt.record(0.8, "梯度1")
    assert not opt.is_converged()

    opt.record(0.81, "梯度2")
    assert opt.is_converged()  # 差值 0.01 < 0.02

    opt.record(0.9, "梯度3")
    assert not opt.is_converged()  # 差值 0.09 > 0.02
    print("✅ convergence 测试通过")


def test_plot_prompt_state_render():
    ps = PlotPromptState(structure_template="模板", pacing_rules="规则")
    rendered = ps.render()
    assert "情节结构模板" in rendered
    assert "节奏控制规则" in rendered
    assert "角色弧线" not in rendered  # 为空不渲染
    print("✅ PlotPromptState.render 测试通过")


if __name__ == "__main__":
    test_plot_structure()
    test_plot_structure_empty()
    test_analyzer_extract_beats()
    test_analyzer_extract_arcs()
    test_analyzer_extract_foreshadowings()
    test_analyzer_full()
    test_analyzer_parse_json()
    test_plot_reward()
    test_plot_optimizer()
    test_plot_optimizer_optimize()
    test_plot_optimizer_convergence()
    test_plot_prompt_state_render()
    print("\n🎉 全部 12 个 plot_design 测试通过")
