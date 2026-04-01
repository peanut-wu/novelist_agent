"""测试：InferenceEngine + Search 接口（mock LLM + Search）"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from rlwriter.inference import (
    GenerationResult, NoOpSearchProvider, WebSearchProvider, InferenceEngine,
)
from rlwriter.checkpoint import CheckpointManager, CheckpointState, TrainingMeta
from rlwriter.utils import LLMConfig


# ========== 数据结构测试 ==========

def test_generation_result():
    """测试 GenerationResult"""
    r = GenerationResult(mode="outline", content="大纲内容", metadata={"idea": "测试"})
    assert r.mode == "outline"
    assert r.content == "大纲内容"
    d = r.to_dict()
    assert d["mode"] == "outline"
    assert d["metadata"]["idea"] == "测试"
    print("✅ GenerationResult 测试通过")


# ========== 搜索接口测试 ==========

def test_noop_search():
    """测试 NoOpSearchProvider"""
    provider = NoOpSearchProvider()
    results = provider.search("测试查询")
    assert results == []
    print("✅ NoOpSearchProvider 测试通过")


def test_web_search_mock():
    """测试 WebSearchProvider（mock httpx）"""
    provider = WebSearchProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {"title": "结果1", "url": "https://a.com", "description": "描述1"},
                {"title": "结果2", "url": "https://b.com", "description": "描述2"},
            ]
        }
    }

    mock_httpx = MagicMock()
    mock_httpx.get.return_value = mock_response

    import sys
    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        results = provider.search("测试", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "结果1"
    assert results[1]["url"] == "https://b.com"
    print("✅ WebSearchProvider mock 测试通过")


def test_web_search_error_handling():
    """测试搜索异常处理"""
    provider = WebSearchProvider(api_key="bad-key")

    mock_httpx = MagicMock()
    mock_httpx.get.side_effect = Exception("网络错误")

    import sys
    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        results = provider.search("测试")

    assert results == []  # 优雅降级
    print("✅ WebSearchProvider 异常处理测试通过")


# ========== Mock LLM ==========

def _mock_llm_for_inference(client, cfg, messages, **kwargs):
    """mock llm_call，根据 prompt 内容返回不同响应"""
    prompt = messages[-1]["content"] if messages else ""

    if "大纲设计师" in prompt or "小说大纲" in prompt:
        return json.dumps({
            "title": "穿越记",
            "genre": "科幻",
            "theme": "科技与人性",
            "chapters": [
                {"chapter": 1, "title": "意外穿越", "summary": "程序员穿越", "beats": ["穿越", "醒来"]},
                {"chapter": 2, "title": "初到古代", "summary": "适应环境", "beats": ["寻找食物", "遇见村民"]},
            ],
            "main_characters": [{"name": "李明", "role": "主角", "arc": "从迷茫到适应"}],
            "conflict": "现代知识 vs 古代环境",
            "climax_chapter": 2,
        }, ensure_ascii=False)

    if "角色" in prompt or "character" in prompt.lower():
        return json.dumps([
            {
                "name": "李明", "appearance": "戴眼镜的年轻人",
                "personality": "聪明但社恐", "background": "程序员",
                "motivation": "找到回家的路", "weakness": "动手能力差",
                "catchphrase": "让我想想", "arc": "从逃避到担当",
                "relationships": ["与张师傅是师徒"],
            }
        ], ensure_ascii=False)

    if "续写助手" in prompt or "章节正文" in prompt or "章正文" in prompt:
        return "李明睁开眼睛，发现自己躺在一片竹林中。阳光透过竹叶洒下斑驳的光影。" * 10

    return "默认响应"


# ========== InferenceEngine 测试 ==========

def _create_test_checkpoint(tmpdir: str) -> str:
    """创建测试用 checkpoint"""
    mgr = CheckpointManager(tmpdir)
    state = CheckpointState(
        meta=TrainingMeta(step=100, best_reward=0.85),
        style_prompt={
            "system_instruction": "你是一位小说续写助手",
            "style_rules": "规则1: 句式长短搭配\n规则2: 使用丰富标点",
            "examples": "示例文本...",
        },
    )
    mgr.save(state, name="step_100")
    return tmpdir


def test_inference_engine_init():
    """测试 InferenceEngine 初始化"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(
        style_prompt={"style_rules": "测试规则"},
    )
    engine = InferenceEngine(state, llm_cfg)
    assert engine.style_prompt["style_rules"] == "测试规则"
    assert isinstance(engine.search, NoOpSearchProvider)
    print("✅ InferenceEngine 初始化测试通过")


def test_inference_engine_with_search():
    """测试带搜索的 InferenceEngine"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState()
    provider = NoOpSearchProvider()
    engine = InferenceEngine(state, llm_cfg, search_provider=provider)
    assert engine.search is provider
    print("✅ InferenceEngine 带搜索测试通过")


def test_format_style_rules():
    """测试文风规则格式化"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(
        style_prompt={"style_rules": "句式长短搭配\n标点丰富"}
    )
    engine = InferenceEngine(state, llm_cfg)
    formatted = engine._format_style_rules()
    assert "文风规则" in formatted
    assert "句式长短搭配" in formatted
    print("✅ _format_style_rules 测试通过")


def test_format_style_rules_empty():
    """测试空文风规则"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(style_prompt={"style_rules": "（待学习）"})
    engine = InferenceEngine(state, llm_cfg)
    assert engine._format_style_rules() == ""
    print("✅ 空文风规则测试通过")


def test_generate_outline():
    """测试大纲生成（mock LLM）"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(style_prompt={"style_rules": "规则1"})
    engine = InferenceEngine(state, llm_cfg)

    with patch("rlwriter.inference.llm_call", side_effect=_mock_llm_for_inference):
        result = engine.generate_outline("一个穿越到古代的程序员")

    assert result.mode == "outline"
    assert result.content
    assert result.metadata["parsed"]["title"] == "穿越记"
    assert len(result.metadata["parsed"]["chapters"]) == 2
    print("✅ generate_outline 测试通过")


def test_generate_outline_with_search():
    """测试大纲生成 + 搜索集成"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState()
    provider = NoOpSearchProvider()
    engine = InferenceEngine(state, llm_cfg, search_provider=provider)

    with patch("rlwriter.inference.llm_call", side_effect=_mock_llm_for_inference):
        result = engine.generate_outline(
            "穿越故事",
            search_query="古代中国生活方式"
        )

    assert result.mode == "outline"
    # NoOpSearchProvider 不返回结果，但不报错
    print("✅ generate_outline with search 测试通过")


def test_generate_characters():
    """测试角色设计（mock LLM）"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState()
    engine = InferenceEngine(state, llm_cfg)

    mock_response = json.dumps([
        {"name": "李明", "motivation": "找到回家的路", "appearance": "戴眼镜",
         "personality": "聪明", "background": "程序员", "weakness": "社恐",
         "catchphrase": "让我想想", "arc": "成长", "relationships": []}
    ], ensure_ascii=False)

    def mock_call(client, cfg, messages, **kwargs):
        return mock_response

    with patch("rlwriter.inference.llm_call", side_effect=mock_call):
        result = engine.generate_characters("test outline")

    assert result.mode == "characters"
    assert result.content == mock_response
    parsed = result.metadata["parsed"]
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "李明"
    print("✅ generate_characters 测试通过")


def test_generate_chapter():
    """测试章节创作（mock LLM）"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(style_prompt={"style_rules": "句式长短搭配"})
    engine = InferenceEngine(state, llm_cfg)

    mock_text = "李明睁开眼睛，发现自己躺在一片竹林中。" * 5

    def mock_call(client, cfg, messages, **kwargs):
        return mock_text

    with patch("rlwriter.inference.llm_call", side_effect=mock_call):
        result = engine.generate_chapter(
            "大纲", "角色",
            chapter_num=1, chapter_title="意外穿越",
            chapter_beats="穿越", previous_summary="",
        )

    assert result.mode == "chapter"
    assert result.content == mock_text
    assert result.metadata["chapter_num"] == 1
    assert result.metadata["title"] == "意外穿越"
    print("✅ generate_chapter 测试通过")


def test_generate_chapter_default_title():
    """测试章节创作 - 默认标题"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState()
    engine = InferenceEngine(state, llm_cfg)

    def mock_call(client, cfg, messages, **kwargs):
        return "章节内容"

    with patch("rlwriter.inference.llm_call", side_effect=mock_call):
        result = engine.generate_chapter("大纲", "角色", chapter_num=3)

    assert result.metadata["title"] == "第3章"
    print("✅ 默认章节标题测试通过")


def test_load_checkpoint():
    """测试从 checkpoint 加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_test_checkpoint(tmpdir)
        llm_cfg = LLMConfig(api_key="test-key")

        engine = InferenceEngine.load(tmpdir, llm_cfg=llm_cfg)
        assert engine.style_prompt["style_rules"] == "规则1: 句式长短搭配\n规则2: 使用丰富标点"
        print("✅ load_checkpoint 测试通过")


def test_system_instruction_injected():
    """测试 system_instruction 被注入为 system message"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(
        style_prompt={
            "system_instruction": "你是一位专业的小说续写助手。严格锁定有限视角。",
            "style_rules": "规则1: 句式长短搭配",
        },
    )
    engine = InferenceEngine(state, llm_cfg)

    messages = engine._build_messages("请续写第一章")

    assert len(messages) == 2, f"期望 2 条 message，实际 {len(messages)}"
    assert messages[0]["role"] == "system"
    assert "有限视角" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "续写第一章" in messages[1]["content"]
    print("✅ system_instruction 注入测试通过")


def test_system_instruction_empty_fallback():
    """测试 system_instruction 为空时只有 user message"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(style_prompt={"style_rules": "规则1"})
    engine = InferenceEngine(state, llm_cfg)

    messages = engine._build_messages("请续写")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    print("✅ system_instruction 空值回退测试通过")


def test_generate_chapter_uses_system_message():
    """测试 generate_chapter 实际发送 system message 给 LLM"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(
        style_prompt={
            "system_instruction": "严格遵循有限视角叙事",
            "style_rules": "动作描写简洁",
        },
    )
    engine = InferenceEngine(state, llm_cfg)

    captured_messages = []

    def mock_call(client, cfg, messages, **kwargs):
        captured_messages.extend(messages)
        return "生成的章节内容"

    with patch("rlwriter.inference.llm_call", side_effect=mock_call):
        engine.generate_chapter("大纲", "角色", chapter_num=1)

    assert len(captured_messages) == 2
    assert captured_messages[0]["role"] == "system"
    assert "有限视角" in captured_messages[0]["content"]
    assert captured_messages[1]["role"] == "user"
    print("✅ generate_chapter system message 测试通过")


def test_ensure_memory_warnings_caches():
    """测试 ensure_memory_warnings 预处理并缓存到 checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建含 memory_nodes 但无 memory_warnings 的 checkpoint
        mgr = CheckpointManager(tmpdir)
        state = CheckpointState(
            meta=TrainingMeta(step=10),
            memory_nodes={
                "node1": {"content": "差异点1: 环境描写过度 → 建议: 减少冗余渲染"},
                "node2": {"content": "差异点2: 对话过于文雅 → 建议: 增加口语化表达"},
            },
        )
        mgr.save(state, name="step_10")

        llm_cfg = LLMConfig(api_key="test-key")
        engine = InferenceEngine.load(tmpdir, llm_cfg=llm_cfg)

        assert engine.state.memory_warnings == ""  # 初始无缓存

        mock_warnings = "**写作禁忌：**\n- 环境描写：减少冗余\n- 对话风格：增加口语化"

        with patch.object(engine, "preprocess_memory_warnings", return_value=mock_warnings):
            result = engine.ensure_memory_warnings(str(Path(tmpdir) / "step_10"))

        assert result == mock_warnings
        assert engine.state.memory_warnings == mock_warnings

        # 验证已写入 checkpoint 文件
        import json
        state_file = Path(tmpdir) / "step_10" / "state.json"
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["memory_warnings"] == mock_warnings
        print("✅ ensure_memory_warnings 缓存测试通过")


def test_ensure_memory_warnings_uses_cache():
    """测试 ensure_memory_warnings 有缓存时直接返回"""
    llm_cfg = LLMConfig(api_key="test-key")
    state = CheckpointState(memory_warnings="已缓存的禁忌清单")
    engine = InferenceEngine(state, llm_cfg)

    result = engine.ensure_memory_warnings()

    assert result == "已缓存的禁忌清单"
    print("✅ ensure_memory_warnings 使用缓存测试通过")


if __name__ == "__main__":
    test_generation_result()
    test_noop_search()
    test_web_search_mock()
    test_web_search_error_handling()
    test_inference_engine_init()
    test_inference_engine_with_search()
    test_format_style_rules()
    test_format_style_rules_empty()
    test_generate_outline()
    test_generate_outline_with_search()
    test_generate_characters()
    test_generate_chapter()
    test_generate_chapter_default_title()
    test_load_checkpoint()
    test_system_instruction_injected()
    test_system_instruction_empty_fallback()
    test_generate_chapter_uses_system_message()
    test_ensure_memory_warnings_caches()
    test_ensure_memory_warnings_uses_cache()
    print("\n🎉 全部 19 个测试通过")
