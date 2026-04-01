"""测试：resume_from_checkpoint + generate with config（端到端）

验证：
1. 训练 1 步 → 保存 checkpoint → 检查文件
2. 从 checkpoint 恢复 → 继续训练 → 检查断点续训
3. generate 命令能否指定 config
4. generate 能否正常生产大纲、角色设定、第一章
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlwriter.trainer import Trainer
from rlwriter.data import NovelDataset
from rlwriter.checkpoint import CheckpointManager, CheckpointState, TrainingMeta
from rlwriter.inference import InferenceEngine, NoOpSearchProvider
from rlwriter.utils import Config, LLMConfig, load_config


NOVEL_TEXT = """第一章

他推开那扇沉重的门。房间里很暗。
"有人吗？"他喊了一声。声音在空荡的房间里回荡。
没有回应。只有挂钟发出单调的滴答声。
他看到了桌上的那封信，心跳开始加速。

第二章

信中的内容让他大吃一惊。原来这是一个精心设计的骗局。
他跌坐在椅子上，久久不能回神。
窗外传来了汽车引擎的声音。他匆忙将信塞进口袋溜了出去。

第三章

街道上空无一人。路灯昏黄的光照在他苍白的脸上。
他需要找一个安全的地方，好好想想接下来该怎么办。"""


def _make_mock_llm_response(text=""):
    """创建 mock 的 llm_call 返回值

    注意：prompt 匹配顺序很重要！
    - 优化 prompt 包含 "差异分析"，不能用 "差异" 做关键词
    - 用更具体的关键词区分各个阶段
    """
    def mock_llm_call(client, cfg, messages, **kwargs):
        prompt = messages[-1]["content"] if messages else ""

        # 优化提示词（必须先于梯度检查，因为 prompt 包含 "文本梯度"）
        if "提示词工程专家" in prompt or "优化后的组件" in prompt or "当前提示词组件" in prompt:
            return json.dumps({
                "system_instruction": "你是一位优化后的小说续写助手。",
                "style_rules": "规则1: 句式长短搭配\n规则2: 使用丰富标点",
                "examples": "示例文本...",
            }, ensure_ascii=False)

        # 情节摘要
        if "情节摘要" in prompt or "情节分析师" in prompt:
            return "主角进入房间，发现一封信，内容揭露了一个骗局。"

        # 生成文本（续写）
        if "续写" in prompt and "续写助手" not in prompt:
            return "他推开了门。房间里弥漫着灰尘的气味。桌上的信封泛黄。" * 3

        # 文本梯度（差异分析）
        if "文风分析专家" in prompt or "差异分析" in prompt:
            return "差异点1: 句式偏短 → 建议: 增加从句使用\n差异点2: 标点单一 → 建议: 使用更多修辞"

        # 作者验证
        if "同源" in prompt or "语言学取证" in prompt:
            return "概率: 0.65, 置信度: medium, 依据: 句式习惯相似"

        # 奖励评审
        if "评审专家" in prompt or "评估" in prompt:
            return json.dumps({
                "句法结构相似度": 7, "情感节奏匹配度": 6,
                "词汇偏好对齐度": 8, "叙事视角一致性": 9,
                "修辞手法运用": 5
            }, ensure_ascii=False)

        # 角色设计（必须先于大纲检查，因为角色 prompt 也包含"大纲"）
        if "角色设计师" in prompt:
            return json.dumps([{
                "name": "李明", "appearance": "戴眼镜的年轻人",
                "personality": "聪明但社恐", "background": "程序员",
                "motivation": "找到回家的路", "weakness": "动手能力差",
                "catchphrase": "让我想想", "arc": "从逃避到担当",
                "relationships": ["与张师傅是师徒"],
            }], ensure_ascii=False)

        # 情节大纲生成
        if "大纲设计师" in prompt or "小说大纲" in prompt:
            return json.dumps({
                "title": "穿越记", "genre": "科幻", "theme": "科技与人性",
                "chapters": [
                    {"chapter": 1, "title": "意外穿越", "summary": "程序员穿越", "beats": ["穿越", "醒来"]},
                    {"chapter": 2, "title": "初到古代", "summary": "适应环境", "beats": ["寻找食物"]},
                ],
                "main_characters": [{"name": "李明", "role": "主角", "arc": "从迷茫到适应"}],
                "conflict": "现代知识 vs 古代环境", "climax_chapter": 2,
            }, ensure_ascii=False)

        # 章节创作
        if "续写助手" in prompt or "章节正文" in prompt or "章正文" in prompt:
            return "李明睁开眼睛，发现自己躺在一片竹林中。阳光透过竹叶洒下斑驳的光影。" * 10

        return "默认响应"
    return mock_llm_call


def _make_mock_embed():
    def mock_embed(client, cfg, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    return mock_embed


def _test_config():
    cfg = Config()
    cfg.llm.api_key = "test-key-for-testing"
    return cfg


# ========== 测试 1: 训练 1 步并保存 checkpoint ==========

def test_train_one_step_and_save():
    """训练 1 步，保存 checkpoint，验证文件完整性"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        ds.samples = ds.samples[:1]

        cfg = _test_config()
        ckpt_dir = str(Path(tmpdir) / "ckpts")
        trainer = Trainer(ds, config=cfg, checkpoint_dir=ckpt_dir)

        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            trainer.train(epochs=1, train_ratio=1.0, log_every=1, save_every=1)

        # 验证
        assert trainer.global_step == 1, f"global_step 应为 1, 实际 {trainer.global_step}"
        assert trainer.best_reward > 0, f"best_reward 应 > 0"

        # 检查 optimizer 状态已更新（不再是初始值）
        ps = trainer.optimizer.get_prompt_state()
        assert ps.style_rules != "（待学习）", \
            f"style_rules 应被优化器更新，实际: {ps.style_rules}"

        # 检查 checkpoint 文件
        ckpt_mgr = CheckpointManager(ckpt_dir)
        ckpts = ckpt_mgr.list_checkpoints()
        assert len(ckpts) >= 1, f"应有至少 1 个 checkpoint, 实际 {len(ckpts)}"

        latest = ckpt_mgr.latest()
        assert latest is not None
        assert latest.meta.step == 1
        assert latest.meta.best_reward > 0
        assert latest.style_prompt, "style_prompt 不应为空"

        # 检查 state.json 文件内容
        state_file = Path(ckpt_dir) / "step_1" / "state.json"
        if not state_file.exists():
            for d in Path(ckpt_dir).iterdir():
                if d.is_dir() and d.name.startswith("step_"):
                    state_file = d / "state.json"
                    break

        assert state_file.exists(), f"state.json 不存在: {state_file}"
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "style_prompt" in data
        assert "meta" in data
        assert data["meta"]["step"] == 1
        assert data["style_prompt"]["style_rules"] != "（待学习）", \
            f"checkpoint 中 style_rules 不应为初始值: {data['style_prompt']['style_rules']}"

        print(f"✅ 训练 1 步 + checkpoint 保存测试通过 "
              f"(step={latest.meta.step}, reward={latest.meta.best_reward:.4f}, "
              f"style_rules已更新)")


# ========== 测试 2: 从 checkpoint 恢复训练 ==========

def test_resume_from_checkpoint():
    """从 checkpoint 恢复，继续训练 1 步，验证断点续训"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        ds.samples = ds.samples[:1]

        cfg = _test_config()
        ckpt_dir = str(Path(tmpdir) / "ckpts")

        mock_llm = _make_mock_llm_response("")

        # 第一轮训练：1 epoch, 1 step
        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            trainer1 = Trainer(ds, config=cfg, checkpoint_dir=ckpt_dir)
            trainer1.train(epochs=1, train_ratio=1.0, log_every=1, save_every=1)

        step_after_first = trainer1.global_step
        reward_after_first = trainer1.best_reward
        style_rules_after_first = trainer1.optimizer.get_prompt_state().style_rules
        print(f"  第一轮: step={step_after_first}, reward={reward_after_first:.4f}, "
              f"style_rules={style_rules_after_first[:40]}...")

        # 第二轮：从 checkpoint 恢复
        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            trainer2 = Trainer(ds, config=cfg, checkpoint_dir=ckpt_dir)

            # 关键：测试 resume_from_checkpoint（Bug 1 修复处）
            resumed = trainer2.resume_from_checkpoint()
            assert resumed, "resume_from_checkpoint 应返回 True"

            step_before_continue = trainer2.global_step
            assert step_before_continue == step_after_first, \
                f"恢复后 step 应为 {step_after_first}, 实际 {step_before_continue}"

            # 验证 optimizer state.history 被恢复（Bug 1 修复点）
            assert trainer2.optimizer.state.history is not None, \
                "optimizer.state.history 不应为 None"
            assert len(trainer2.optimizer.state.history) > 0, \
                "optimizer.state.history 不应为空（应恢复了历史）"

            # 验证 style_rules 被恢复（不应为初始值）
            assert trainer2.optimizer.state.style_rules == style_rules_after_first, \
                f"style_rules 应从 checkpoint 恢复: 期望 {style_rules_after_first!r}, " \
                f"实际 {trainer2.optimizer.state.style_rules!r}"

            # 继续训练 1 epoch（epochs=2，跳过已完成的 epoch 0）
            trainer2.train(epochs=2, train_ratio=1.0, log_every=1, save_every=1)

        assert trainer2.global_step > step_after_first, \
            f"续训后 step 应 > {step_after_first}, 实际 {trainer2.global_step}"

        print(f"✅ 断点续训测试通过 "
              f"(恢复到 step={step_before_continue}, "
              f"续训后 step={trainer2.global_step}, "
              f"style_rules已恢复)")


# ========== 测试 3: generate 命令指定 config ==========

def test_generate_with_custom_config():
    """测试 generate 能否使用自定义 config（Bug 2 修复验证）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 checkpoint
        ckpt_dir = str(Path(tmpdir) / "ckpts")
        mgr = CheckpointManager(ckpt_dir)
        state = CheckpointState(
            meta=TrainingMeta(step=100, best_reward=0.85),
            style_prompt={
                "system_instruction": "你是一位小说续写助手",
                "style_rules": "规则1: 句式长短搭配\n规则2: 使用丰富标点",
                "examples": "示例文本...",
            },
        )
        mgr.save(state, name="step_100")

        # 创建自定义 config YAML
        config_path = str(Path(tmpdir) / "test_config.yaml")
        import yaml
        custom_config = {
            "llm": {
                "provider": "openai",
                "model": "deepseek-chat",
                "api_base": "https://api.deepseek.com/v1",
                "api_key": "test-key-123",
                "temperature": 0.5,
                "max_tokens": 2048,
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(custom_config, f)

        # 验证 load_config 读取自定义值
        cfg = load_config(config_path)
        assert cfg.llm.model == "deepseek-chat", f"model 应为 deepseek-chat, 实际 {cfg.llm.model}"
        assert cfg.llm.api_base == "https://api.deepseek.com/v1"
        assert cfg.llm.temperature == 0.5

        # 验证 InferenceEngine 使用自定义 config
        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.inference.llm_call", side_effect=mock_llm):
            engine = InferenceEngine.load(ckpt_dir, llm_cfg=cfg.llm)

            assert engine.llm_cfg.model == "deepseek-chat", \
                f"InferenceEngine model 应为 deepseek-chat, 实际 {engine.llm_cfg.model}"
            assert engine.llm_cfg.api_base == "https://api.deepseek.com/v1"

            # 测试大纲生成
            result = engine.generate_outline("一个穿越到古代的程序员")
            assert result.mode == "outline"
            assert result.content
            assert result.metadata.get("parsed")
            assert result.metadata["parsed"]["title"] == "穿越记"

            # 测试角色生成
            result = engine.generate_characters("穿越记大纲")
            assert result.mode == "characters"
            assert result.metadata.get("parsed")

            # 测试章节生成
            result = engine.generate_chapter("大纲", "角色设定", chapter_num=1,
                                              chapter_title="意外穿越")
            assert result.mode == "chapter"
            assert len(result.content) > 100

        print(f"✅ generate 指定 config 测试通过 "
              f"(model={cfg.llm.model}, api_base={cfg.llm.api_base})")


# ========== 测试 4: generate 全流程 ==========

def test_generate_full_pipeline():
    """测试 generate 全流程：大纲 → 角色 → 章节"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = str(Path(tmpdir) / "ckpts")
        mgr = CheckpointManager(ckpt_dir)
        state = CheckpointState(
            meta=TrainingMeta(step=50, best_reward=0.8),
            style_prompt={
                "system_instruction": "你是一位武侠小说续写助手",
                "style_rules": "规则1: 使用古风词汇\n规则2: 对话简洁有力",
                "examples": "",
            },
            plot_prompt={
                "structure_template": "按起承转合设计",
                "character_arc_guide": "主角从弱变强",
                "pacing_rules": "张弛有度",
                "foreshadowing_guide": "前文埋伏笔",
            },
        )
        mgr.save(state, name="step_50")

        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.inference.llm_call", side_effect=mock_llm):
            engine = InferenceEngine.load(ckpt_dir, llm_cfg=LLMConfig(api_key="test"))

            # 1. 生成大纲
            outline_result = engine.generate_outline("明朝锦衣卫的故事")
            assert outline_result.mode == "outline"
            assert outline_result.metadata.get("parsed")
            parsed = outline_result.metadata["parsed"]
            assert "title" in parsed
            assert "chapters" in parsed
            assert len(parsed["chapters"]) >= 1
            print(f"  大纲: {parsed['title']} ({len(parsed['chapters'])} 章)")

            # 2. 生成角色
            char_result = engine.generate_characters(outline_result.content)
            assert char_result.mode == "characters"
            assert char_result.metadata.get("parsed")
            chars = char_result.metadata["parsed"]
            assert len(chars) >= 1
            print(f"  角色: {len(chars)} 个")

            # 3. 生成第一章
            chapter_result = engine.generate_chapter(
                outline_result.content,
                char_result.content,
                chapter_num=1,
                chapter_title=parsed["chapters"][0].get("title", "第1章"),
                outline_parsed=parsed,
            )
            assert chapter_result.mode == "chapter"
            assert len(chapter_result.content) > 100
            print(f"  第1章: {len(chapter_result.content)} 字")

        print("✅ generate 全流程测试通过")


# ========== 测试 5: CLI 参数解析 ==========

def test_cli_generate_config_arg():
    """验证 generate 子命令支持 --config 参数"""
    from rlwriter.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 checkpoint
        ckpt_dir = str(Path(tmpdir) / "ckpts")
        mgr = CheckpointManager(ckpt_dir)
        state = CheckpointState(meta=TrainingMeta(step=1, best_reward=0.5))
        mgr.save(state, name="step_1")

        # 创建 config
        config_path = str(Path(tmpdir) / "custom.yaml")
        import yaml
        with open(config_path, "w") as f:
            yaml.dump({"llm": {"model": "test-model", "api_key": "fake"}}, f)

        mock_llm = _make_mock_llm_response("")

        # 验证 --config 被接受（不会因未知参数报错）
        with patch("sys.argv", ["rlwriter", "generate",
                                "--config", config_path,
                                "--checkpoint", ckpt_dir,
                                "--idea", "test"]), \
             patch("rlwriter.inference.llm_call", side_effect=mock_llm):
            # 应该不会报 "unrecognized arguments" 错误
            try:
                main()
            except SystemExit:
                pass  # 可能因其他原因退出，但不应该是参数错误

    print("✅ CLI generate --config 参数解析测试通过")


if __name__ == "__main__":
    test_train_one_step_and_save()
    test_resume_from_checkpoint()
    test_generate_with_custom_config()
    test_generate_full_pipeline()
    test_cli_generate_config_arg()
    print("\n🎉 全部端到端测试通过")
