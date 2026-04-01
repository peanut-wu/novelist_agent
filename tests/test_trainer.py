"""测试：Trainer（mock LLM 调用）"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, "src")

from rlwriter.trainer import StepMetrics, EpochMetrics, Trainer
from rlwriter.data import NovelDataset, TrainingSample
from rlwriter.checkpoint import CheckpointManager
from rlwriter.utils import Config, LLMConfig


# ========== 测试用小说文本 ==========

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


# ========== StepMetrics / EpochMetrics 测试 ==========

def test_step_metrics():
    """测试 StepMetrics 数据结构"""
    m = StepMetrics(
        epoch=0, step=5, sample_id="abc123",
        reward_total=0.75, reward_details={"total": 0.75},
        gradient="差异点: 句式偏短", elapsed_ms=1234.5,
    )
    assert m.epoch == 0
    assert m.step == 5
    assert m.reward_total == 0.75

    d = m.to_dict()
    assert d["step"] == 5
    assert d["reward_total"] == 0.75
    assert "abc123" in d["sample_id"]

    s = m.summary()
    assert "Epoch 0" in s
    assert "Step 5" in s
    assert "0.7500" in s
    print("✅ StepMetrics 测试通过")


def test_epoch_metrics():
    """测试 EpochMetrics 数据结构"""
    m = EpochMetrics(
        epoch=0, num_steps=10,
        avg_reward=0.65, min_reward=0.4, max_reward=0.9,
        total_time_s=120.5,
    )
    s = m.summary()
    assert "Epoch 0" in s
    assert "0.6500" in s
    assert "0.4000" in s
    assert "0.9000" in s
    print("✅ EpochMetrics 测试通过")


# ========== 测试配置 ==========

def _test_config():
    """创建测试用 Config（含 fake API key）"""
    cfg = Config()
    cfg.llm.api_key = "test-key-for-testing"
    return cfg


# ========== Mock LLM ==========

def _make_mock_llm_response(text: str):
    """创建 mock 的 llm_call 返回值工厂"""
    def mock_llm_call(client, cfg, messages, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        if "情节摘要" in prompt:
            return "主角进入房间，发现一封信，内容揭露了一个骗局。"
        if "续写" in prompt or "情节摘要" in prompt:
            return "他推开了门。房间里弥漫着灰尘的气味。桌上的信封泛黄。" * 3
        if "差异" in prompt or "梯度" in prompt:
            return "差异点1: 句式偏短 → 建议: 增加从句使用\n差异点2: 标点单一 → 建议: 使用更多修辞"
        if "优化" in prompt or "提示词" in prompt:
            return json.dumps({
                "system_instruction": "你是一位专业的小说续写助手。",
                "style_rules": "规则1: 句式长短搭配\n规则2: 使用丰富标点",
                "examples": "示例文本...",
            }, ensure_ascii=False)
        if "同源" in prompt or "概率" in prompt:
            return "概率: 0.65, 置信度: medium, 依据: 句式习惯相似"
        if "评审" in prompt or "评估" in prompt:
            return json.dumps({
                "句法结构相似度": 7, "情感节奏匹配度": 6,
                "词汇偏好对齐度": 8, "叙事视角一致性": 9,
                "修辞手法运用": 5
            }, ensure_ascii=False)
        return "默认响应"
    return mock_llm_call


def _make_mock_embed():
    """创建 mock 的 llm_embed"""
    def mock_embed(client, cfg, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    return mock_embed


# ========== Trainer 集成测试 ==========

def test_trainer_init():
    """测试 Trainer 初始化"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        cfg = _test_config()

        trainer = Trainer(ds, config=cfg, checkpoint_dir=str(Path(tmpdir) / "ckpts"))
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_reward == 0.0
        assert len(trainer.dataset) > 0
        print("✅ Trainer 初始化测试通过")


def test_trainer_train_single_step():
    """测试单步训练（mock 所有 LLM 调用）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        # 只取一个样本
        ds.samples = ds.samples[:1]

        cfg = _test_config()
        trainer = Trainer(ds, config=cfg, checkpoint_dir=str(Path(tmpdir) / "ckpts"))

        mock_llm = _make_mock_llm_response("")
        mock_embed = _make_mock_embed()

        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm):

            metrics = trainer._train_step(ds.samples[0], epoch=0)

        assert metrics.step == 1
        assert metrics.epoch == 0
        assert 0.0 <= metrics.reward_total <= 1.0
        assert metrics.elapsed_ms > 0
        assert metrics.gradient  # 非空
        print(f"✅ 单步训练测试通过 (reward={metrics.reward_total:.4f})")


def test_trainer_full_train():
    """测试完整训练流程（mock LLM，2 个样本 × 1 epoch）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        ds.samples = ds.samples[:2]  # 只用 2 个样本

        cfg = _test_config()
        trainer = Trainer(ds, config=cfg, checkpoint_dir=str(Path(tmpdir) / "ckpts"))

        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            epoch_metrics = trainer.train(epochs=1, train_ratio=1.0,
                                          log_every=1, save_every=1)

        assert len(epoch_metrics) == 1
        assert epoch_metrics[0].num_steps == 2
        assert epoch_metrics[0].avg_reward > 0
        assert trainer.global_step == 2
        assert trainer.best_reward > 0

        # 验证 checkpoint 被保存
        ckpts = trainer.ckpt_mgr.list_checkpoints()
        assert len(ckpts) >= 2  # 至少 step_1, step_2, final
        print(f"✅ 完整训练流程测试通过 "
              f"(avg_reward={epoch_metrics[0].avg_reward:.4f}, "
              f"checkpoints={len(ckpts)})")


def test_trainer_checkpoint_creation():
    """测试 checkpoint 自动创建"""
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

        # 验证 checkpoint 文件
        ckpt_mgr = CheckpointManager(ckpt_dir)
        latest = ckpt_mgr.latest()
        assert latest is not None
        assert latest.meta.step > 0
        assert latest.meta.best_reward > 0
        assert latest.style_prompt  # 非空
        print("✅ Checkpoint 创建测试通过")


def test_trainer_get_training_log():
    """测试训练日志获取"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        ds.samples = ds.samples[:1]

        cfg = _test_config()
        trainer = Trainer(ds, config=cfg, checkpoint_dir=str(Path(tmpdir) / "ckpts"))

        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            trainer.train(epochs=1, train_ratio=1.0, log_every=1, save_every=1)

        log = trainer.get_training_log()
        assert "global_step" in log
        assert "best_reward" in log
        assert "epoch_metrics" in log
        assert "step_metrics" in log
        assert log["global_step"] > 0
        print("✅ get_training_log 测试通过")


def test_trainer_multiple_epochs():
    """测试多 epoch 训练"""
    with tempfile.TemporaryDirectory() as tmpdir:
        novel_dir = Path(tmpdir) / "novels"
        novel_dir.mkdir()
        (novel_dir / "test.txt").write_text(NOVEL_TEXT, encoding="utf-8")

        ds = NovelDataset.from_folder(str(novel_dir), min_scene_length=50)
        ds.samples = ds.samples[:1]  # 1 个样本

        cfg = _test_config()
        trainer = Trainer(ds, config=cfg, checkpoint_dir=str(Path(tmpdir) / "ckpts"))

        mock_llm = _make_mock_llm_response("")

        with patch("rlwriter.trainer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.optimizer.llm_call", side_effect=mock_llm), \
             patch("rlwriter.reward.llm_call", side_effect=mock_llm), \
             patch("rlwriter.memory.llm_call", side_effect=mock_llm), \
             patch("rlwriter.plot_design.optimizer.llm_call", side_effect=mock_llm):

            epoch_metrics = trainer.train(epochs=3, train_ratio=1.0,
                                          log_every=1, save_every=999)

        assert len(epoch_metrics) == 3
        assert trainer.global_step == 3  # 1 sample × 3 epochs
        for em in epoch_metrics:
            assert em.num_steps == 1
            assert em.avg_reward > 0
        print(f"✅ 多 epoch 训练测试通过 (total_steps={trainer.global_step})")


if __name__ == "__main__":
    test_step_metrics()
    test_epoch_metrics()
    test_trainer_init()
    test_trainer_train_single_step()
    test_trainer_full_train()
    test_trainer_checkpoint_creation()
    test_trainer_get_training_log()
    test_trainer_multiple_epochs()
    print("\n🎉 全部 8 个测试通过")
