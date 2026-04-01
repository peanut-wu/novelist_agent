"""测试：CheckpointManager"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from rlwriter.checkpoint import CheckpointState, CheckpointManager, TrainingMeta


def test_training_meta_defaults():
    """测试 TrainingMeta 默认值"""
    meta = TrainingMeta()
    assert meta.epoch == 0
    assert meta.step == 0
    assert meta.best_reward == 0.0
    assert meta.reward_history == []
    print("✅ TrainingMeta 默认值测试通过")


def test_checkpoint_state_roundtrip():
    """测试 CheckpointState 序列化/反序列化"""
    state = CheckpointState(
        meta=TrainingMeta(epoch=2, step=100, best_reward=0.85),
        style_prompt={"system_instruction": "你是一位...", "style_rules": "规则1"},
        plot_prompt={"structure_template": "模板1"},
        memory_nodes={"abc123": {"content": "发现1", "weight": 1.5}},
        optimizer_history=[{"iteration": 1, "reward": 0.7}],
    )

    d = state.to_dict()
    assert d["meta"]["step"] == 100
    assert d["meta"]["best_reward"] == 0.85
    assert d["style_prompt"]["system_instruction"] == "你是一位..."
    assert "abc123" in d["memory_nodes"]

    # 反序列化
    restored = CheckpointState.from_dict(d)
    assert restored.meta.step == 100
    assert restored.meta.best_reward == 0.85
    assert restored.style_prompt["system_instruction"] == "你是一位..."
    assert "abc123" in restored.memory_nodes
    assert restored.optimizer_history[0]["reward"] == 0.7
    print("✅ CheckpointState 序列化 roundtrip 测试通过")


def test_checkpoint_state_empty():
    """测试空 CheckpointState"""
    state = CheckpointState()
    d = state.to_dict()
    restored = CheckpointState.from_dict(d)
    assert restored.meta.step == 0
    assert restored.style_prompt == {}
    assert restored.memory_nodes == {}
    print("✅ CheckpointState 空值测试通过")


def test_save_and_load():
    """测试保存和加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        state = CheckpointState(
            meta=TrainingMeta(epoch=1, step=50, best_reward=0.72, reward_history=[0.6, 0.65, 0.72]),
            style_prompt={"system_instruction": "测试指令"},
        )

        path = mgr.save(state, name="test_ckpt")
        assert Path(path).exists()

        loaded = mgr.load("test_ckpt")
        assert loaded.meta.step == 50
        assert loaded.meta.best_reward == 0.72
        assert loaded.style_prompt["system_instruction"] == "测试指令"
        assert len(loaded.meta.reward_history) == 3
        print("✅ save/load 测试通过")


def test_save_default_name():
    """测试默认名称（step_N）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        state = CheckpointState(meta=TrainingMeta(step=42))
        path = mgr.save(state)

        assert "step_42" in path
        loaded = mgr.load("step_42")
        assert loaded.meta.step == 42
        print("✅ 默认名称测试通过")


def test_list_checkpoints():
    """测试列出检查点"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        for step in [10, 20, 30]:
            state = CheckpointState(
                meta=TrainingMeta(step=step, best_reward=step * 0.01)
            )
            mgr.save(state)

        ckpts = mgr.list_checkpoints()
        assert len(ckpts) == 3
        # 按 step 排序
        assert ckpts[0]["step"] == 10
        assert ckpts[2]["step"] == 30
        print(f"✅ list_checkpoints 测试通过 ({len(ckpts)} 个)")


def test_list_checkpoints_empty():
    """测试空检查点目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        assert mgr.list_checkpoints() == []
    print("✅ list_checkpoints 空目录测试通过")


def test_latest():
    """测试加载最新检查点"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        mgr.save(CheckpointState(meta=TrainingMeta(step=10, best_reward=0.5)))
        mgr.save(CheckpointState(meta=TrainingMeta(step=30, best_reward=0.7)))
        mgr.save(CheckpointState(meta=TrainingMeta(step=20, best_reward=0.6)))

        latest = mgr.latest()
        assert latest is not None
        assert latest.meta.step == 30  # step 最大的
        print("✅ latest 测试通过")


def test_best():
    """测试加载最佳检查点"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        mgr.save(CheckpointState(meta=TrainingMeta(step=10, best_reward=0.5)))
        mgr.save(CheckpointState(meta=TrainingMeta(step=30, best_reward=0.7)))
        mgr.save(CheckpointState(meta=TrainingMeta(step=20, best_reward=0.9)))

        best = mgr.best()
        assert best is not None
        assert best.meta.best_reward == 0.9
        assert best.meta.step == 20
        print("✅ best 测试通过")


def test_load_not_found():
    """测试加载不存在的检查点"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        try:
            mgr.load("nonexistent")
            assert False, "应抛出 FileNotFoundError"
        except FileNotFoundError:
            pass
    print("✅ load_not_found 测试通过")


def test_delete():
    """测试删除检查点"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        mgr.save(CheckpointState(meta=TrainingMeta(step=10)))
        mgr.save(CheckpointState(meta=TrainingMeta(step=20)))

        assert len(mgr.list_checkpoints()) == 2
        assert mgr.delete("step_10") is True
        assert len(mgr.list_checkpoints()) == 1
        assert mgr.delete("step_10") is False  # 已删除
        print("✅ delete 测试通过")


def test_save_metadata():
    """测试保存时自动更新 saved_at"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        state = CheckpointState(meta=TrainingMeta(step=5))
        assert state.meta.saved_at == 0

        mgr.save(state)
        assert state.meta.saved_at > 0  # 被自动设置
        print("✅ save metadata 测试通过")


def test_multiple_saves_same_name():
    """测试同名覆盖"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        mgr.save(CheckpointState(meta=TrainingMeta(step=1, best_reward=0.5)), name="ckpt")
        mgr.save(CheckpointState(meta=TrainingMeta(step=2, best_reward=0.8)), name="ckpt")

        loaded = mgr.load("ckpt")
        assert loaded.meta.step == 2
        assert loaded.meta.best_reward == 0.8
        assert len(mgr.list_checkpoints()) == 1  # manifest 中只有一个
        print("✅ 同名覆盖测试通过")


if __name__ == "__main__":
    test_training_meta_defaults()
    test_checkpoint_state_roundtrip()
    test_checkpoint_state_empty()
    test_save_and_load()
    test_save_default_name()
    test_list_checkpoints()
    test_list_checkpoints_empty()
    test_latest()
    test_best()
    test_load_not_found()
    test_delete()
    test_save_metadata()
    test_multiple_saves_same_name()
    print("\n🎉 全部 13 个测试通过")
