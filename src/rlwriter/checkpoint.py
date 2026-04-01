"""Checkpoint 管理器

管理训练状态的序列化/反序列化，类似 PyTorch 的 torch.save/load。

保存内容：
- prompt_states: 文风提示词 + 情节提示词（当前最优）
- memory: 文风记忆库节点 + 权重
- optimizer_state: 迭代历史、收敛状态
- training_meta: epoch/step/奖励曲线/时间戳
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any


@dataclass
class TrainingMeta:
    """训练元数据"""
    epoch: int = 0
    step: int = 0
    total_samples: int = 0
    best_reward: float = 0.0
    reward_history: list[float] = field(default_factory=list)
    started_at: float = 0.0
    saved_at: float = 0.0
    notes: str = ""


@dataclass
class CheckpointState:
    """完整的检查点状态"""
    meta: TrainingMeta = field(default_factory=TrainingMeta)

    # 文风提示词组件
    style_prompt: dict = field(default_factory=dict)  # {system_instruction, style_rules, examples}

    # 情节设计提示词组件
    plot_prompt: dict = field(default_factory=dict)    # {structure_template, character_arc_guide, ...}

    # 文风记忆库
    memory_nodes: dict = field(default_factory=dict)   # {node_id: StyleNode dict}

    # 优化器状态
    optimizer_history: list[dict] = field(default_factory=list)

    # LLM 预处理的 memory_warnings 缓存（可选）
    memory_warnings: str = ""

    def to_dict(self) -> dict:
        d = {
            "meta": asdict(self.meta),
            "style_prompt": self.style_prompt,
            "plot_prompt": self.plot_prompt,
            "memory_nodes": self.memory_nodes,
            "optimizer_history": self.optimizer_history,
        }
        if self.memory_warnings:
            d["memory_warnings"] = self.memory_warnings
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        meta_data = data.get("meta", {})
        return cls(
            meta=TrainingMeta(**{k: v for k, v in meta_data.items()
                                 if k in TrainingMeta.__dataclass_fields__}),
            style_prompt=data.get("style_prompt", {}),
            plot_prompt=data.get("plot_prompt", {}),
            memory_nodes=data.get("memory_nodes", {}),
            optimizer_history=data.get("optimizer_history", []),
            memory_warnings=data.get("memory_warnings", ""),
        )


class CheckpointManager:
    """检查点管理器

    用法：
        mgr = CheckpointManager("output/checkpoints")
        mgr.save(state, name="step_100")
        state = mgr.load("step_100")
        names = mgr.list_checkpoints()
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: CheckpointState, name: Optional[str] = None) -> str:
        """保存检查点

        Args:
            state: 检查点状态
            name: 检查点名称，默认用 step_{step}

        Returns:
            保存的路径
        """
        if name is None:
            name = f"step_{state.meta.step}"

        ckpt_dir = self.checkpoint_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state.meta.saved_at = time.time()

        # 保存状态
        state_path = ckpt_dir / "state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

        # 更新 manifest
        self._update_manifest(name, state.meta)

        return str(ckpt_dir)

    def load(self, name: str) -> CheckpointState:
        """加载检查点

        Args:
            name: 检查点名称

        Returns:
            检查点状态
        """
        ckpt_dir = self.checkpoint_dir / name
        state_path = ckpt_dir / "state.json"

        if not state_path.exists():
            raise FileNotFoundError(f"检查点不存在: {ckpt_dir}")

        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return CheckpointState.from_dict(data)

    def list_checkpoints(self) -> list[dict]:
        """列出所有检查点

        Returns:
            列表，每个元素包含 name, step, reward, saved_at
        """
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        if not manifest_path.exists():
            return []

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 按 step 排序
        entries = list(manifest.values())
        entries.sort(key=lambda x: x.get("step", 0))
        return entries

    def latest(self) -> Optional[CheckpointState]:
        """加载最新的检查点"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return self.load(checkpoints[-1]["name"])

    def best(self) -> Optional[CheckpointState]:
        """加载奖励最高的检查点"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        best = max(checkpoints, key=lambda x: x.get("best_reward", 0))
        return self.load(best["name"])

    def delete(self, name: str) -> bool:
        """删除检查点

        Returns:
            是否成功删除
        """
        import shutil
        ckpt_dir = self.checkpoint_dir / name
        if not ckpt_dir.exists():
            return False

        shutil.rmtree(ckpt_dir)

        # 更新 manifest
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest.pop(name, None)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

        return True

    def _update_manifest(self, name: str, meta: TrainingMeta):
        """更新 manifest 文件"""
        manifest_path = self.checkpoint_dir / self.MANIFEST_FILE

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {}

        manifest[name] = {
            "name": name,
            "step": meta.step,
            "epoch": meta.epoch,
            "best_reward": meta.best_reward,
            "saved_at": meta.saved_at,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
