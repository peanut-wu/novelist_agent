"""训练器：RL 训练循环 + Step 级日志 + 自动 Checkpoint

对标经典 DL 训练流程：
  trainer = Trainer(dataset, config)
  trainer.train()  # 每 log_every 步打印 reward，每 save_every 步保存 checkpoint
"""

import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import jieba.analyse

from .data import NovelDataset, TrainingSample
from .checkpoint import CheckpointManager, CheckpointState, TrainingMeta
from .utils import Config, LLMConfig, load_config, get_llm_client, llm_call
from .preprocessing import SceneSegmenter, StyleDecoupledSummarizer, Scene
from .memory import MemoryManager
from .reward import RewardModel
from .optimizer import TextGradientOptimizer
from .plot_design import PlotStructureAnalyzer, PlotDesignReward, PlotDesignOptimizer


@dataclass
class StepMetrics:
    """单步训练指标"""
    epoch: int
    step: int
    sample_id: str
    reward_total: float
    reward_details: dict
    gradient: str
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "sample_id": self.sample_id,
            "reward_total": round(self.reward_total, 4),
            "reward_details": self.reward_details,
            "gradient": self.gradient[:200],  # 截断，日志用
            "elapsed_ms": round(self.elapsed_ms, 1),
        }

    def summary(self) -> str:
        """单行摘要，用于 step 日志"""
        return (f"[Epoch {self.epoch} Step {self.step}] "
                f"reward={self.reward_total:.4f} "
                f"sample={self.sample_id} "
                f"({self.elapsed_ms:.0f}ms)")


@dataclass
class EpochMetrics:
    """单 epoch 汇总指标"""
    epoch: int
    num_steps: int
    avg_reward: float
    min_reward: float
    max_reward: float
    total_time_s: float

    def summary(self) -> str:
        return (f"=== Epoch {self.epoch} 完成 === "
                f"steps={self.num_steps} "
                f"avg_reward={self.avg_reward:.4f} "
                f"min={self.min_reward:.4f} "
                f"max={self.max_reward:.4f} "
                f"time={self.total_time_s:.1f}s")


class Trainer:
    """RL 训练器

    用法：
        dataset = NovelDataset.from_folder("data/novels")
        trainer = Trainer(dataset, config_path="config/default.yaml")
        trainer.train(epochs=3)
    """

    def __init__(self, dataset: NovelDataset, config: Optional[Config] = None,
                 config_path: Optional[str] = None,
                 checkpoint_dir: str = "output/checkpoints"):
        self.config = config or load_config(config_path)
        self.dataset = dataset
        self.ckpt_mgr = CheckpointManager(checkpoint_dir)

        # 初始化组件
        self.client = get_llm_client(self.config.llm)
        self.memory = MemoryManager(self.config.memory, persistent=False)
        self.reward_model = RewardModel(self.config.reward, self.config.llm)
        self.optimizer = TextGradientOptimizer(self.config.optimizer, self.config.llm, self.memory)

        # 情节设计模块
        self.plot_analyzer = PlotStructureAnalyzer(self.client, self.config.llm)
        self.plot_reward = PlotDesignReward(self.config.llm)
        self.plot_optimizer = PlotDesignOptimizer(self.config.llm)
        self._plot_scores: list[float] = []  # 每步的情节评分

        # 日志
        self.logger = logging.getLogger("rlwriter.trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.logger.addHandler(handler)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.step_metrics: list[StepMetrics] = []
        self.epoch_metrics: list[EpochMetrics] = []
        self.best_reward = 0.0

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找 checkpoint 目录中最新的 step checkpoint 名称"""
        ckpt_dir = Path(self.ckpt_mgr.checkpoint_dir)
        if not ckpt_dir.exists():
            return None
        step_dirs = [d for d in ckpt_dir.iterdir()
                     if d.is_dir() and d.name.startswith("step_")]
        if not step_dirs:
            # 检查是否有 final
            if (ckpt_dir / "final" / "state.json").exists():
                return "final"
            return None
        latest = max(step_dirs, key=lambda d: int(d.name.replace("step_", "")))
        return latest.name

    def resume_from_checkpoint(self, name: Optional[str] = None) -> bool:
        """从 checkpoint 恢复训练状态

        Args:
            name: checkpoint 名称，None 则自动找最新的

        Returns:
            是否成功恢复
        """
        if name is None:
            name = self._find_latest_checkpoint()
        if name is None:
            return False

        try:
            state = self.ckpt_mgr.load(name)
        except Exception as e:
            self.logger.warning(f"加载 checkpoint {name} 失败: {e}")
            return False

        # 恢复训练状态
        self.global_step = state.meta.step
        self.current_epoch = state.meta.epoch
        self.best_reward = state.meta.best_reward

        # 恢复提示词状态
        if state.style_prompt:
            prompt_state = self.optimizer.get_prompt_state()
            prompt_state.style_rules = state.style_prompt.get("style_rules", prompt_state.style_rules)
            prompt_state.system_instruction = state.style_prompt.get("system_instruction", prompt_state.system_instruction)
        if state.plot_prompt:
            self.plot_optimizer.state.structure_template = state.plot_prompt.get("structure_template", "")
            self.plot_optimizer.state.character_arc_guide = state.plot_prompt.get("character_arc_guide", "")
            self.plot_optimizer.state.pacing_rules = state.plot_prompt.get("pacing_rules", "")
            self.plot_optimizer.state.foreshadowing_guide = state.plot_prompt.get("foreshadowing_guide", "")

        # 恢复记忆库
        if state.memory_nodes:
            from .memory import StyleNode
            for nid, ndata in state.memory_nodes.items():
                # 兼容旧字段名：tags→context_tags, keywords→trigger_keywords 等
                if "context_tags" not in ndata and "tags" in ndata:
                    ndata["context_tags"] = ndata.pop("tags")
                if "trigger_keywords" not in ndata and "keywords" in ndata:
                    ndata["trigger_keywords"] = ndata.pop("keywords")
                if "scene_indices" not in ndata and "scene_idx" in ndata:
                    ndata["scene_indices"] = [ndata.pop("scene_idx")]
                if "iteration" not in ndata and "access_count" in ndata:
                    ndata["iteration"] = ndata.pop("access_count")
                if "reward_score" not in ndata and "reward" in ndata:
                    ndata["reward_score"] = ndata.pop("reward")
                # 过滤掉 StyleNode 不存在的字段
                valid = {k: v for k, v in ndata.items() if k in StyleNode.__dataclass_fields__}
                node = StyleNode(**valid)
                self.memory.style_memory.nodes[nid] = node

        # 恢复优化历史
        if state.optimizer_history:
            self.optimizer.state.history = state.optimizer_history

        self.logger.info(f"✅ 从 checkpoint [{name}] 恢复: step={self.global_step}, "
                         f"epoch={self.current_epoch}, best_reward={self.best_reward:.4f}")
        return True

    def _generate_summary(self, text: str) -> str:
        """用 LLM 生成风格剥离摘要（替代占位 input_text）"""
        prompt = f"""你是一位专业的小说情节分析师。请对以下小说文本进行情节摘要。

**核心要求：**
1. **剥离风格**：不要保留原文的修辞手法、句式结构、特定词汇
2. **保留骨架**：记录所有关键情节点、人物行为、对话要点
3. **客观叙述**：用平实的白话文描述，如同向他人转述故事情节

**原文：**

{text[:2000]}

---

**情节摘要："""
        messages = [{"role": "user", "content": prompt}]
        return llm_call(self.client, self.config.llm, messages, temperature=0.3, max_tokens=1024)

    def _train_step(self, sample: TrainingSample, epoch: int) -> StepMetrics:
        """单步训练：generate → score → gradient → optimize"""
        t0 = time.time()

        # 1. 获取或生成摘要
        summary = sample.summary or self._generate_summary(sample.label_text)
        if not sample.summary:
            sample.summary = summary

        # 2. 生成文本（用当前提示词）
        generated = self.optimizer.generate_text(summary)

        # 3. 奖励评分（含嵌入式情节评分）
        reward_result = self.reward_model.compute_reward(sample.label_text, generated)
        total_reward = reward_result["total"]
        plot_score = reward_result.get("plot_judge_score", 0.5)
        self._plot_scores.append(plot_score)

        # 4. 计算文本梯度
        gradient = self.optimizer.compute_gradient(sample.label_text, generated, reward_result)

        # 5. 优化提示词
        self.optimizer.optimize_prompt(gradient)
        self.optimizer.record_iteration(total_reward, reward_result, gradient)

        # 6. 更新记忆
        keywords = jieba.analyse.extract_tags(gradient, topK=5)
        self.memory.record_discovery(
            rule_content=gradient,
            tags=[f"scene_{sample.scene_index}", f"epoch_{epoch}"],
            keywords=keywords,
            scene_idx=sample.scene_index,
            reward=total_reward,
        )

        elapsed = (time.time() - t0) * 1000
        self.global_step += 1

        return StepMetrics(
            epoch=epoch,
            step=self.global_step,
            sample_id=sample.sample_id,
            reward_total=total_reward,
            reward_details=reward_result,
            gradient=gradient,
            elapsed_ms=elapsed,
        )

    def train(self, epochs: int = 1, train_ratio: float = 0.8,
              log_every: int = 5, save_every: int = 20,
              shuffle: bool = True, seed: int = 42) -> list[EpochMetrics]:
        """训练入口（文风+情节架构同步训练）

        Args:
            epochs: 训练轮数
            train_ratio: 训练集比例
            log_every: 每 N 步打印日志
            save_every: 每 N 步保存 checkpoint
            shuffle: 是否打乱数据
            seed: 随机种子

        Returns:
            每个 epoch 的汇总指标
        """
        train_samples, val_samples = self.dataset.split(train_ratio, shuffle, seed)

        self.logger.info("=" * 60)
        self.logger.info("训练开始")
        self.logger.info(f"  数据集: {len(self.dataset)} samples "
                         f"(train={len(train_samples)}, val={len(val_samples)})")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  日志间隔: 每 {log_every} 步")
        self.logger.info(f"  保存间隔: 每 {save_every} 步")
        self.logger.info("=" * 60)

        # 如果已恢复 checkpoint，跳过已训练的 epoch
        start_epoch = self.current_epoch if self.current_epoch < epochs else 0
        if start_epoch > 0:
            self.logger.info(f"  跳过已完成的 epoch 0-{start_epoch - 1}")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            epoch_rewards = []

            self.logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            for i, sample in enumerate(train_samples):
                metrics = self._train_step(sample, epoch)
                self.step_metrics.append(metrics)
                epoch_rewards.append(metrics.reward_total)

                # Step 日志
                if self.global_step % log_every == 0 or i == 0:
                    self.logger.info(metrics.summary())

                # 自动 Checkpoint
                if self.global_step % save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # 更新 best
                if metrics.reward_total > self.best_reward:
                    self.best_reward = metrics.reward_total

            # Epoch 汇总
            epoch_time = time.time() - epoch_start
            epoch_summary = EpochMetrics(
                epoch=epoch,
                num_steps=len(train_samples),
                avg_reward=sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
                min_reward=min(epoch_rewards) if epoch_rewards else 0,
                max_reward=max(epoch_rewards) if epoch_rewards else 0,
                total_time_s=epoch_time,
            )
            self.epoch_metrics.append(epoch_summary)
            self.logger.info(epoch_summary.summary())

            # 记忆权重更新
            self.memory.finalize_iteration()

            # 情节架构训练（用累积的 plot 评分更新 plot 优化器）
            if self._plot_scores:
                avg_plot = sum(self._plot_scores) / len(self._plot_scores)
                self.logger.info(f"  📖 情节架构评分 (嵌入): {avg_plot:.4f} (n={len(self._plot_scores)})")
                # 构建有意义的梯度并调用优化
                if avg_plot < 0.6:
                    gradient = (f"Epoch {epoch} 情节评分偏低({avg_plot:.2f})，"
                                f"需要加强结构模板的具体性和伏笔手法的多样性。")
                elif avg_plot < 0.8:
                    gradient = (f"Epoch {epoch} 情节评分中等({avg_plot:.2f})，"
                                f"节奏控制和角色弧线可更具体化。")
                else:
                    gradient = (f"Epoch {epoch} 情节评分良好({avg_plot:.2f})，"
                                f"保持当前模式，微调细节。")
                self.plot_optimizer.optimize(gradient)
                self.plot_optimizer.record(avg_plot, gradient)
                self._plot_scores = []  # 重置

        # 训练结束，保存最终 checkpoint
        self._save_checkpoint("final")

        self.logger.info("\n" + "=" * 60)
        self.logger.info("训练完成!")
        self.logger.info(f"  总步数: {self.global_step}")
        self.logger.info(f"  最佳奖励: {self.best_reward:.4f}")
        self.logger.info("=" * 60)

        return self.epoch_metrics

    def _save_checkpoint(self, name: str):
        """保存当前训练状态到 checkpoint"""
        prompt_state = self.optimizer.get_prompt_state()

        state = CheckpointState(
            meta=TrainingMeta(
                epoch=self.current_epoch,
                step=self.global_step,
                total_samples=len(self.dataset),
                best_reward=self.best_reward,
                reward_history=[m.reward_total for m in self.step_metrics],
                started_at=time.time(),
            ),
            style_prompt=prompt_state.to_prompt_components(),
            plot_prompt=self.plot_optimizer.state.to_prompt_components(),
            memory_nodes={
                nid: node.to_dict()
                for nid, node in self.memory.style_memory.nodes.items()
            },
            optimizer_history=prompt_state.history[-20:],  # 保留最近 20 条
        )

        self.ckpt_mgr.save(state, name)
        self.logger.info(f"  💾 Checkpoint 已保存: {name}")


    def get_training_log(self) -> dict:
        """获取完整的训练日志"""
        return {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "epoch_metrics": [
                {"epoch": e.epoch, "avg_reward": e.avg_reward,
                 "min": e.min_reward, "max": e.max_reward, "time_s": e.total_time_s}
                for e in self.epoch_metrics
            ],
            "step_metrics": [m.to_dict() for m in self.step_metrics[-50:]],  # 最近 50 步
        }
