"""端到端训练流水线

流程：
1. 加载原文 → 预处理（场景分割 + 摘要抽取）
2. 初始化记忆库、奖励模型、优化器
3. 迭代循环：
   a. 对每个场景，用当前提示词生成正文
   b. 奖励模型评分
   c. 计算文本梯度
   d. 优化提示词组件
   e. 更新记忆库
4. 输出最终提示词 + 生成样例 + 训练日志
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .utils import Config, load_config, get_llm_client, llm_call
from .preprocessing import SceneSegmenter, StyleDecoupledSummarizer, Scene
from .preprocessing.hierarchical import HierarchicalSummarizer
from .preprocessing.dialogue_transform import DialogueTransformer
from .memory import MemoryManager
from .reward import RewardModel
from .optimizer import TextGradientOptimizer


@dataclass
class IterationLog:
    """单轮迭代日志"""
    iteration: int
    scene_index: int
    reward_total: float
    reward_details: dict
    generated_text: str
    gradient: str
    prompt_state: dict
    timestamp: float


@dataclass
class TrainingResult:
    """训练最终结果"""
    total_iterations: int
    final_prompt: dict
    final_reward: float
    reward_history: list[float]
    memory_stats: dict
    logs: list[dict]
    output_dir: str


class RLPipeline:
    """强化学习文风模拟训练流水线"""

    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        self.config = config or load_config(config_path)
        self.logger = self._setup_logger()

        # 初始化组件
        self.client = get_llm_client(self.config.llm)
        self.segmenter = SceneSegmenter(self.client, self.config.preprocessing.segmentation, self.config.llm)
        self.summarizer = StyleDecoupledSummarizer(self.client, self.config.preprocessing.summarization, self.config.llm)
        self.hierarchical = HierarchicalSummarizer(
            self.client, self.config.llm,
            scenes_per_chapter=self.config.preprocessing.summarization.scenes_per_chapter,
            chapters_per_volume=self.config.preprocessing.summarization.chapters_per_volume,
        )
        self.dialogue_transformer = DialogueTransformer(self.client, self.config.llm) if self.config.preprocessing.summarization.use_dialogue_transform else None
        self.memory = MemoryManager(self.config.memory)
        self.reward_model = RewardModel(self.config.reward, self.config.llm)
        self.optimizer = TextGradientOptimizer(self.config.optimizer, self.config.llm, self.memory)

        self.logs: list[IterationLog] = []

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("rlwriter")
        logger.setLevel(getattr(logging, self.config.pipeline.log_level))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
        return logger

    def load_novel(self, file_path: str) -> str:
        """加载小说文本"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"小说文件不存在: {file_path}")

        # 尝试多种编码
        for encoding in ["utf-8", "gb18030", "gbk", "gb2312", "latin-1"]:
            try:
                text = path.read_text(encoding=encoding)
                self.logger.info(f"加载成功: {file_path} ({encoding}, {len(text)}字)")
                return text
            except (UnicodeDecodeError, UnicodeError):
                continue

        raise ValueError(f"无法解码文件: {file_path}")

    def preprocess(self, novel_text: str) -> list[Scene]:
        """预处理：场景分割 → 对话转换 → 摘要抽取 → 层次化合并"""
        self.logger.info("=" * 50)
        self.logger.info("阶段 1: 文本预处理")
        self.logger.info("=" * 50)

        # 1. 场景分割
        self.logger.info("  → 语义场景分割...")
        scenes = self.segmenter.segment(novel_text)
        self.logger.info(f"  → 分割完成: {len(scenes)} 个场景")

        # 2. 对话→结构化描述转换（可选，风格剥离增强）
        if self.dialogue_transformer:
            self.logger.info("  → 对话→结构化描述转换...")
            for scene in scenes:
                scene.text = self.dialogue_transformer.transform(scene.text)
            self.logger.info("  → 转换完成")

        # 3. 风格剥离摘要抽取
        self.logger.info("  → 风格剥离摘要抽取...")
        scenes = self.summarizer.process_scenes(scenes)
        self.logger.info("  → 摘要完成")

        # 4. 递归层次化合并（场景→章节→卷宗→全书）
        self.logger.info("  → 递归层次化合并...")
        scene_summaries = [s.summary for s in scenes if s.summary]
        summary_tree = self.hierarchical.build_tree(scene_summaries)
        book_anchors = self.hierarchical.get_book_anchors()
        self.logger.info(f"  → 合并完成: {len(summary_tree.get('chapter', []))} 章, "
                         f"{len(summary_tree.get('volume', []))} 卷")

        # 5. 保存预处理结果
        output_dir = Path(self.config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        preprocess_data = {
            "scenes": [{
                "index": s.index, "char_count": s.char_count,
                "text_preview": s.text[:200] + "..." if len(s.text) > 200 else s.text,
                "summary": s.summary,
            } for s in scenes],
            "summary_tree": {
                level: [{"index": n.index, "text": n.text[:300], "children": n.children}
                         for n in nodes]
                for level, nodes in summary_tree.items()
            },
            "book_anchors": book_anchors,
        }

        with open(output_dir / "preprocessed_scenes.json", "w", encoding="utf-8") as f:
            json.dump(preprocess_data, f, ensure_ascii=False, indent=2)

        return scenes

    def train(self, novel_path: str, num_scenes: Optional[int] = None) -> TrainingResult:
        """主训练循环"""
        # 加载 & 预处理
        novel_text = self.load_novel(novel_path)
        scenes = self.preprocess(novel_text)

        if num_scenes:
            scenes = scenes[:num_scenes]

        self.logger.info("=" * 50)
        self.logger.info("阶段 2: 强化学习迭代优化")
        self.logger.info(f"  场景数: {len(scenes)}")
        self.logger.info(f"  最大迭代: {self.config.optimizer.max_iterations}")
        self.logger.info("=" * 50)

        reward_history = []

        for iteration in range(self.config.optimizer.max_iterations):
            self.logger.info(f"\n--- 迭代 {iteration + 1}/{self.config.optimizer.max_iterations} ---")

            iteration_rewards = []

            for scene in scenes:
                if not scene.summary:
                    continue

                # 1. 生成（附带多层级上下文）
                self.logger.info(f"  场景 {scene.index}: 生成中...")
                hierarchy_ctx = self.hierarchical.get_context_for_scene(scene.index)
                enriched_summary = f"{scene.summary}\n\n{hierarchy_ctx}" if hierarchy_ctx else scene.summary
                generated = self.optimizer.generate_text(enriched_summary)

                # 2. 评分
                reward_result = self.reward_model.compute_reward(scene.text, generated)
                total_reward = reward_result["total"]
                iteration_rewards.append(total_reward)
                self.logger.info(f"  场景 {scene.index}: 奖励 = {total_reward:.4f}")

                # 3. 计算梯度
                gradient = self.optimizer.compute_gradient(scene.text, generated, reward_result)

                # 4. 记录
                self.optimizer.record_iteration(total_reward, reward_result, gradient)
                self.logs.append(IterationLog(
                    iteration=iteration,
                    scene_index=scene.index,
                    reward_total=total_reward,
                    reward_details=reward_result,
                    generated_text=generated[:500],
                    gradient=gradient,
                    prompt_state=self.optimizer.get_prompt_state().to_prompt_components(),
                    timestamp=time.time(),
                ))

                # 5. 更新记忆
                self.memory.record_discovery(
                    rule_content=gradient,
                    tags=[f"scene_{scene.index}", f"iter_{iteration}"],
                    keywords=gradient[:50].split()[:5],
                    scene_idx=scene.index,
                    reward=total_reward,
                )

            # 迭代级别：优化提示词 + 更新记忆权重
            if iteration_rewards:
                avg_reward = sum(iteration_rewards) / len(iteration_rewards)
                reward_history.append(avg_reward)
                self.logger.info(f"  平均奖励: {avg_reward:.4f}")

                # 用最后一条梯度优化提示词（聚合效果）
                if self.logs:
                    last_gradient = self.logs[-1].gradient
                    self.optimizer.optimize_prompt(last_gradient)

                # 收敛检查
                if self.optimizer.is_converged():
                    self.logger.info(f"  ✓ 收敛于迭代 {iteration + 1}")
                    break

            # 记忆权重更新
            self.memory.finalize_iteration()

        # 输出结果
        return self._save_results(reward_history)

    def _save_results(self, reward_history: list[float]) -> TrainingResult:
        """保存训练结果"""
        output_dir = Path(self.config.pipeline.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        final_state = self.optimizer.get_prompt_state()

        result = TrainingResult(
            total_iterations=final_state.iteration,
            final_prompt=final_state.to_prompt_components(),
            final_reward=reward_history[-1] if reward_history else 0.0,
            reward_history=reward_history,
            memory_stats=self.memory.get_stats(),
            logs=[asdict(l) for l in self.logs[-20:]],  # 最后 20 条
            output_dir=str(output_dir),
        )

        # 保存最终提示词
        with open(output_dir / "final_prompt.json", "w", encoding="utf-8") as f:
            json.dump(result.final_prompt, f, ensure_ascii=False, indent=2)

        # 保存训练日志
        with open(output_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_iterations": result.total_iterations,
                "final_reward": result.final_reward,
                "reward_history": result.reward_history,
                "memory_stats": {k: v for k, v in result.memory_stats.items() if k != "top_weighted"},
                "logs": result.logs,
            }, f, ensure_ascii=False, indent=2)

        # 保存可读的最终提示词
        with open(output_dir / "final_prompt_readable.md", "w", encoding="utf-8") as f:
            f.write("# RLWriter 最终提示词\n\n")
            f.write(f"## 迭代次数: {result.total_iterations}\n")
            f.write(f"## 最终奖励: {result.final_reward:.4f}\n\n")
            f.write("---\n\n")
            f.write("## 系统指令\n\n")
            f.write(final_state.system_instruction)
            f.write("\n\n## 文风规则\n\n")
            f.write(final_state.style_rules)
            f.write("\n\n## 参考示例\n\n")
            f.write(final_state.examples)

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"训练完成!")
        self.logger.info(f"  总迭代: {result.total_iterations}")
        self.logger.info(f"  最终奖励: {result.final_reward:.4f}")
        self.logger.info(f"  记忆节点: {result.memory_stats.get('total_nodes', 0)}")
        self.logger.info(f"  输出目录: {output_dir}")
        self.logger.info(f"{'=' * 50}")

        return result


def run(config_path: Optional[str] = None, novel_path: Optional[str] = None,
        num_scenes: Optional[int] = None) -> TrainingResult:
    """便捷入口"""
    pipeline = RLPipeline(config_path=config_path)
    novel = novel_path or os.path.join(pipeline.config.pipeline.input_dir, "novel.txt")
    return pipeline.train(novel, num_scenes=num_scenes)
