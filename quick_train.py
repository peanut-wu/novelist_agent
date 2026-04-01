"""快速训练测试：跑 10 步，验证端到端流程"""
import sys
sys.path.insert(0, "src")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from rlwriter.data import NovelDataset
from rlwriter.trainer import Trainer
from rlwriter.utils import load_config

print("=" * 60)
print("RLWriter 快速训练测试 (10 steps)")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载鹿鼎记数据...")
ds = NovelDataset.from_folder("data/novels", min_scene_length=200)
print(f"    总样本数: {len(ds.samples)}")

# 只取前 5 个样本做快速测试
ds.samples = ds.samples[:5]
print(f"    测试样本数: {len(ds.samples)}")
for i, s in enumerate(ds.samples):
    preview = s.label_text[:60].replace("\n", " ")
    print(f"    样本 {i}: scene={s.scene_index}, len={len(s.label_text)}, preview={preview}...")

# 2. 加载配置
print("\n[2] 加载 DeepSeek 配置...")
cfg = load_config("config/deepseek.yaml")
print(f"    模型: {cfg.llm.model}")
print(f"    API: {cfg.llm.api_base}")

# 3. 创建 Trainer
print("\n[3] 创建 Trainer...")
trainer = Trainer(ds, config=cfg, checkpoint_dir="output/checkpoints_test")

# 4. 训练 1 epoch (5 samples)
print("\n[4] 开始训练 (1 epoch, 5 steps)...")
epoch_metrics = trainer.train(
    epochs=1,
    train_ratio=1.0,   # 全部用于训练
    log_every=1,       # 每步打印
    save_every=5,      # 最后保存
)

# 5. 结果汇总
print("\n" + "=" * 60)
print("训练结果汇总")
print("=" * 60)
for em in epoch_metrics:
    print(f"  Epoch {em.epoch}: avg_reward={em.avg_reward:.4f}, "
          f"min={em.min_reward:.4f}, max={em.max_reward:.4f}, "
          f"time={em.total_time_s:.1f}s")

print(f"  总步数: {trainer.global_step}")
print(f"  最佳奖励: {trainer.best_reward:.4f}")

# 查看最终提示词状态
prompt_state = trainer.optimizer.get_prompt_state()
print(f"\n--- 最终文风规则 ---")
print(prompt_state.style_rules[:500] if prompt_state.style_rules else "（空）")

# 查看 checkpoint
ckpts = trainer.ckpt_mgr.list_checkpoints()
print(f"\n--- Checkpoints ({len(ckpts)} 个) ---")
for c in ckpts:
    print(f"  {c['name']}: step={c['step']}, reward={c['best_reward']:.4f}")

print("\n✅ 端到端训练测试完成!")
