#!/usr/bin/env python3
"""预处理 memory_nodes → LLM 聚合为写作禁忌清单 → 缓存到 checkpoint

用法：
  python preprocess_memory.py -c output/checkpoints_50
  python preprocess_memory.py -c output/checkpoints_50 --ckpt-step final
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from rlwriter.inference import InferenceEngine, NoOpSearchProvider
from rlwriter.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="预处理 memory_nodes → 写作禁忌清单")
    parser.add_argument("--checkpoint", "-c", default="output/checkpoints_50", help="checkpoint 目录")
    parser.add_argument("--config", default="config/deepseek.yaml", help="配置文件路径")
    parser.add_argument("--ckpt-step", default=None, help="checkpoint 步骤名（默认自动选最新）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"LLM: {cfg.llm.model} @ {cfg.llm.api_base}")

    engine = InferenceEngine.load(args.checkpoint, llm_cfg=cfg.llm, search_provider=NoOpSearchProvider())

    # 检查是否已有缓存
    if engine.state.memory_warnings:
        print(f"已有缓存（{len(engine.state.memory_warnings)} 字符），跳过")
        print("如需重新生成，请先删除 state.json 中的 memory_warnings 字段")
        return

    print(f"memory_nodes: {len(engine.memory_nodes)} 条")
    print("正在用 LLM 聚合...")
    warnings = engine.preprocess_memory_warnings()

    if not warnings:
        print("未生成任何禁忌清单（memory_nodes 可能为空或格式不符）")
        return

    print(f"生成完成（{len(warnings)} 字符）")
    print("---")
    print(warnings)
    print("---")

    # 确定 checkpoint 路径
    ckpt_path = Path(args.checkpoint)
    if not (ckpt_path / "state.json").exists():
        # 可能是 checkpoint 目录，找最终的 step
        if args.ckpt_step:
            ckpt_path = ckpt_path / args.ckpt_step
        else:
            steps = sorted([d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
                           key=lambda d: int(d.name.replace("step_", "")))
            ckpt_path = steps[-1] if steps else ckpt_path / "final"

    engine.save_memory_warnings_to_checkpoint(warnings, str(ckpt_path))
    print(f"已缓存到: {ckpt_path / 'state.json'}")


if __name__ == "__main__":
    main()
