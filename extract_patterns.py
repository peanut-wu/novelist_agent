#!/usr/bin/env python3
"""后处理：从训练数据提取结构化情节模式 → 存入 checkpoint

用法：
  python extract_patterns.py -c output/checkpoints_50 -n data/novels
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from rlwriter.data import NovelDataset
from rlwriter.plot_design import PlotStructureAnalyzer
from rlwriter.utils import load_config


def build_summary_tree(dataset: NovelDataset) -> dict:
    """从数据集构建摘要树"""
    chapters = []
    scenes = []

    # 按 novel 分组，每个 scene 视为一个"章节"
    for i, sample in enumerate(dataset.samples):
        preview = sample.label_text[:300].replace("\n", " ")
        chapters.append({"text": preview, "index": i})
        scenes.append({
            "summary": preview[:200],
            "text_preview": preview[:200],
            "scene_index": sample.scene_index,
        })

    # 全书摘要 = 各 scene 摘要拼接
    book_text = "\n".join(f"场景{i}: {s['summary'][:100]}" for i, s in enumerate(scenes[:20]))

    return {
        "book": [{"text": book_text}],
        "chapter": chapters,
        "scene": scenes,
    }


def main():
    parser = argparse.ArgumentParser(description="提取结构化情节模式")
    parser.add_argument("--checkpoint", "-c", default="output/checkpoints_50", help="checkpoint 目录")
    parser.add_argument("--novel", "-n", default="data/novels", help="训练数据路径")
    parser.add_argument("--config", default="config/deepseek.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"LLM: {cfg.llm.model}")

    # 加载数据
    ds = NovelDataset.from_folder(args.novel, min_scene_length=200)
    print(f"样本数: {len(ds.samples)}")

    # 构建摘要树
    summary_tree = build_summary_tree(ds)
    print(f"章节摘要: {len(summary_tree['chapter'])} 个")
    print(f"场景摘要: {len(summary_tree['scene'])} 个")

    # 检查是否已有缓存
    ckpt_path = Path(args.checkpoint)
    steps = sorted([d for d in ckpt_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
                    key=lambda d: int(d.name.replace("step_", "")))
    state_path = (steps[-1] if steps else ckpt_path / "final") / "state.json"
    state_data = json.loads(state_path.read_text(encoding="utf-8"))

    if state_data.get("plot_prompt", {}).get("structure_template", "").startswith("{"):
        print("已有结构化 plot_prompt，跳过")
        print("如需重新生成，请将 plot_prompt 重置为初始值")
        return

    # 提取结构化模式
    from rlwriter.utils import get_llm_client
    client = get_llm_client(cfg.llm)
    analyzer = PlotStructureAnalyzer(client, cfg.llm)

    print("\n正在分析情节结构...")
    structure = analyzer.analyze(summary_tree)
    print(f"  节拍: {len(structure.beats)} 个")
    print(f"  角色弧线: {len(structure.character_arcs)} 条")
    print(f"  伏笔: {len(structure.foreshadowings)} 个")
    print(f"  张力曲线: {len(structure.tension_curve)} 个点")

    print("\n正在提取结构化模式...")
    patterns = analyzer.extract_structured_patterns(structure)

    if not patterns:
        print("⚠️ 模式提取失败")
        return

    print("\n提取结果：")
    for key, val in patterns.items():
        print(f"  {key}: {json.dumps(val, ensure_ascii=False)[:100]}...")

    # 写入 checkpoint
    state_data["plot_prompt"] = patterns
    state_path.write_text(json.dumps(state_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存到: {state_path}")


if __name__ == "__main__":
    main()
