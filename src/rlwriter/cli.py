"""RLWriter 命令行入口

用法：
  rlwriter train -n data/novels/          # 从文件夹训练
  rlwriter train -n novel.txt             # 从单个文件训练
  rlwriter generate -c ckpts/best -i "创意"  # 推理：大纲
  rlwriter generate -c ckpts/best -i "创意" --mode characters  # 推理：人设
  rlwriter generate -c ckpts/best -i "创意" --mode chapter --chapter 1  # 推理：章节
  rlwriter analyze -t novel.txt           # 文体学分析
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="rlwriter",
        description="基于强化学习与记忆机制的大模型小说文风模拟框架",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ========== train 子命令 ==========
    train_parser = subparsers.add_parser("train", help="训练文风模拟")
    train_parser.add_argument("--novel", "-n", required=True,
                              help="小说文件/文件夹路径（.txt 文件或包含 .txt 的文件夹）")
    train_parser.add_argument("--config", "-c", default="config/default.yaml",
                              help="配置文件路径")
    train_parser.add_argument("--epochs", "-e", type=int, default=3,
                              help="训练轮数 (默认: 3)")
    train_parser.add_argument("--train-ratio", type=float, default=0.8,
                              help="训练集比例 (默认: 0.8)")
    train_parser.add_argument("--log-every", type=int, default=5,
                              help="每 N 步打印日志 (默认: 5)")
    train_parser.add_argument("--save-every", type=int, default=5,
                              help="每 N 步保存 checkpoint (默认: 5)")
    train_parser.add_argument("--fresh", action="store_true",
                              help="从头训练，忽略已有 checkpoint")
    train_parser.add_argument("--checkpoint-dir", default=None,
                              help="checkpoint 目录（默认: {output}/checkpoints）")
    train_parser.add_argument("--output", "-o", default="output",
                              help="输出目录")

    # ========== generate 子命令 ==========
    gen_parser = subparsers.add_parser("generate", help="推理：从 checkpoint 生成")
    gen_parser.add_argument("--config", default="config/default.yaml",
                            help="配置文件路径 (默认: config/default.yaml)")
    gen_parser.add_argument("--checkpoint", "-c", required=True,
                            help="checkpoint 路径")
    gen_parser.add_argument("--idea", "-i", default=None,
                            help="创意/指令文本（outline 模式必填，其他模式可选）")
    gen_parser.add_argument("--mode", "-m", default="outline",
                            choices=["outline", "characters", "chapter"],
                            help="生成模式 (默认: outline)")
    gen_parser.add_argument("--chapter", type=int, default=1,
                            help="章节号 (chapter 模式)")
    gen_parser.add_argument("--outline-file", default=None,
                            help="大纲文件路径 (characters/chapter 模式)")
    gen_parser.add_argument("--characters-file", default=None,
                            help="角色设定文件路径 (chapter 模式)")
    gen_parser.add_argument("--search", default=None,
                            help="搜索引擎查询词 (可选)")
    gen_parser.add_argument("--output", "-o", default=None,
                            help="输出文件路径")

    # ========== analyze 子命令 ==========
    analyze_parser = subparsers.add_parser("analyze", help="分析文本的文体学特征")
    analyze_parser.add_argument("--text", "-t", required=True, help="文本文件路径")
    analyze_parser.add_argument("--compare", default=None, help="对比文本文件路径")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "generate":
        _cmd_generate(args)
    elif args.command == "analyze":
        _cmd_analyze(args)
    else:
        parser.print_help()


def _cmd_train(args):
    """训练命令"""
    from .data import NovelDataset
    from .trainer import Trainer
    from .utils import load_config

    novel_path = Path(args.novel)

    # 加载数据集
    if novel_path.is_dir():
        print(f"从文件夹加载: {novel_path}")
        dataset = NovelDataset.from_folder(str(novel_path))
    elif novel_path.is_file():
        print(f"从文件加载: {novel_path}")
        dataset = NovelDataset.from_single(str(novel_path))
    else:
        print(f"错误: 路径不存在: {novel_path}")
        sys.exit(1)

    print(f"数据集: {len(dataset.novels)} 部小说, {len(dataset)} 个样本")

    # 加载配置
    cfg = load_config(args.config)

    checkpoint_dir = args.checkpoint_dir or str(Path(args.output) / "checkpoints")

    # 创建训练器
    trainer = Trainer(
        dataset, config=cfg,
        checkpoint_dir=checkpoint_dir,
    )

    # 自动续训
    if not args.fresh:
        if trainer.resume_from_checkpoint():
            print(f"  🔄 已从断点恢复训练")
        else:
            print(f"  🆕 未找到 checkpoint，从头开始")

    # 训练
    epoch_metrics = trainer.train(
        epochs=args.epochs,
        train_ratio=args.train_ratio,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    # 输出结果
    print(f"\n训练完成!")
    print(f"  总步数: {trainer.global_step}")
    print(f"  最佳奖励: {trainer.best_reward:.4f}")
    print(f"  Checkpoint 目录: {checkpoint_dir}")

    # 保存训练日志
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.get_training_log(), f, ensure_ascii=False, indent=2)
    print(f"  训练日志: {log_path}")


def _cmd_generate(args):
    """推理命令"""
    from .inference import InferenceEngine, NoOpSearchProvider, WebSearchProvider
    from .utils import load_config

    # 加载配置
    cfg = load_config(args.config)
    llm_cfg = cfg.llm

    # 搜索引擎
    search_provider = NoOpSearchProvider()
    if args.search:
        import os
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if api_key:
            search_provider = WebSearchProvider(api_key=api_key)
            print(f"已启用 Brave Search (query: {args.search})")
        else:
            print("提示: 设置 BRAVE_API_KEY 环境变量可启用搜索")

    engine = InferenceEngine.load(args.checkpoint, llm_cfg=llm_cfg,
                                  search_provider=search_provider)

    idea = args.idea

    if args.mode == "outline":
        if not idea:
            print("错误: outline 模式需要 --idea 参数")
            sys.exit(1)
        print(f"生成大纲: {idea[:50]}...")
        result = engine.generate_outline(idea, search_query=args.search)
        output = result.content
        if result.metadata.get("parsed"):
            parsed = result.metadata["parsed"]
            output = json.dumps(parsed, ensure_ascii=False, indent=2)
            print(f"  标题: {parsed.get('title', 'N/A')}")
            print(f"  章节数: {len(parsed.get('chapters', []))}")

    elif args.mode == "characters":
        if not args.outline_file:
            print("错误: characters 模式需要 --outline-file 参数")
            sys.exit(1)
        outline = _load_text_or_file(args.outline_file, "")
        print(f"设计角色 (大纲长度: {len(outline)} 字)...")
        result = engine.generate_characters(outline, search_query=args.search)
        output = result.content
        if result.metadata.get("parsed"):
            output = json.dumps(result.metadata["parsed"], ensure_ascii=False, indent=2)
            print(f"  角色数: {len(result.metadata['parsed'])}")

    elif args.mode == "chapter":
        if not args.outline_file:
            print("错误: chapter 模式需要 --outline-file 参数")
            sys.exit(1)
        outline = _load_text_or_file(args.outline_file, "")
        characters = _load_text_or_file(args.characters_file, "")
        print(f"创作第 {args.chapter} 章...")
        result = engine.generate_chapter(
            outline, characters,
            chapter_num=args.chapter,
            search_query=args.search,
        )
        output = result.content
        print(f"  字数: {len(output)}")

    else:
        print(f"未知模式: {args.mode}")
        sys.exit(1)

    # 输出
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"已保存: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(output[:2000])
        if len(output) > 2000:
            print(f"... (共 {len(output)} 字)")


def _cmd_analyze(args):
    """文体学分析命令"""
    from .reward import StylometricAnalyzer

    analyzer = StylometricAnalyzer()

    text = Path(args.text).read_text(encoding="utf-8")
    features = analyzer.analyze(text)
    print("\n=== 文体学特征 ===")
    print(json.dumps(features.to_dict(), ensure_ascii=False, indent=2))

    if args.compare:
        compare_text = Path(args.compare).read_text(encoding="utf-8")
        compare_features = analyzer.analyze(compare_text)
        print("\n=== 对比文本特征 ===")
        print(json.dumps(compare_features.to_dict(), ensure_ascii=False, indent=2))


def _load_text_or_file(path: str, fallback: str) -> str:
    """加载文件内容或使用 fallback 文本"""
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return fallback


if __name__ == "__main__":
    main()
