#!/usr/bin/env python3
"""串联生成多章小说（项目文件夹模式）

用法：
  python gen_series.py --project novels/tangdynasty --idea "穿越到唐朝的故事" --prepare
  python gen_series.py --project novels/tangdynasty --chapters 1-3
  python gen_series.py -p novels/tangdynasty --chapters 1 --checkpoint output/checkpoints_50
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from rlwriter.project import ProjectManager
from rlwriter.inference import InferenceEngine
from rlwriter.search import SearchManager
from rlwriter.foreshadow_tracker import ForeshadowTracker
from rlwriter.nexus_sum import NexusSum
from rlwriter.utils import load_config, get_llm_client, extract_json_from_text
from rlwriter.series.utils import (
    summarize_chapter, update_character_state, update_foreshadow_hooks,
    format_active_characters, clean_chapter_output,
)


def parse_chapters(chapters_str: str, max_ch: int = 100) -> list[int]:
    """解析章节范围，支持 '1,2,3' 和 '1-12' 格式"""
    if "-" in chapters_str:
        start, end = chapters_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in chapters_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="串联生成多章小说")
    parser.add_argument("--project", "-p", required=True,
                        help="小说项目文件夹路径")
    parser.add_argument("--idea", "-i", default=None,
                        help="创意/指令文本（--prepare 时必填）")
    parser.add_argument("--prepare", action="store_true",
                        help="生成大纲 + 角色设定（项目初始化）")
    parser.add_argument("--chapters", default=None,
                        help="章节范围: '1,2' 或 '1-12'（不传则不生成章节）")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="checkpoint 路径（覆盖 project.yaml）")
    parser.add_argument("--config", default=None,
                        help="配置文件路径（覆盖 project.yaml）")
    args = parser.parse_args()

    # 加载项目
    pm = ProjectManager(args.project)
    pm.ensure_dirs()

    # 确定 checkpoint 和 config 路径（命令行 > project.yaml）
    if args.checkpoint:
        pm.config.checkpoint = args.checkpoint
    if args.config:
        pm.config.config = args.config
    checkpoint_path = pm.config.checkpoint
    config_path = pm.config.config

    if not checkpoint_path:
        print("错误: 需要指定 checkpoint 路径（--checkpoint 或 project.yaml 中的 checkpoint）")
        sys.exit(1)

    # 加载配置
    cfg = load_config(config_path)
    inference_cfg = cfg.inference
    print(f"LLM: {cfg.llm.model} @ {cfg.llm.api_base}")
    print(f"项目: {pm.root}")

    # 创建搜索管理器
    search_config = {}
    if hasattr(cfg, 'search') and cfg.search:
        sc = cfg.search
        search_config = {
            "enabled": sc.enabled,
            "mode": sc.mode,
            "source": sc.source,
            "max_results": sc.max_results,
            "api_keys": sc.api_keys if isinstance(sc.api_keys, dict) else {},
        }
    search_mgr = (SearchManager(search_config)
                  if search_config.get("enabled") else None)

    # 加载推理引擎
    engine = InferenceEngine.load(
        checkpoint_path, llm_cfg=cfg.llm,
        search_manager=search_mgr, inference_cfg=inference_cfg)

    # 预处理 memory_warnings
    ckpt_path = Path(checkpoint_path)
    final_dir = ckpt_path / "final"
    if not final_dir.exists():
        steps = [d for d in ckpt_path.iterdir()
                 if d.is_dir() and d.name.startswith("step_")]
        final_dir = (max(steps, key=lambda d: int(d.name.replace("step_", "")))
                     if steps else ckpt_path)
    if not engine.state.memory_warnings:
        print("预处理 memory_warnings（LLM 聚合训练差异）...")
        warnings = engine.ensure_memory_warnings(str(final_dir))
        print(f"  ✓ 已缓存 ({len(warnings)} chars)")
    else:
        print(f"memory_warnings 已缓存 ({len(engine.state.memory_warnings)} chars)")

    # === prepare：生成大纲 + 角色 ===
    if args.prepare:
        if not args.idea:
            print("错误: --prepare 需要 --idea 参数")
            sys.exit(1)

        idea = args.idea
        pm.config.idea = idea

        # 大纲
        if pm.outline_path.exists():
            print(f"  跳过大纲（已有 outline.json）")
            outline_text, outline_data = pm.load_outline()
        else:
            print(f"生成大纲: {idea[:50]}...")
            result = engine.generate_outline(idea)
            outline_data = result.metadata.get("parsed") or {}
            if not outline_data:
                outline_data = extract_json_from_text(result.content, expect_list=False)
            if not outline_data:
                print("错误: 大纲生成失败，未获取到有效 JSON")
                print(f"  LLM 原始输出前 500 字: {result.content[:500]}")
                sys.exit(1)
            pm.save_outline(outline_data)
            outline_text = json.dumps(outline_data, ensure_ascii=False, indent=2)
            title = outline_data.get("title", "N/A")
            n_ch = len(outline_data.get("chapters", []))
            print(f"  ✓ 大纲已保存: {pm.outline_path}")
            print(f"  标题: {title}, 章节数: {n_ch}")

        # 角色
        if pm.characters_path.exists():
            print(f"  跳过角色（已有 characters.json）")
        else:
            print(f"生成角色设定...")
            result = engine.generate_characters(outline_text)
            characters_data = result.metadata.get("parsed") or []
            if not characters_data:
                characters_data = extract_json_from_text(result.content, expect_list=True)
            if not characters_data:
                print("  ⚠️ 角色 JSON 解析失败，保存原始文本")
                characters_data = [{"name": "未解析", "raw": result.content}]
            pm.save_characters(characters_data)
            n_chars = len(characters_data) if isinstance(characters_data, list) else 1
            print(f"  ✓ 角色已保存: {pm.characters_path}")
            print(f"  角色数: {n_chars}")

        pm.save_config()
        print(f"  💾 项目配置已保存: {pm.root / 'project.yaml'}")
        print(f"项目初始化完成！接下来运行:")
        print(f"  python gen_series.py --project {pm.root} --chapters 1-12")
        return

    # === 生成章节 ===
    # 加载大纲和角色
    outline_text, outline_data = pm.load_outline()
    characters_text = pm.load_characters()

    if not outline_text or not outline_data:
        print("错误: 项目中缺少 outline.json（先用 --prepare --idea '...' 初始化）")
        sys.exit(1)
    if not characters_text:
        print("错误: 项目中缺少 characters.json（先用 --prepare --idea '...' 初始化）")
        sys.exit(1)

    if not args.chapters:
        print("提示: 请指定 --chapters '1-12' 或 --prepare 来使用 gen_series.py")
        sys.exit(0)

    chapters = parse_chapters(args.chapters)
    print(f"生成章节: {chapters}")

    client = get_llm_client(cfg.llm)

    for ch_num in chapters:
        # 跳过已有章节
        if pm.chapter_exists(ch_num):
            print(f"  跳过第{ch_num}章（已有文件）")
            continue

        # 从上一章加载状态
        if ch_num > 1:
            prev_state = pm.load_state(ch_num - 1)
            char_state = prev_state["char_state"]
            foreshadow_hooks = prev_state["foreshadow"]
            nexus_history = prev_state["nexus_history"]
        else:
            char_state = {"active": {}, "inactive": {}}
            foreshadow_hooks = []
            nexus_history = ""

        # 重建 ForeshadowTracker
        tracker = ForeshadowTracker()
        tracker.hooks = foreshadow_hooks

        # 重建 NexusSum（注入已有历史）
        nexus = NexusSum(
            str(pm.root / "state"),
            max_length=inference_cfg.nexus_sum.max_length,
            llm_cfg=cfg.llm,
            compress_max_tokens=inference_cfg.nexus_sum.max_tokens)
        nexus.history = nexus_history

        # 获取章节信息
        chapter_title = ""
        if outline_data and "chapters" in outline_data:
            for ch in outline_data["chapters"]:
                if ch.get("chapter") == ch_num:
                    chapter_title = ch.get("title", "")
                    break

        print(f"\n{'='*50}")
        print(f"创作第 {ch_num} 章"
              f"{'「' + chapter_title + '」' if chapter_title else ''}...")

        # 构建 running summary
        running_summary = nexus_history or "（无前文，本章为开篇）"
        char_info = format_active_characters(char_state)
        if char_info:
            running_summary += f"\n\n{char_info}"

        # 伏笔注入
        pending_hooks = tracker.format_open_hooks(ch_num)
        if pending_hooks:
            open_count = len([h for h in tracker.hooks
                              if h.get("status") == "open"])
            print(f"  📌 未回收伏笔: {open_count} 个")

        # 生成章节
        result = engine.generate_chapter(
            outline_text, characters_text,
            chapter_num=ch_num, chapter_title=chapter_title,
            previous_summary=running_summary,
            outline_parsed=outline_data,
            pending_hooks=pending_hooks,
        )

        content = clean_chapter_output(result.content)
        pm.save_chapter(ch_num, content)
        print(f"  字数: {len(content)}")
        print(f"  已保存: {pm.chapter_path(ch_num)}")

        # 生成摘要 → 更新 NexusSum
        print(f"  生成摘要...")
        summary = summarize_chapter(
            client, cfg.llm, ch_num, content, inference_cfg=inference_cfg)
        nexus.append_chapter(ch_num, summary)
        updated_nexus_history = nexus.get_history()

        # 更新角色状态
        print(f"  更新角色状态...")
        char_state = update_character_state(
            client, cfg.llm, ch_num, content, char_state,
            inference_cfg=inference_cfg)

        # 更新伏笔追踪
        print(f"  更新伏笔追踪...")
        update_foreshadow_hooks(
            client, cfg.llm, ch_num, content, tracker,
            inference_cfg=inference_cfg)

        # 保存本章状态（独立快照）
        pm.save_state(ch_num, char_state, tracker.hooks,
                      updated_nexus_history)
        print(f"  💾 状态已保存: {pm.state_dir(ch_num)}")

    # 打印统计
    latest = pm.get_latest_chapter_num()
    if latest > 0:
        last_state = pm.load_state(latest)
        hooks = last_state["foreshadow"]
        total = len(hooks)
        resolved = len([h for h in hooks if h.get("status") == "resolved"])
        open_h = total - resolved
        print(f"\n伏笔统计: 总计 {total}，未回收 {open_h}，已回收 {resolved}")

    print("完成!")


if __name__ == "__main__":
    main()
