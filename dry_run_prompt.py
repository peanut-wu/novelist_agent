#!/usr/bin/env python3
"""Dry-run: 构建 generate_chapter 的完整 prompt，保存到文件（不调 LLM）"""
import sys, json
sys.path.insert(0, "src")

from pathlib import Path
from rlwriter.inference import InferenceEngine
from rlwriter.utils import load_config
from rlwriter.series.utils import extract_outline_data

cfg = load_config("config/deepseek.yaml")
engine = InferenceEngine.load("output/checkpoints_50", llm_cfg=cfg.llm)

outline_text = Path("output/gen_tangdynasty_outline.md").read_text(encoding="utf-8")
characters_text = Path("output/gen_tangdynasty_characters.md").read_text(encoding="utf-8")
outline_data = extract_outline_data("output/gen_tangdynasty_outline.md")

# 提取第1章信息
chapter_title = ""
chapter_beats = ""
if outline_data and "chapters" in outline_data:
    for ch in outline_data["chapters"]:
        if ch.get("chapter") == 1:
            chapter_title = ch.get("title", "")
            beats = ch.get("beats", [])
            chapter_beats = "\n".join(f"- {b}" for b in beats) if beats else ""
            break

# 构建各组件
style_rules = engine._format_style_rules()
plot_rules = engine._format_plot_rules()
memory_warnings = engine._format_memory_warnings()

# 智能截断大纲
outline_truncated = engine._smart_outline_truncate(outline_text, 1, outline_data)

# 构建完整 user prompt（同 CHAPTER_PROMPT 模板）
user_prompt = engine.CHAPTER_PROMPT.format(
    style_rules=style_rules,
    plot_rules=plot_rules,
    memory_warnings=memory_warnings,
    pending_hooks="",
    search_context="",
    outline=outline_truncated,
    characters=characters_text[:3000],
    chapter_num=1,
    chapter_title=chapter_title or "第1章",
    chapter_beats=chapter_beats or "（无特定情节点）",
    previous_summary="（无前文，本章为开篇）",
)

# 构建 messages（使用新的 _build_messages）
messages = engine._build_messages(user_prompt)

# 保存
tag = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
out = Path(f"output/prompt_{tag}.txt")
sys_instr = engine.style_prompt.get("system_instruction", "")

with open(out, "w", encoding="utf-8") as f:
    f.write(f"=== MESSAGES (tag: {tag}) ===\n")
    f.write(f"共 {len(messages)} 条 message\n\n")
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        f.write(f"{'='*60}\n")
        f.write(f"[{i}] role={role}  ({len(content)} chars)\n")
        f.write(f"{'='*60}\n\n")
        f.write(content)
        f.write(f"\n\n")

    f.write(f"{'='*60}\n")
    f.write(f"=== COMPONENT DETAILS ===\n\n")
    f.write(f"--- system_instruction ({len(sys_instr)} chars) ---\n")
    has_sys_msg = any(m["role"] == "system" for m in messages)
    f.write(f"{'✅ INJECTED as system message' if has_sys_msg else '❌ NOT INJECTED'}\n")
    f.write(f"{sys_instr}\n\n")
    f.write(f"--- style_rules ({len(style_rules)} chars) ---\n{style_rules}\n\n")
    f.write(f"--- plot_rules ({len(plot_rules)} chars) ---\n{plot_rules}\n\n")
    f.write(f"--- memory_warnings ({len(memory_warnings)} chars) ---\n{memory_warnings}\n\n")
    f.write(f"--- memory_warnings_cached ---\n")
    cached = engine.state.memory_warnings
    f.write(f"{'YES (' + str(len(cached)) + ' chars)' if cached else 'NO (fallback to keyword)'}\n")

print(f"✓ 已保存: {out} ({out.stat().st_size} bytes)")
print(f"  messages: {len(messages)} 条")
has_sys = any(m["role"] == "system" for m in messages)
print(f"  system_instruction: {len(sys_instr)} chars — {'✅ 已注入 system message' if has_sys else '❌ 未注入'}")
print(f"  style_rules: {len(style_rules)} chars")
print(f"  plot_rules: {len(plot_rules)} chars")
print(f"  memory_warnings: {len(memory_warnings)} chars — {'缓存' if engine.state.memory_warnings else '关键词回退'}")
