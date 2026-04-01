"""串联生成工具函数：摘要/角色状态/伏笔/后处理"""

import json
import re
from typing import Optional

from ..utils import LLMConfig, llm_call
from ..foreshadow_tracker import ForeshadowTracker


# ========== Prompts ==========

SUMMARIZE_PROMPT = """请用 200-300 字概括以下小说章节的核心事件，重点保留：
1. 人物互动（谁见了谁、说了什么关键话）
2. 新获得的物品/信息
3. 人物位置变化
4. 未解之谜/伏笔

**第 {chapter_num} 章原文：**
{text}

**概括："""


UPDATE_CHAR_STATE_PROMPT = """根据以下章节内容，更新角色状态追踪表。

**当前活跃角色（本章出场）：**
{active_state}

**当前休眠角色（未出场，仅保留基本信息）：**
{inactive_state}

**第 {chapter_num} 章内容：**
{text}

请完成：
1. 识别本章出场的角色 → 更新为活跃状态
2. 活跃角色中未在本章出场的 → 降级为休眠（仅保留 name + last_seen）
3. 休眠角色在本章回归的 → 移回活跃

**输出格式（JSON）：**
{{
  "active": {{
    "角色名": {{
      "location": "当前位置",
      "last_seen": "最后场景描述",
      "key_events": ["本章关键事件"],
      "possessions": ["持有物品"],
      "relationships_update": {{"与某人": "关系变化"}}
    }}
  }},
  "inactive": {{
    "角色名": {{
      "name": "角色名",
      "last_seen": "最后出场章节和场景"
    }}
  }}
}}

注意：每个角色的 key_events 不超过 2 条，possessions 不超过 3 个，relationships_update 不超过 2 条。
只输出 JSON，不要其他内容。"""


UPDATE_HOOKS_PROMPT = """根据以下小说章节内容，检查伏笔的埋设与回收。

**当前未回收伏笔：**
{open_hooks}

**第 {chapter_num} 章内容：**
{text}

请完成两项任务：

1. **检查回收**：本章是否回收了上述任何伏笔？用关键词匹配（如"铁牌"匹配"铁牌上的云雷纹"）。
2. **检查新伏笔**：本章是否埋设了新的伏笔？

**输出格式（JSON）：**
{{
  "resolved": ["关键词1", "关键词2"],
  "new_hooks": [
    {{"description": "伏笔描述", "importance": "high/medium/low", "technique": "手法"}}
  ]
}}

只输出 JSON，不要其他内容。如果没有回收或新伏笔，对应字段为空数组。"""


# ========== 工具函数 ==========

def summarize_chapter(client, llm_cfg: LLMConfig, chapter_num: int, text: str,
                      inference_cfg=None) -> str:
    """用 LLM 生成章节摘要"""
    cfg = inference_cfg.summarize if inference_cfg else None
    truncate = cfg.input_truncate if cfg and cfg.input_truncate else 5000
    max_tok = cfg.max_tokens if cfg else 500
    temp = cfg.temperature if cfg else 0.2
    prompt = SUMMARIZE_PROMPT.format(chapter_num=chapter_num, text=text[:truncate])
    messages = [{"role": "user", "content": prompt}]
    try:
        return llm_call(client, llm_cfg, messages, temperature=temp, max_tokens=max_tok)
    except Exception as e:
        print(f"  ⚠️ 摘要生成失败: {e}")
        return text[:500]


def update_character_state(client, llm_cfg: LLMConfig, chapter_num: int,
                            text: str, char_state: dict,
                            inference_cfg=None) -> dict:
    """用 LLM 更新角色冷热状态"""
    cfg = inference_cfg.char_state if inference_cfg else None
    max_tok = cfg.max_tokens if cfg else 4000
    temp = cfg.temperature if cfg else 0.3
    active_state = json.dumps(char_state.get("active", {}), ensure_ascii=False, indent=2) or "{}"
    inactive_state = json.dumps(char_state.get("inactive", {}), ensure_ascii=False, indent=2) or "{}"

    # 超长时先摘要（保留对话和关键信息）
    input_text = text
    if len(text) > 30000:
        try:
            summary_prompt = f"请用 3000 字概括以下小说章节的所有角色交互和状态变化，保留对话细节：\n\n{text}"
            input_text = llm_call(client, llm_cfg, [{"role": "user", "content": summary_prompt}],
                                  temperature=0.2, max_tokens=4000)
        except Exception:
            input_text = text[-30000:]  # fallback: 取最后部分

    prompt = UPDATE_CHAR_STATE_PROMPT.format(
        active_state=active_state,
        inactive_state=inactive_state,
        chapter_num=chapter_num,
        text=input_text,
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = llm_call(client, llm_cfg, messages, temperature=temp, max_tokens=max_tok)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            raw = json_match.group()
            # 尝试解析，如果截断则补全
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 补全未闭合的括号
                depth = 0
                for ch in raw:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                if depth > 0:
                    raw += '}' * depth
                    data = json.loads(raw)
                else:
                    raise
            if "active" in data or "inactive" in data:
                return data
    except Exception as e:
        print(f"  ⚠️ 角色状态更新失败: {e}")
    return char_state


def update_foreshadow_hooks(client, llm_cfg: LLMConfig, chapter_num: int,
                             text: str, tracker: ForeshadowTracker,
                             inference_cfg=None):
    """用 LLM 检查本章的伏笔回收和新埋设"""
    cfg = inference_cfg.foreshadow if inference_cfg else None
    truncate = cfg.input_truncate if cfg and cfg.input_truncate else 5000
    max_tok = cfg.max_tokens if cfg else 1500
    temp = cfg.temperature if cfg else 0.3

    open_hooks = tracker.get_hooks_for_chapter(chapter_num)
    if not open_hooks:
        hooks_text = "（无）"
    else:
        hooks_text = "\n".join(f"- [{h['hook_id']}] {h['description']}" for h in open_hooks)

    prompt = UPDATE_HOOKS_PROMPT.format(
        open_hooks=hooks_text,
        chapter_num=chapter_num,
        text=text[:truncate],
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = llm_call(client, llm_cfg, messages, temperature=temp, max_tokens=max_tok)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            for keyword in data.get("resolved", []):
                result = tracker.resolve_hook(keyword, chapter_num)
                if result:
                    print(f"  🎯 回收伏笔: {result['description']}")
            for hook in data.get("new_hooks", []):
                tracker.add_hook(
                    description=hook.get("description", ""),
                    chapter=chapter_num,
                    importance=hook.get("importance", "medium"),
                    technique=hook.get("technique", ""),
                )
                print(f"  🌱 新伏笔: {hook.get('description', '')}")
    except Exception as e:
        print(f"  ⚠️ 伏笔更新失败: {e}")


def format_active_characters(char_state: dict) -> str:
    """格式化活跃角色状态，用于注入 running summary"""
    active = char_state.get("active", {})
    if not active:
        return ""
    lines = ["**活跃角色状态：**"]
    for name, info in active.items():
        events = info.get("key_events", [])
        loc = info.get("location", "?")
        event_str = "；".join(events[:2]) if events else "无关键事件"
        lines.append(f"- **{name}**（{loc}）：{event_str}")
    return "\n".join(lines)


def clean_chapter_output(text: str) -> str:
    """清理章节末尾的元分析文字"""
    markers = [
        "\n---\n\n**本章关键点",
        "\n---\n\n**关键点收束",
        "\n---\n\n**叙事节奏",
        "\n---\n\n**文风贯彻",
        "\n**本章关键点",
        "\n**关键点收束",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break

    text = text.rstrip()
    if text.endswith("（本章完）"):
        text = text[:-6].rstrip()
    if text.endswith("(本章完)"):
        text = text[:-6].rstrip()

    return text


def extract_outline_data(outline_path: str) -> dict:
    """尝试从大纲文件中解析 JSON"""
    from pathlib import Path
    text = Path(outline_path).read_text(encoding="utf-8")
    try:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return {}
