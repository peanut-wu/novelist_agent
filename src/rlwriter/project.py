"""小说项目文件夹管理"""
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ProjectConfig:
    title: str = ""
    idea: str = ""
    checkpoint: str = ""
    config: str = "config/deepseek.yaml"
    created_at: str = ""


class ProjectManager:
    """管理小说项目的文件读写和路径解析"""

    def __init__(self, project_dir: str):
        self.root = Path(project_dir)
        self.config = self._load_config()

    def _load_config(self) -> ProjectConfig:
        """加载 project.yaml"""
        config_path = self.root / "project.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return ProjectConfig(
                **{k: v for k, v in raw.items()
                   if k in ProjectConfig.__dataclass_fields__}
            )
        return ProjectConfig()

    def save_config(self):
        """保存 project.yaml"""
        config_path = self.root / "project.yaml"
        self.root.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self.config), f,
                      allow_unicode=True, default_flow_style=False)

    def ensure_dirs(self):
        """确保项目目录结构存在"""
        (self.root / "chapters").mkdir(parents=True, exist_ok=True)
        (self.root / "state").mkdir(parents=True, exist_ok=True)

    # === 路径方法 ===

    @property
    def outline_path(self) -> Path:
        return self.root / "outline.json"

    @property
    def characters_path(self) -> Path:
        return self.root / "characters.json"

    def chapter_path(self, chapter_num: int) -> Path:
        return self.root / "chapters" / f"ch{chapter_num:03d}.txt"

    def state_dir(self, chapter_num: int) -> Path:
        return self.root / "state" / f"ch{chapter_num:03d}"

    def char_state_path(self, chapter_num: int) -> Path:
        return self.state_dir(chapter_num) / "char_state.json"

    def foreshadow_path(self, chapter_num: int) -> Path:
        return self.state_dir(chapter_num) / "foreshadow.json"

    def nexus_path(self, chapter_num: int) -> Path:
        return self.state_dir(chapter_num) / "nexus.json"

    # === 数据读写 ===

    def load_outline(self) -> tuple[str, dict]:
        """加载大纲，返回 (原始文本, 解析后的 dict)"""
        if not self.outline_path.exists():
            return "", {}
        data = json.loads(self.outline_path.read_text(encoding="utf-8"))
        # 构建文本版本用于注入 prompt
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return text, data

    def save_outline(self, data: dict):
        """保存大纲 JSON"""
        self.outline_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def load_characters(self) -> str:
        """加载角色设定文本"""
        if not self.characters_path.exists():
            return ""
        data = json.loads(self.characters_path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)

    def save_characters(self, data):
        """保存角色设定 JSON"""
        self.characters_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def load_chapter(self, chapter_num: int) -> Optional[str]:
        """加载章节正文，不存在返回 None"""
        path = self.chapter_path(chapter_num)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def save_chapter(self, chapter_num: int, content: str):
        """保存章节正文"""
        path = self.chapter_path(chapter_num)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def load_state(self, chapter_num: int) -> dict:
        """加载指定章节后的完整状态"""
        state = {
            "char_state": {"active": {}, "inactive": {}},
            "foreshadow": [],
            "nexus_history": "",
        }

        cs_path = self.char_state_path(chapter_num)
        if cs_path.exists():
            state["char_state"] = json.loads(
                cs_path.read_text(encoding="utf-8"))

        fs_path = self.foreshadow_path(chapter_num)
        if fs_path.exists():
            state["foreshadow"] = json.loads(
                fs_path.read_text(encoding="utf-8"))

        nx_path = self.nexus_path(chapter_num)
        if nx_path.exists():
            state["nexus_history"] = json.loads(
                nx_path.read_text(encoding="utf-8")).get("history", "")

        return state

    def save_state(self, chapter_num: int, char_state: dict,
                   foreshadow_hooks: list, nexus_history: str):
        """保存指定章节后的完整状态"""
        state_dir = self.state_dir(chapter_num)
        state_dir.mkdir(parents=True, exist_ok=True)

        self.char_state_path(chapter_num).write_text(
            json.dumps(char_state, ensure_ascii=False, indent=2),
            encoding="utf-8")

        self.foreshadow_path(chapter_num).write_text(
            json.dumps(foreshadow_hooks, ensure_ascii=False, indent=2),
            encoding="utf-8")

        self.nexus_path(chapter_num).write_text(
            json.dumps({"history": nexus_history}, ensure_ascii=False,
                       indent=2),
            encoding="utf-8")

    def get_latest_chapter_num(self) -> int:
        """获取已生成的最大章节号"""
        chapters_dir = self.root / "chapters"
        if not chapters_dir.exists():
            return 0
        nums = []
        for f in chapters_dir.glob("ch*.txt"):
            try:
                nums.append(int(f.stem.replace("ch", "")))
            except ValueError:
                pass
        return max(nums) if nums else 0

    def chapter_exists(self, chapter_num: int) -> bool:
        return self.chapter_path(chapter_num).exists()
