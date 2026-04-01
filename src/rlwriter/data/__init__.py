"""Dataset & DataLoader 模块

将小说 txt 文件夹转化为 RL 训练所需的 (input, label) 样本。

数据流：
  文件夹/*.txt → 加载 → 场景分割 → TrainingSample 列表 → train/val split

每个 TrainingSample：
  - input_text: 场景摘要/情节骨架（预处理阶段由 LLM 生成）
  - label_text: 原始场景正文（用于奖励计算的 ground truth）
  - scene_index: 场景在小说中的位置
  - novel_path: 来源小说路径
  - context: 前序场景摘要（用于叙事连贯性）
"""

import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrainingSample:
    """单个训练样本"""
    input_text: str          # 场景摘要 / 情节骨架（label 的压缩表示）
    label_text: str          # 原始场景正文（ground truth）
    scene_index: int         # 场景在小说中的序号
    novel_path: str          # 来源小说文件路径
    context: str = ""        # 前序场景摘要（叙事连贯性上下文）
    summary: str = ""        # 风格剥离摘要（LLM 预处理后填充）

    @property
    def sample_id(self) -> str:
        """唯一标识：基于小说路径 + 场景索引"""
        key = f"{self.novel_path}:{self.scene_index}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NovelData:
    """一部小说的完整数据"""
    path: str
    text: str
    scenes: list[str] = field(default_factory=list)
    samples: list[TrainingSample] = field(default_factory=list)


class NovelDataset:
    """小说数据集

    用法：
        dataset = NovelDataset.from_folder("data/novels")
        train_set, val_set = dataset.split(0.8)
        for sample in train_set:
            print(sample.input_text, sample.label_text)
    """

    def __init__(self):
        self.novels: list[NovelData] = []
        self.samples: list[TrainingSample] = []

    @classmethod
    def from_folder(cls, folder_path: str, min_scene_length: int = 200,
                    max_scene_length: int = 3000) -> "NovelDataset":
        """从文件夹加载所有 .txt 小说"""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")

        txt_files = sorted(folder.glob("**/*.txt"))
        if not txt_files:
            raise ValueError(f"文件夹中没有 .txt 文件: {folder_path}")

        ds = cls()
        for txt_file in txt_files:
            novel = cls._load_novel(txt_file, min_scene_length, max_scene_length)
            if novel and novel.scenes:
                ds.novels.append(novel)
                ds.samples.extend(novel.samples)

        return ds

    @classmethod
    def from_single(cls, file_path: str, min_scene_length: int = 200,
                    max_scene_length: int = 3000) -> "NovelDataset":
        """从单个小说文件加载"""
        ds = cls()
        novel = cls._load_novel(Path(file_path), min_scene_length, max_scene_length)
        if novel and novel.scenes:
            ds.novels.append(novel)
            ds.samples.extend(novel.samples)
        return ds

    @staticmethod
    def load_text(file_path: Path) -> Optional[str]:
        """加载文本文件，自动检测编码"""
        for encoding in ["utf-8", "gb18030", "gbk", "gb2312", "latin-1"]:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    @classmethod
    def _load_novel(cls, file_path: Path, min_scene_length: int,
                    max_scene_length: int) -> Optional[NovelData]:
        """加载并处理单部小说"""
        text = cls.load_text(file_path)
        if not text or len(text.strip()) < min_scene_length:
            return None

        scenes = cls.segment_scenes(text, min_scene_length, max_scene_length)
        samples = cls._create_samples(scenes, str(file_path))

        return NovelData(
            path=str(file_path),
            text=text,
            scenes=scenes,
            samples=samples,
        )

    @staticmethod
    def segment_scenes(text: str, min_length: int = 200,
                       max_length: int = 3000) -> list[str]:
        """将小说文本切分为场景

        切分策略：
        1. 按空行分割段落
        2. 合并段落直到达到最小长度
        3. 超过最大长度时在句号处拆分
        """
        # 按空行切段落
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

        if not paragraphs:
            return [text.strip()] if text.strip() else []

        # 合并段落为场景
        scenes = []
        buffer = ""

        for para in paragraphs:
            candidate = buffer + "\n\n" + para if buffer else para
            if len(candidate) >= min_length:
                scenes.append(candidate.strip())
                buffer = ""
            else:
                buffer = candidate

        # 处理剩余
        if buffer.strip():
            if scenes and len(scenes[-1]) + len(buffer) < max_length:
                scenes[-1] = scenes[-1] + "\n\n" + buffer.strip()
            else:
                scenes.append(buffer.strip())

        # 拆分过长场景
        final = []
        for scene in scenes:
            if len(scene) > max_length:
                chunks = NovelDataset._split_long(scene, max_length)
                final.extend(chunks)
            else:
                final.append(scene)

        return final

    @staticmethod
    def _split_long(text: str, max_length: int) -> list[str]:
        """在句号处拆分过长文本"""
        sentences = re.split(r'([。！？])', text)
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            reconstructed.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            reconstructed.append(sentences[-1])

        chunks = []
        current = ""
        for s in reconstructed:
            if len(current) + len(s) > max_length and current:
                chunks.append(current.strip())
                current = s
            else:
                current += s
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [text]

    @staticmethod
    def _create_samples(scenes: list[str], novel_path: str) -> list[TrainingSample]:
        """从场景列表创建训练样本

        每个样本：
        - input_text: 场景前 100 字（占位，预处理阶段替换为 LLM 摘要）
        - label_text: 完整场景正文
        - context: 前一个场景的前 100 字
        """
        samples = []
        for i, scene in enumerate(samples := []):
            pass  # 占位，下面重写

        samples = []
        for i, scene in enumerate(scenes):
            # input_text 占位：取前 100 字作为原始骨架
            # 实际训练时由 LLM 生成风格剥离摘要来替换
            input_text = scene[:100] + "..." if len(scene) > 100 else scene

            # context: 前一个场景的摘要
            context = ""
            if i > 0:
                prev = scenes[i - 1]
                context = prev[:100] + "..." if len(prev) > 100 else prev

            samples.append(TrainingSample(
                input_text=input_text,
                label_text=scene,
                scene_index=i,
                novel_path=novel_path,
                context=context,
            ))

        return samples

    def split(self, train_ratio: float = 0.8,
              shuffle: bool = False, seed: int = 42) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """划分训练集和验证集

        Args:
            train_ratio: 训练集比例
            shuffle: 是否打乱
            seed: 随机种子

        Returns:
            (train_samples, val_samples)
        """
        samples = list(self.samples)

        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        return samples[:split_idx], samples[split_idx:]

    def generate_summaries(self, llm_call_fn, llm_cfg) -> None:
        """用 LLM 为所有样本生成风格剥离摘要（替换占位 input_text）

        Args:
            llm_call_fn: llm_call(client, cfg, messages, **kwargs) -> str
            llm_cfg: LLM 配置
        """
        prompt_template = """你是一位专业的小说情节分析师。请对以下小说文本进行情节摘要。

**核心要求：**
1. **剥离风格**：不要保留原文的修辞手法、句式结构、特定词汇
2. **保留骨架**：记录所有关键情节点、人物行为、对话要点、环境要素
3. **客观叙述**：用平实的白话文描述，如同向他人转述故事情节

**原文：**

{text}

---

**情节摘要："""

        # 这里需要 client，但 dataset 不应该持有 client
        # 由外部 pipeline/trainer 调用时传入
        for sample in self.samples:
            if sample.summary:  # 已有摘要则跳过
                continue
            prompt = prompt_template.format(text=sample.label_text[:2000])
            # 摘要生成由调用方负责
            sample.summary = prompt  # 占位，实际由外部填充

    def get_novel_samples(self, novel_path: str) -> list[TrainingSample]:
        """获取指定小说的所有样本"""
        return [s for s in self.samples if s.novel_path == novel_path]

    def save(self, output_path: str) -> None:
        """保存数据集到 JSON"""
        data = {
            "num_novels": len(self.novels),
            "num_samples": len(self.samples),
            "novels": [
                {"path": n.path, "num_scenes": len(n.scenes)}
                for n in self.novels
            ],
            "samples": [s.to_dict() for s in self.samples],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "NovelDataset":
        """从 JSON 加载数据集"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ds = cls()
        for s in data["samples"]:
            ds.samples.append(TrainingSample(**s))
        return ds

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (f"<NovelDataset: {len(self.novels)} novels, "
                f"{len(self.samples)} samples>")
