"""工具模块：配置加载、LLM 客户端、文本处理"""

import json
import logging
import os
import re
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class SegmentationConfig:
    method: str = "embedding_similarity"
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.35
    min_scene_length: int = 300
    max_scene_length: int = 2000
    overlap: int = 50


@dataclass
class SummarizationConfig:
    style_decoupling: bool = True
    abstract_level: str = "high"
    max_summary_ratio: float = 0.3
    use_dialogue_transform: bool = True
    scenes_per_chapter: int = 5
    chapters_per_volume: int = 8


@dataclass
class PreprocessingConfig:
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)


@dataclass
class MemoryConfig:
    backend: str = "file"
    path: str = "output/memory_store"
    max_context_tokens: int = 8000
    retrieval_top_k: int = 5


@dataclass
class RewardWeights:
    stylometric: float = 0.30
    authorship_verification: float = 0.35
    llm_judge: float = 0.35


@dataclass
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    stylometric_features: list = field(default_factory=lambda: [
        "ttr", "entropy", "burstiness", "pos_ngram",
        "sentence_length_var", "punctuation_density"
    ])
    llm_judge_dimensions: list = field(default_factory=lambda: [
        "句法结构相似度", "情感节奏匹配度", "词汇偏好对齐度",
        "叙事视角一致性", "修辞手法运用"
    ])


@dataclass
class OptimizerConfig:
    method: str = "apo"
    max_iterations: int = 20
    convergence_threshold: float = 0.02
    prompt_components: list = field(default_factory=lambda: [
        "system_instruction", "style_rules", "example_demonstrations"
    ])


@dataclass
class PipelineConfig:
    input_dir: str = "data/samples"
    output_dir: str = "output"
    log_level: str = "INFO"


@dataclass
class SearchConfig:
    enabled: bool = True
    mode: str = "llm_select"
    source: str = "auto"
    max_results: int = 3
    api_keys: dict = field(default_factory=dict)


@dataclass
class InferenceStageConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    input_truncate: int = 0  # 0 means no truncation
    target_words: str = ""   # e.g. "3000-5000"


@dataclass
class NexusSumConfig:
    max_length: int = 5000
    max_tokens: int = 2500


@dataclass
class InferenceConfig:
    outline: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=8192, temperature=0.7))
    characters: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=8192, temperature=0.7))
    chapter: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=4096, temperature=0.8, target_words="3000-5000"))
    summarize: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=500, temperature=0.2, input_truncate=5000))
    char_state: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=2000, temperature=0.2))
    foreshadow: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=1000, temperature=0.2, input_truncate=5000))
    nexus_sum: NexusSumConfig = field(default_factory=NexusSumConfig)
    memory_warnings: InferenceStageConfig = field(default_factory=lambda: InferenceStageConfig(max_tokens=2000, temperature=0.3))
    auto_continue: bool = True
    max_continues: int = 2


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """从 YAML 文件加载配置，缺失字段用默认值填充"""
    # 自动加载 .env（如果 python-dotenv 可用）
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(_env_path, override=False)
    except ImportError:
        pass

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    cfg = Config()

    # LLM
    if "llm" in raw:
        llm = raw["llm"]
        cfg.llm = LLMConfig(
            provider=llm.get("provider", "openai"),
            model=llm.get("model", "gpt-4o"),
            api_base=llm.get("api_base"),
            api_key=llm.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            temperature=llm.get("temperature", 0.7),
            max_tokens=llm.get("max_tokens", 4096),
            timeout=llm.get("timeout", 120),
        )

    # Preprocessing
    if "preprocessing" in raw:
        pp = raw["preprocessing"]
        seg = pp.get("segmentation", {})
        summ = pp.get("summarization", {})
        cfg.preprocessing = PreprocessingConfig(
            segmentation=SegmentationConfig(**{k: v for k, v in seg.items() if k in SegmentationConfig.__dataclass_fields__}),
            summarization=SummarizationConfig(**{k: v for k, v in summ.items() if k in SummarizationConfig.__dataclass_fields__}),
        )

    # Memory
    if "memory" in raw:
        mem = raw["memory"]
        cfg.memory = MemoryConfig(**{k: v for k, v in mem.items() if k in MemoryConfig.__dataclass_fields__})

    # Reward
    if "reward" in raw:
        rw = raw["reward"]
        weights = rw.get("weights", {})
        cfg.reward = RewardConfig(
            weights=RewardWeights(**{k: v for k, v in weights.items() if k in RewardWeights.__dataclass_fields__}),
            stylometric_features=rw.get("stylometric", {}).get("features", cfg.reward.stylometric_features),
            llm_judge_dimensions=rw.get("llm_judge", {}).get("dimensions", cfg.reward.llm_judge_dimensions),
        )

    # Optimizer
    if "optimizer" in raw:
        opt = raw["optimizer"]
        cfg.optimizer = OptimizerConfig(**{k: v for k, v in opt.items() if k in OptimizerConfig.__dataclass_fields__})

    # Pipeline
    if "pipeline" in raw:
        pl = raw["pipeline"]
        cfg.pipeline = PipelineConfig(**{k: v for k, v in pl.items() if k in PipelineConfig.__dataclass_fields__})

    # Search
    if "search" in raw:
        sc = raw["search"]
        cfg.search = SearchConfig(
            enabled=sc.get("enabled", True),
            mode=sc.get("mode", "llm_select"),
            source=sc.get("source", "auto"),
            max_results=sc.get("max_results", 3),
            api_keys={
                "tavily": (sc.get("api_keys", {}).get("tavily") or os.environ.get("TAVILY_API_KEY") or ""),
                "exa": (sc.get("api_keys", {}).get("exa") or os.environ.get("EXA_API_KEY") or ""),
            },
        )

    # Inference
    if "inference" in raw:
        inf = raw["inference"]
        def _parse_stage(d: dict) -> InferenceStageConfig:
            return InferenceStageConfig(**{k: v for k, v in d.items() if k in InferenceStageConfig.__dataclass_fields__})

        ic = InferenceConfig()
        for stage_name in ["outline", "characters", "chapter", "summarize", "char_state", "foreshadow", "memory_warnings"]:
            if stage_name in inf:
                setattr(ic, stage_name, _parse_stage(inf[stage_name]))
        if "nexus_sum" in inf:
            ns = inf["nexus_sum"]
            ic.nexus_sum = NexusSumConfig(**{k: v for k, v in ns.items() if k in NexusSumConfig.__dataclass_fields__})
        ic.auto_continue = inf.get("auto_continue", True)
        ic.max_continues = inf.get("max_continues", 2)
        cfg.inference = ic

    return cfg


def get_llm_client(cfg: LLMConfig):
    """根据配置创建 LLM 客户端"""
    from openai import OpenAI

    kwargs = {}
    if cfg.api_base:
        kwargs["base_url"] = cfg.api_base
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key

    return OpenAI(**kwargs)


def llm_call(client, cfg: LLMConfig, messages: list, max_retries: int = 3,
             auto_continue: bool = False, max_continues: int = 2, **kwargs) -> str:
    """统一的 LLM 调用接口，带重试机制和截断续写

    Args:
        auto_continue: 当 finish_reason=="length" 时自动续写
        max_continues: 最多续写次数
    """
    import time
    params = {
        "model": cfg.model,
        "messages": messages,
        "temperature": kwargs.get("temperature", cfg.temperature),
        "max_tokens": kwargs.get("max_tokens", cfg.max_tokens),
        "timeout": cfg.timeout,
    }
    params.update({k: v for k, v in kwargs.items() if k not in params})

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**params)
            if not response.choices:
                raise RuntimeError(f"LLM returned empty choices (finish_reason={getattr(response, 'finish_reason', 'unknown')}): {str(response)[:300]}")

            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], 'finish_reason', None)

            # Auto-continue on truncation
            if auto_continue and finish_reason == "length" and max_continues > 0:
                log.info(f"LLM output truncated (finish_reason=length), auto-continuing ({max_continues} remaining)...")
                cont_messages = list(messages) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "请继续，从中断处接着写，不要重复已写内容"},
                ]
                continuation = llm_call(
                    client, cfg, cont_messages, max_retries=max_retries,
                    auto_continue=True, max_continues=max_continues - 1, **kwargs
                )
                return content + continuation

            return content
        except Exception as e:
            last_err = e
            err_str = str(e)
            # 可重试的错误类型
            retryable = any(s in err_str for s in [
                "SSL", "Connection", "Timeout", "timeout",
                "UNEXPECTED_EOF", "ConnectionReset", "RemoteDisconnected",
                "APIConnectionError", "RateLimitError", "502", "503", "529",
                "JSONDecodeError", "Expecting value",
                "NoneType", "empty choices",
            ])
            if not retryable or attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            log.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {err_str[:120]}")
            time.sleep(wait)
    raise last_err


def llm_embed(client, cfg: LLMConfig, texts: list[str]) -> list[list[float]]:
    """批量文本嵌入"""
    from openai import OpenAI

    embedding_model = "text-embedding-3-small"
    resp = client.embeddings.create(input=texts, model=embedding_model)
    return [d.embedding for d in resp.data]


def extract_json_from_text(text: str, expect_list: bool = False) -> dict | list | None:
    """从 LLM 输出中稳健提取 JSON（清理 markdown 代码块再解析）"""
    # 1. 清理 markdown 代码块标记
    cleaned = text
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)

    # 2. 尝试正则匹配 + 解析
    if expect_list:
        pattern = r'\[[\s\S]*\]'
    else:
        pattern = r'\{[\s\S]*\}'

    try:
        match = re.search(pattern, cleaned)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. 需要数组但没找到 []，尝试找多个独立 {} 对象
    if expect_list:
        objects = []
        # 找所有 { ... } 块
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    objects.append(obj)
            except (json.JSONDecodeError, ValueError):
                continue
        if objects:
            return objects

    # 4. 逐行找 JSON 起点（兜底）
    for line_start in ['[', '{']:
        idx = cleaned.find(line_start)
        if idx >= 0:
            try:
                return json.loads(cleaned[idx:])
            except (json.JSONDecodeError, ValueError):
                pass

    return None
