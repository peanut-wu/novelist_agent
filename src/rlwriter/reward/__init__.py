"""多维奖励模型：计算文体学 + 作者身份验证 + LLM 裁判"""

import json
import re
import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import jieba
import jieba.posseg as pseg

from ..utils import RewardConfig, LLMConfig, get_llm_client, llm_call


@dataclass
class StylometricFeatures:
    """计算文体学特征集"""
    ttr: float = 0.0                # 类符/形符比
    log_ttr: float = 0.0            # 对数 TTR
    hapax_ratio: float = 0.0        # 绝对一次词比例
    char_entropy: float = 0.0       # 字符级熵
    word_entropy: float = 0.0       # 词级熵
    sentence_entropy: float = 0.0   # 句子长度熵
    burstiness: float = 0.0         # 爆发性（句子长度方差）
    avg_sentence_len: float = 0.0   # 平均句长
    punctuation_density: float = 0.0  # 标点密度
    pos_ngram_dist: dict = None     # 词性 N-gram 分布

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "pos_ngram_dist"}
        d["pos_ngram_dist"] = self.pos_ngram_dist or {}
        return d


class StylometricAnalyzer:
    """计算文体学分析器（针对中文文本）"""

    PUNCTUATION = set('，。！？、；：""''（）《》【】…—～·')

    def analyze(self, text: str) -> StylometricFeatures:
        """提取完整的文体学特征"""
        # 分词
        words = list(jieba.cut(text))
        words_no_punct = [w for w in words if w.strip() and w not in self.PUNCTUATION]

        # 句子切分
        sentences = re.split(r'[。！？]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

        # 词性标注
        pos_tags = [flag for _, flag in pseg.cut(text)]

        features = StylometricFeatures()
        features.ttr = self._ttr(words_no_punct)
        features.log_ttr = self._log_ttr(words_no_punct)
        features.hapax_ratio = self._hapax_ratio(words_no_punct)
        features.char_entropy = self._char_entropy(text)
        features.word_entropy = self._word_entropy(words_no_punct)
        features.sentence_entropy = self._sentence_entropy(sentences)
        features.burstiness = self._burstiness(sentences)
        features.avg_sentence_len = np.mean([len(s) for s in sentences]) if sentences else 0
        features.punctuation_density = self._punctuation_density(text)
        features.pos_ngram_dist = self._pos_ngrams(pos_tags, n=2)

        return features

    def _ttr(self, words: list[str]) -> float:
        """类符/形符比"""
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _log_ttr(self, words: list[str]) -> float:
        """对数 TTR"""
        if not words:
            return 0.0
        return math.log(len(set(words))) / math.log(len(words)) if len(words) > 1 else 0.0

    def _hapax_ratio(self, words: list[str]) -> float:
        """绝对一次词比例"""
        if not words:
            return 0.0
        freq = Counter(words)
        hapax = sum(1 for v in freq.values() if v == 1)
        return hapax / len(set(words))

    def _char_entropy(self, text: str) -> float:
        """字符级信息熵"""
        chars = [c for c in text if c.strip()]
        if not chars:
            return 0.0
        freq = Counter(chars)
        total = len(chars)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _word_entropy(self, words: list[str]) -> float:
        """词级信息熵"""
        if not words:
            return 0.0
        freq = Counter(words)
        total = len(words)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _sentence_entropy(self, sentences: list[str]) -> float:
        """句子长度分布熵"""
        if not sentences:
            return 0.0
        lengths = [len(s) for s in sentences]
        freq = Counter(lengths)
        total = len(lengths)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _burstiness(self, sentences: list[str]) -> float:
        """爆发性：句子长度的方差"""
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s) for s in sentences]
        return float(np.var(lengths))

    def _punctuation_density(self, text: str) -> float:
        """标点符号密度"""
        if not text:
            return 0.0
        punct_count = sum(1 for c in text if c in self.PUNCTUATION)
        return punct_count / len(text)

    def _pos_ngrams(self, pos_tags: list[str], n: int = 2) -> dict:
        """词性 N-gram 分布"""
        if len(pos_tags) < n:
            return {}
        ngrams = [tuple(pos_tags[i:i + n]) for i in range(len(pos_tags) - n + 1)]
        freq = Counter(ngrams)
        total = sum(freq.values())
        # 返回 top 20
        top = freq.most_common(20)
        return {"_".join(k): round(v / total, 4) for k, v in top}


class RewardModel:
    """多维奖励模型

    三部分加权求和：
    1. 计算文体学指标对比（生成文本 vs 原文）
    2. 作者身份验证模型
    3. LLM 裁判评估
    """

    JUDGE_PROMPT = """你是一位专业的小说文风评审专家，同时评估写作质量与情节架构。请从以下维度评估【续写文本】对【原文风格】的模仿程度。

**风格维度（0-10分）：**
1. 句法结构相似度：句式长短搭配、从句使用、排比等修辞
2. 情感节奏匹配度：情绪起伏、紧张缓急的节奏
3. 词汇偏好对齐度：常用词汇、专有搭配、口头禅
4. 叙事视角一致性：第一/三人称、全知/受限视角
5. 修辞手法运用：比喻、拟人、反讽等手法的使用

**情节架构维度（0-10分）：**
6. 节拍结构合理性：起承转合的节拍分布是否自然
7. 张力曲线连贯性：紧张-缓急的起伏是否符合叙事节奏
8. 角色动机一致性：角色行为是否符合其动机和性格
9. 叙事逻辑自洽性：事件发展是否前后连贯、无矛盾

**【原文（前500字）】**
{original}

**【续写文本】**
{generated}

**【输出格式】**
请用 JSON 格式输出全部9个维度，例如：
{{"句法结构相似度": 7, "情感节奏匹配度": 6, "词汇偏好对齐度": 8, "叙事视角一致性": 9, "修辞手法运用": 5, "节拍结构合理性": 7, "张力曲线连贯性": 6, "角色动机一致性": 8, "叙事逻辑自洽性": 7}}"""

    AUTHORSHIP_PROMPT = """你是一位语言学取证专家。请判断【测试文本】是否可能出自与【参考文本】同一位作者之手。

**判断依据：**
- 句式习惯（平均句长、从句频率）
- 词汇选择偏好（高频词、虚词使用）
- 标点使用习惯
- 段落组织方式
- 情感表达模式

**【参考文本】**
{reference}

**【测试文本】**
{generated}

**请输出：**
- 同源概率（0-1 小数）
- 置信度（high/medium/low）
- 主要依据（1-2句话）

格式：概率:xx, 置信度:xx, 依据:xxx"""

    def __init__(self, cfg: RewardConfig, llm_cfg: LLMConfig):
        self.cfg = cfg
        self.llm_cfg = llm_cfg
        self.client = get_llm_client(llm_cfg)
        self.analyzer = StylometricAnalyzer()

    def stylometric_reward(self, original: str, generated: str) -> dict:
        """计算文体学奖励：对比原文和生成文本的特征差异

        返回：各特征的差异分数（越小越好，这里转为 0-1 奖励）
        """
        feat_orig = self.analyzer.analyze(original)
        feat_gen = self.analyzer.analyze(generated)

        diffs = {}
        # TTR 差异
        diffs["ttr"] = 1.0 - min(abs(feat_orig.ttr - feat_gen.ttr) / max(feat_orig.ttr, 0.01), 1.0)
        # 熵差异
        diffs["entropy"] = 1.0 - min(abs(feat_orig.word_entropy - feat_gen.word_entropy) / max(feat_orig.word_entropy, 0.01), 1.0)
        # 爆发性差异（对数尺度比较）
        log_orig_burst = math.log1p(feat_orig.burstiness)
        log_gen_burst = math.log1p(feat_gen.burstiness)
        diffs["burstiness"] = 1.0 - min(abs(log_orig_burst - log_gen_burst) / max(log_orig_burst, 0.01), 1.0)
        # 句长差异
        diffs["sentence_length"] = 1.0 - min(abs(feat_orig.avg_sentence_len - feat_gen.avg_sentence_len) / max(feat_orig.avg_sentence_len, 0.01), 1.0)
        # 标点密度差异
        diffs["punctuation"] = 1.0 - min(abs(feat_orig.punctuation_density - feat_gen.punctuation_density) / max(feat_orig.punctuation_density, 0.01), 1.0)
        # POS N-gram Jaccard 相似度
        if feat_orig.pos_ngram_dist and feat_gen.pos_ngram_dist:
            keys_orig = set(feat_orig.pos_ngram_dist.keys())
            keys_gen = set(feat_gen.pos_ngram_dist.keys())
            intersection = len(keys_orig & keys_gen)
            union = len(keys_orig | keys_gen)
            diffs["pos_ngram"] = intersection / max(union, 1)
        else:
            diffs["pos_ngram"] = 0.5

        # 加权平均
        avg = sum(diffs.values()) / max(len(diffs), 1)
        return {"score": avg, "details": diffs}

    def authorship_reward(self, original: str, generated: str) -> dict:
        """作者身份验证奖励"""
        prompt = self.AUTHORSHIP_PROMPT.format(
            reference=original[:1000],
            generated=generated[:1000],
        )
        messages = [{"role": "user", "content": prompt}]
        response = llm_call(self.client, self.llm_cfg, messages, temperature=0.2, max_tokens=500)

        # 解析概率
        prob = 0.5
        try:
            match = re.search(r'概率[:：]\s*([\d.]+)', response)
            if match:
                prob = float(match.group(1))
                prob = max(0.0, min(1.0, prob))
        except (ValueError, IndexError):
            pass

        return {"score": prob, "raw_response": response}

    def llm_judge_reward(self, original: str, generated: str) -> dict:
        """LLM 裁判评估"""
        prompt = self.JUDGE_PROMPT.format(
            original=original[:800],
            generated=generated[:1500],
        )
        messages = [{"role": "user", "content": prompt}]
        response = llm_call(self.client, self.llm_cfg, messages, temperature=0.2, max_tokens=500)

        # 解析 JSON 评分
        scores = {}
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            # 降级：用正则逐项提取
            for dim in self.cfg.llm_judge_dimensions:
                match = re.search(rf'{dim}[：:]\s*(\d+)', response)
                if match:
                    scores[dim] = int(match.group(1))

        # 归一化到 0-1
        if scores:
            avg = sum(scores.values()) / (len(scores) * 10.0)
        else:
            avg = 0.5

        return {"score": avg, "dimensions": scores, "raw_response": response}

    STYLE_DIMENSIONS = {"句法结构相似度", "情感节奏匹配度", "词汇偏好对齐度", "叙事视角一致性", "修辞手法运用"}
    PLOT_DIMENSIONS = {"节拍结构合理性", "张力曲线连贯性", "角色动机一致性", "叙事逻辑自洽性"}

    def compute_reward(self, original: str, generated: str) -> dict:
        """综合奖励计算（风格 + 情节架构）"""
        styl = self.stylometric_reward(original, generated)
        auth = self.authorship_reward(original, generated)
        judge = self.llm_judge_reward(original, generated)

        # 从 LLM 裁判结果中分离风格和情节评分
        judge_dims = judge.get("dimensions", {})
        style_scores = [v for k, v in judge_dims.items() if k in self.STYLE_DIMENSIONS]
        plot_scores = [v for k, v in judge_dims.items() if k in self.PLOT_DIMENSIONS]

        style_judge = sum(style_scores) / (len(style_scores) * 10.0) if style_scores else 0.5
        plot_judge = sum(plot_scores) / (len(plot_scores) * 10.0) if plot_scores else 0.5

        w = self.cfg.weights
        total = (
            w.stylometric * styl["score"] +
            w.authorship_verification * auth["score"] +
            w.llm_judge * style_judge
        )

        return {
            "total": round(total, 4),
            "stylometric": styl,
            "authorship": auth,
            "llm_judge": judge,
            "style_judge_score": round(style_judge, 4),
            "plot_judge_score": round(plot_judge, 4),
            "plot_dimensions": {k: v for k, v in judge_dims.items() if k in self.PLOT_DIMENSIONS},
            "weights": {"stylometric": w.stylometric, "authorship_verification": w.authorship_verification, "llm_judge": w.llm_judge},
        }
