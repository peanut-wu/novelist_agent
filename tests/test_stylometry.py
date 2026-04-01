"""测试：计算文体学特征提取"""

import sys
sys.path.insert(0, "src")

from rlwriter.reward import StylometricAnalyzer


def test_stylometric_features():
    analyzer = StylometricAnalyzer()

    text = """他推开那扇沉重的门，走了进去。
    房间里很暗。窗帘拉得严严实实。
    "有人吗？"他喊了一声。声音在空荡的房间里回荡。
    没有回应。只有墙上的挂钟发出单调的滴答声。
    他慢慢地向前走，每一步都踩在吱嘎作响的木地板上。突然，他看到了桌上的那封信。
    信封已经泛黄，边缘有些破损。他的心跳开始加速。"""

    features = analyzer.analyze(text)

    assert 0 < features.ttr <= 1, f"TTR 应在 (0, 1] 之间: {features.ttr}"
    assert 0 < features.log_ttr <= 1, f"log TTR 应在 (0, 1] 之间: {features.log_ttr}"
    assert 0 <= features.hapax_ratio <= 1, f"hapax ratio 应在 [0, 1] 之间: {features.hapax_ratio}"
    assert features.char_entropy > 0, f"字符熵应 > 0: {features.char_entropy}"
    assert features.word_entropy > 0, f"词级熵应 > 0: {features.word_entropy}"
    assert features.burstiness >= 0, f"爆发性应 >= 0: {features.burstiness}"
    assert features.punctuation_density > 0, f"标点密度应 > 0: {features.punctuation_density}"

    print("✅ 文体学特征提取测试通过")
    print(f"   TTR: {features.ttr:.4f}")
    print(f"   字符熵: {features.char_entropy:.4f}")
    print(f"   词级熵: {features.word_entropy:.4f}")
    print(f"   爆发性: {features.burstiness:.4f}")
    print(f"   标点密度: {features.punctuation_density:.4f}")


def test_feature_comparison():
    """测试两段不同风格文本的特征差异"""
    analyzer = StylometricAnalyzer()

    formal = """基于上述分析框架，本研究提出了一个综合性的解决方案。
    该方案的核心在于将多维度评估指标进行有机整合，
    从而构建一个具有高度鲁棒性的分析系统。"""

    casual = """哎你知道吗，昨天那个事儿太搞笑了。
    我跟你说啊，就是老张那个二货，他居然把钥匙锁车里了！
    然后他就站在停车场傻眼了哈哈哈哈哈。"""

    feat_formal = analyzer.analyze(formal)
    feat_casual = analyzer.analyze(casual)

    # 正式文本通常 TTR 更高
    assert feat_formal.ttr != feat_casual.ttr, "不同风格文本的 TTR 应有差异"
    # 口语文本标点密度通常不同
    assert feat_formal.punctuation_density != feat_casual.punctuation_density

    print("✅ 特征对比测试通过")
    print(f"   正式 TTR: {feat_formal.ttr:.4f} vs 口语 TTR: {feat_casual.ttr:.4f}")


def test_empty_text():
    """测试空文本边界情况"""
    analyzer = StylometricAnalyzer()
    features = analyzer.analyze("")
    assert features.ttr == 0.0
    assert features.char_entropy == 0.0
    print("✅ 空文本边界测试通过")


if __name__ == "__main__":
    test_stylometric_features()
    test_feature_comparison()
    test_empty_text()
    print("\n🎉 全部测试通过")
