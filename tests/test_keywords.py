"""测试：关键词提取与记忆检索"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))


def test_jieba_extract_tags():
    """jieba TF-IDF 提取关键词"""
    import jieba.analyse
    text = "环境描写堆砌了过多景物细节，应该只保留与情节相关的关键意象，韦小宝的视角受限，不应描写他看不到的东西"
    keywords = jieba.analyse.extract_tags(text, topK=5)
    assert len(keywords) > 0
    assert all(len(kw) > 1 for kw in keywords)
    # 验证提取出有意义的词，不是标点或单字
    print(f"  关键词: {keywords}")


def test_keywords_trigger_search():
    """验证提取的关键词能触发 memory search"""
    from rlwriter.memory import StyleMemory, StyleNode
    from rlwriter.utils import MemoryConfig

    cfg = MemoryConfig(path="/tmp/test_kw_memory", retrieval_top_k=3)
    mem = StyleMemory(cfg, persistent=False)

    # 添加一个节点，用 jieba 提取的关键词
    import jieba.analyse
    gradient = "韦小宝的内心独白过多，应保持有限视角，减少心理描写"
    keywords = jieba.analyse.extract_tags(gradient, topK=5)

    node = StyleNode(
        node_id="test_kw_1",
        content=gradient,
        context_tags=["视角", "心理描写"],
        trigger_keywords=keywords,
    )
    mem.add_node(node)

    # 用包含相关词的文本搜索
    results = mem.search("韦小宝站在皇宫门前，心里想着天地会的事")
    assert len(results) > 0, f"搜索未命中，keywords={keywords}"
    print(f"  命中节点: {results[0].content[:30]}...")

    # 用不相关的文本搜索
    results2 = mem.search("一阵风吹过窗外的柳树")
    # 可能命中也可能不命中，但得分应该更低


def test_keywords_not_truncated():
    """验证关键词不会被截断成半截"""
    import jieba.analyse
    text = "环境描写冗余，每个场景只聚焦1-2个关键细节，避免景物堆砌"
    keywords = jieba.analyse.extract_tags(text, topK=5)
    for kw in keywords:
        assert len(kw) >= 2, f"关键词太短: '{kw}'"
        # jieba 不会产生截断的词
        assert not any(c in kw for c in "，。、"), f"关键词含标点: '{kw}'"


if __name__ == "__main__":
    test_jieba_extract_tags()
    test_keywords_trigger_search()
    test_keywords_not_truncated()
    print("\n🎉 全部 3 个关键词测试通过")
