"""测试：Dataset & DataLoader"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

from rlwriter.data import TrainingSample, NovelData, NovelDataset


# ========== 测试用的小说文本 ==========

NOVEL_TEXT_1 = """第一章 入城

他推开那扇沉重的门，走了进去。房间里很暗。窗帘拉得严严实实。
"有人吗？"他喊了一声。声音在空荡的房间里回荡。
没有回应。只有墙上的挂钟发出单调的滴答声。
他慢慢地向前走，每一步都踩在吱嘎作响的木地板上。

突然，他看到了桌上的那封信。
信封已经泛黄，边缘有些破损。他的心跳开始加速。
他伸出手，颤抖着拿起了信封。

第二章 真相

信中的内容让他大吃一惊。原来这一切都是一个精心设计的骗局。
他跌坐在椅子上，久久不能回神。
窗外传来了汽车引擎的声音，有人来了。
他匆忙将信塞进口袋，从后门溜了出去。

街道上空无一人。路灯昏黄的光照在他苍白的脸上。
他需要找一个安全的地方，好好想想接下来该怎么办。
口袋里的信像一块烧红的烙铁，灼烧着他的大腿。"""

NOVEL_TEXT_2 = """序幕

那是一个风雨交加的夜晚。闪电撕裂了天空。
老人坐在壁炉旁，手中握着一把生锈的钥匙。
"总有一天，"他喃喃自语，"总有一天会有人来取的。"

第一章 启程

年轻人背着行囊，站在山脚下。面前是一条蜿蜒的小路。
他没有回头。身后已经没有什么值得留恋的了。
风吹过松林，发出沙沙的响声。
他深吸一口气，迈出了第一步。

山路越来越陡。太阳渐渐西沉。
他找到了一处岩洞，决定在这里过夜。
篝火燃起的时候，他想起了老人的话。"""

LONG_NOVEL_TEXT = "\n\n".join([
    f"第{i}章\n\n" + "这是第" + str(i) + "章的内容。" * 50
    for i in range(1, 6)
])


# ========== 测试函数 ==========

def test_training_sample():
    """测试 TrainingSample 数据结构"""
    sample = TrainingSample(
        input_text="主角进入房间",
        label_text="他推开那扇沉重的门...",
        scene_index=0,
        novel_path="/data/novel1.txt",
        context="前一场景的摘要",
    )
    assert sample.input_text == "主角进入房间"
    assert sample.label_text == "他推开那扇沉重的门..."
    assert sample.scene_index == 0
    assert sample.sample_id  # 非空
    assert len(sample.sample_id) == 12

    d = sample.to_dict()
    assert d["input_text"] == "主角进入房间"
    assert d["scene_index"] == 0
    print("✅ TrainingSample 测试通过")


def test_training_sample_id_unique():
    """不同样本应有不同 ID"""
    s1 = TrainingSample("in1", "label1", 0, "novel1.txt")
    s2 = TrainingSample("in2", "label2", 1, "novel1.txt")
    s3 = TrainingSample("in1", "label1", 0, "novel2.txt")
    assert s1.sample_id != s2.sample_id  # 不同 scene_index
    assert s1.sample_id != s3.sample_id  # 不同 novel_path
    print("✅ TrainingSample ID 唯一性测试通过")


def test_segment_scenes_basic():
    """测试基本场景分割"""
    scenes = NovelDataset.segment_scenes(NOVEL_TEXT_1, min_length=50)
    assert len(scenes) >= 1
    assert all(len(s) >= 50 for s in scenes)
    # 重新拼接后应包含原文内容
    combined = "\n\n".join(scenes)
    assert "他推开那扇沉重的门" in combined
    assert "信中的内容让他大吃一惊" in combined
    print(f"✅ 基本场景分割测试通过 ({len(scenes)} 个场景)")


def test_segment_scenes_min_length():
    """测试最短场景长度约束"""
    scenes = NovelDataset.segment_scenes(NOVEL_TEXT_2, min_length=100)
    for scene in scenes:
        assert len(scene) >= 100 or scene == scenes[-1]  # 最后一个可能略短
    print(f"✅ 最短场景长度测试通过")


def test_segment_scenes_max_length():
    """测试最长场景长度约束"""
    scenes = NovelDataset.segment_scenes(LONG_NOVEL_TEXT, min_length=50, max_length=500)
    for scene in scenes:
        assert len(scene) <= 800  # 允许一定弹性（句子不能断）
    print(f"✅ 最长场景长度测试通过 ({len(scenes)} 个场景)")


def test_segment_scenes_empty():
    """测试空文本"""
    assert NovelDataset.segment_scenes("") == []
    assert NovelDataset.segment_scenes("   \n\n  ") == []
    print("✅ 空文本场景分割测试通过")


def test_segment_scenes_short():
    """测试极短文本"""
    scenes = NovelDataset.segment_scenes("只有一句话。", min_length=10)
    assert len(scenes) == 1
    assert scenes[0] == "只有一句话。"
    print("✅ 极短文本场景分割测试通过")


def test_from_folder():
    """测试从文件夹加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入两个小说文件
        (Path(tmpdir) / "novel1.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        (Path(tmpdir) / "novel2.txt").write_text(NOVEL_TEXT_2, encoding="utf-8")

        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        assert len(ds.novels) == 2
        assert len(ds.samples) > 0
        assert all(s.label_text for s in ds.samples)
        assert all(s.novel_path for s in ds.samples)

        # 验证两个小说的样本都被加载
        novel1_samples = ds.get_novel_samples(str(Path(tmpdir) / "novel1.txt"))
        novel2_samples = ds.get_novel_samples(str(Path(tmpdir) / "novel2.txt"))
        assert len(novel1_samples) > 0
        assert len(novel2_samples) > 0
        assert len(novel1_samples) + len(novel2_samples) == len(ds.samples)

        print(f"✅ from_folder 测试通过 (2 novels, {len(ds.samples)} samples)")


def test_from_folder_empty():
    """测试空文件夹"""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            NovelDataset.from_folder(tmpdir)
            assert False, "应抛出 ValueError"
        except ValueError:
            pass
    print("✅ 空文件夹测试通过")


def test_from_folder_not_exist():
    """测试不存在的文件夹"""
    try:
        NovelDataset.from_folder("/nonexistent/path")
        assert False, "应抛出 FileNotFoundError"
    except FileNotFoundError:
        pass
    print("✅ 不存在文件夹测试通过")


def test_from_single():
    """测试从单个文件加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "novel.txt"
        fpath.write_text(NOVEL_TEXT_1, encoding="utf-8")

        ds = NovelDataset.from_single(str(fpath), min_scene_length=50)
        assert len(ds.novels) == 1
        assert len(ds.samples) > 0
        print(f"✅ from_single 测试通过 ({len(ds.samples)} samples)")


def test_split():
    """测试 train/val 划分"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "novel.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        total = len(ds.samples)
        train, val = ds.split(train_ratio=0.7)

        assert len(train) + len(val) == total
        assert len(train) == int(total * 0.7)
        assert len(val) == total - len(train)
        print(f"✅ split 测试通过 (train={len(train)}, val={len(val)})")


def test_split_shuffle():
    """测试打乱划分"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "novel.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        train1, val1 = ds.split(train_ratio=0.5, shuffle=True, seed=42)
        train2, val2 = ds.split(train_ratio=0.5, shuffle=True, seed=42)
        train3, val3 = ds.split(train_ratio=0.5, shuffle=True, seed=99)

        # 相同 seed 应产生相同划分
        assert [s.sample_id for s in train1] == [s.sample_id for s in train2]
        # 不同 seed 可能不同（但不强制要求不同）
        print("✅ split shuffle 测试通过")


def test_save_load():
    """测试保存和加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "novel.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        # 保存
        out_path = str(Path(tmpdir) / "dataset.json")
        ds.save(out_path)
        assert Path(out_path).exists()

        # 加载
        ds2 = NovelDataset.load(out_path)
        assert len(ds2.samples) == len(ds.samples)
        for s1, s2 in zip(ds.samples, ds2.samples):
            assert s1.sample_id == s2.sample_id
            assert s1.input_text == s2.input_text
            assert s1.label_text == s2.label_text
            assert s1.scene_index == s2.scene_index

        print(f"✅ save/load 测试通过 ({len(ds2.samples)} samples)")


def test_context_field():
    """测试 context 字段：前一场景的摘要"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "novel.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        # 第一个场景的 context 应为空
        assert ds.samples[0].context == ""
        # 后续场景的 context 应非空
        if len(ds.samples) > 1:
            assert ds.samples[1].context != ""
        print("✅ context 字段测试通过")


def test_len_and_repr():
    """测试 __len__ 和 __repr__"""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "novel.txt").write_text(NOVEL_TEXT_1, encoding="utf-8")
        ds = NovelDataset.from_folder(tmpdir, min_scene_length=50)

        assert len(ds) == len(ds.samples)
        r = repr(ds)
        assert "NovelDataset" in r
        assert str(len(ds.novels)) in r
        assert str(len(ds.samples)) in r
        print("✅ __len__ / __repr__ 测试通过")


if __name__ == "__main__":
    test_training_sample()
    test_training_sample_id_unique()
    test_segment_scenes_basic()
    test_segment_scenes_min_length()
    test_segment_scenes_max_length()
    test_segment_scenes_empty()
    test_segment_scenes_short()
    test_from_folder()
    test_from_folder_empty()
    test_from_folder_not_exist()
    test_from_single()
    test_split()
    test_split_shuffle()
    test_save_load()
    test_context_field()
    test_len_and_repr()
    print("\n🎉 全部 16 个测试通过")
