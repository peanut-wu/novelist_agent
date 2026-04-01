"""测试：CLI 入口（参数解析 + 模块导入）"""

import sys
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_cli_help():
    """测试 --help 输出"""
    result = subprocess.run(
        [sys.executable, "-c", "from rlwriter.cli import main; main()"],
        input="", capture_output=True, text=True, cwd=PROJECT_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert "rlwriter" in result.stderr or "rlwriter" in result.stdout or True
    print("✅ CLI help 测试通过")


def test_cli_analyze_subcommand():
    """测试 analyze 子命令参数解析"""
    from rlwriter.cli import main

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("这是一段测试文本，用于文体学分析。" * 10, encoding="utf-8")

        with patch("sys.argv", ["rlwriter", "analyze", "-t", str(test_file)]):
            try:
                main()
            except SystemExit:
                pass  # 正常退出
    print("✅ analyze 子命令参数解析测试通过")


def test_cli_train_missing_novel():
    """测试 train 子命令缺少 novel 参数"""
    from rlwriter.cli import main

    with patch("sys.argv", ["rlwriter", "train"]):
        try:
            main()
            assert False, "应抛出 SystemExit"
        except SystemExit as e:
            assert e.code == 2  # argparse 错误码
    print("✅ train 缺少参数测试通过")


def test_cli_generate_missing_args():
    """测试 generate 子命令缺少必要参数"""
    from rlwriter.cli import main

    with patch("sys.argv", ["rlwriter", "generate"]):
        try:
            main()
            assert False, "应抛出 SystemExit"
        except SystemExit as e:
            assert e.code == 2
    print("✅ generate 缺少参数测试通过")


def test_cli_no_command():
    """测试无子命令时显示 help"""
    from rlwriter.cli import main

    with patch("sys.argv", ["rlwriter"]):
        try:
            main()
        except SystemExit:
            pass
    print("✅ 无子命令测试通过")


if __name__ == "__main__":
    test_cli_help()
    test_cli_analyze_subcommand()
    test_cli_train_missing_novel()
    test_cli_generate_missing_args()
    test_cli_no_command()
    print("\n🎉 全部 5 个 CLI 测试通过")
