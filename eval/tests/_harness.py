# -*- coding: utf-8 -*-
"""零依赖测试运行器：既支持 pytest 收集，也支持 python 直接运行。

背景（v2.1 后修 P1）：原来的 eval/tests/*.py 只是"打印脚本"——没有 assert、没有
`def test_*`，pytest 一条都收集不到，脚本也永远不会失败。现在统一为：
    - 每个用例写成 `def test_xxx(): assert ...`
    - `python eval/tests/test_xxx.py` 用本运行器执行（不需要装 pytest）
    - 装了 pytest 时 `pytest eval/tests` 也能正常收集
"""
import sys


class SkipTest(Exception):
    """本文件被跳过（例如依赖的能力尚未实现）"""


try:  # 装了 pytest 就用它的 skip，语义更标准
    import pytest as _pytest

    def skip(reason: str):
        _pytest.skip(reason)
except ImportError:  # 没装 pytest：自己抛 SkipTest
    def skip(reason: str):
        raise SkipTest(reason)


def test_names(namespace: dict) -> list[str]:
    return sorted(n for n in namespace if n.startswith("test_") and callable(namespace[n]))


def run_all(namespace: dict) -> int:
    """执行 namespace 里所有 test_* 函数，返回进程退出码（0=全过）"""
    names = test_names(namespace)
    passed = failed = skipped = 0
    for name in names:
        try:
            namespace[name]()
            passed += 1
            print(f"PASS  {name}")
        except SkipTest as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # 非断言异常同样算失败
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n合计：{len(names)} 个用例 | 通过 {passed} | 失败 {failed} | 跳过 {skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all({}))
