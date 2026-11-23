# -*- coding: utf-8 -*-
"""v2.4 P0 回归：对话轮次缓冲必须按上下文隔离，不能跨会话串台。

背景：v2.2 的 utils/context_buffer.py 是模块级全局 list，两个并发会话会互相覆盖，
      导致 A 的「我家 120 平」进入 B 的检索改写（跨用户串数据）。
      本用例在 v2.4 改成 ContextVar 后必须通过。

用法：python eval/tests/test_context_buffer_concurrency.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_all  # noqa: E402

from utils.context_buffer import clear, get_recent_turns, set_recent_turns  # noqa: E402


async def _worker(name: str) -> list[str]:
    """模拟一个会话：写自己的轮次 → 让出控制权 → 再追加一轮。"""
    set_recent_turns([f"{name}-1"])
    await asyncio.sleep(0.01)
    set_recent_turns(get_recent_turns() + [f"{name}-2"])
    await asyncio.sleep(0.01)
    return get_recent_turns()


def test_buffer_isolated_between_async_sessions():
    async def main():
        return await asyncio.gather(_worker("A"), _worker("B"))

    a, b = asyncio.run(main())
    print("会话 A 的轮次：", a)
    print("会话 B 的轮次：", b)
    assert a == ["A-1", "A-2"], f"会话 A 被污染：{a}"
    assert b == ["B-1", "B-2"], f"会话 B 被污染：{b}"


def test_buffer_sequential_still_works():
    """顺序场景行为不变：先写后读，取最后 8 条。"""
    clear()
    set_recent_turns([f"t{i}" for i in range(12)])
    got = get_recent_turns()
    assert got == [f"t{i}" for i in range(4, 12)], got


if __name__ == "__main__":
    sys.exit(run_all(globals()))
