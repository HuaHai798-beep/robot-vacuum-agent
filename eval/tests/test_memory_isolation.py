# -*- coding: utf-8 -*-
"""记忆隔离测试（前置骨架）：验证「同一 thread 记得住、不同 thread 不串味」。

现状：agent 尚未接入 checkpointer（`create_agent` 没有 `checkpointer=`）→ 本用例 SKIP。
       等 T13 落地记忆管理后，本用例会自动开始跑，并且**真的会失败**（有 assert）。

v2.1 后修 P1：从"只 print"改成 `def test_*` + assert，SKIP 走 _harness.skip。
用法：python eval/tests/test_memory_isolation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_all, skip

PROBES = [
    ("我家 120 平，木地板，养了猫", "what_is_my_home"),   # 写入
    ("我家多大面积来着？", "recall_same_thread"),          # 同 thread 应记得 120
]


def has_checkpointer() -> bool:
    try:
        from agent.react_agent import ReactAgent
        agent = ReactAgent().agent
        # create_agent(checkpointer=...) 后，图对象上会有 checkpointer 属性
        return getattr(agent, "checkpointer", None) is not None
    except Exception as e:
        print(f"[skip] 无法构建 agent：{type(e).__name__}: {e}")
        return False


def _ask(agent, q: str, thread: str) -> str:
    final = None
    for chunk in agent.stream({"messages": [{"role": "user", "content": q}]},
                              stream_mode="values", context={"report": False},
                              config={"configurable": {"thread_id": thread}}):
        final = chunk
    return (final["messages"][-1].content or "").strip()


def test_memory_isolation():
    """同一 thread 的第二轮要记得 120 平；换新 thread 不能继承上一用户画像"""
    if not has_checkpointer():
        skip("尚未接入 checkpointer（记忆管理未实现）：同一 thread 记不住、跨 thread 也无从验证")

    from agent.react_agent import ReactAgent
    agent = ReactAgent().agent

    _ask(agent, PROBES[0][0], "mem_iso_a")                       # 第一轮：写入画像
    a2 = _ask(agent, PROBES[1][0], "mem_iso_a")                  # 第二轮：同 thread 应记得
    b1 = _ask(agent, "我家多大面积来着？", "mem_iso_b")            # 新 thread：不应继承

    print("同 thread 第二轮回答：", a2[:120])
    print("新 thread 回答：      ", b1[:120])

    assert "120" in a2, f"同一 thread 第二轮没记住「120 平」：{a2[:100]}"
    assert "120" not in b1, f"新 thread 串味了（继承了上一个用户的画像）：{b1[:100]}"


if __name__ == "__main__":
    sys.exit(run_all(globals()))
