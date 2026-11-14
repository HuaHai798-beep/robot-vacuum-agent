# -*- coding: utf-8 -*-
"""记忆隔离测试（前置骨架）：验证「同一 thread 记得住、不同 thread 不串味」。

现状：agent 尚未接入 checkpointer → 自动 SKIP。等记忆管理落地后，本脚本会自动开始跑。
用法：python eval/tests/test_memory_isolation.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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


def main() -> None:
    if not has_checkpointer():
        print("[SKIP] 尚未接入 checkpointer（记忆管理未实现），本测试不执行。")
        print("       落地 checkpointer 后，这里会验证：")
        print("       1) 同一 thread 第二轮能答出 120 平；")
        print("       2) 换一个新 thread 后，不应继承上一个用户的画像。")
        return

    from agent.react_agent import ReactAgent
    agent = ReactAgent().agent

    def ask(q: str, thread: str):
        final = None
        for chunk in agent.stream({"messages": [{"role": "user", "content": q}]},
                                  stream_mode="values", context={"report": False},
                                  config={"configurable": {"thread_id": thread}}):
            final = chunk
        return (final["messages"][-1].content or "").strip()

    a1 = ask(PROBES[0][0], "mem_iso_a")
    a2 = ask(PROBES[1][0], "mem_iso_a")
    b1 = ask("我家多大面积来着？", "mem_iso_b")

    print("同 thread 第二轮回答：", a2[:120])
    print("新 thread 回答：      ", b1[:120])

    ok_same = "120" in a2
    ok_iso = "120" not in b1
    print(("✓" if ok_same else "✗"), "同 thread 记忆：", ok_same)
    print(("✓" if ok_iso else "✗"), "跨 thread 隔离：", ok_iso)
    raise SystemExit(0 if (ok_same and ok_iso) else 1)


if __name__ == "__main__":
    main()
