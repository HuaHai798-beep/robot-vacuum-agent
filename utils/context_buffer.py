# -*- coding: utf-8 -*-
"""对话轮次缓冲（v2.2 引入，v2.4 修并发串台）。

用途：中间件在每次调模型前把最近的 user/assistant 文本放进来；
      RAG 工具在检索前用它做「多轮查询改写」。

v2.4 修复（P0）：原实现是模块级全局 list，进程内所有会话共享——
  两个用户并发时，A 的「我家 120 平」会进 B 的改写上下文（跨用户串数据）。
  改为 ContextVar：每个 asyncio 任务 / 线程上下文各自独立，
  顺序执行的单会话行为与旧版完全一致（评测口径不变）。
"""
import contextvars

_MAX_TURNS = 8
_TURNS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "recent_turns", default=()
)


def set_recent_turns(turns: list[str]) -> None:
    cleaned = tuple(t for t in (turns or []) if t and t.strip())[-_MAX_TURNS:]
    _TURNS.set(cleaned)


def get_recent_turns() -> list[str]:
    return list(_TURNS.get())


def clear() -> None:
    _TURNS.set(())
