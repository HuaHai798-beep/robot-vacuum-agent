# -*- coding: utf-8 -*-
"""对话轮次缓冲（v2.2）

用途：中间件在每次调模型前把最近的 user/assistant 文本放进来；
      RAG 工具在检索前用它做「查询改写」。
为什么不放在 rag_service 里：middleware 与 rag 都要用，放独立模块避免循环依赖。
"""

_MAX_TURNS = 8
_TURNS: list[str] = []


def set_recent_turns(turns: list[str]) -> None:
    global _TURNS
    _TURNS = [t for t in (turns or []) if t and t.strip()][-_MAX_TURNS:]


def get_recent_turns() -> list[str]:
    return list(_TURNS)


def clear() -> None:
    global _TURNS
    _TURNS = []
