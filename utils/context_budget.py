# -*- coding: utf-8 -*-
"""v2.3 上下文预算与「静态/动态 token」拆分。

为什么需要：
  Agent 每一轮都要把 system + 工具定义 + 全部历史重新发给模型（A2）。
  usage_metadata 只给 total input tokens，看不出「多少是静态提示、多少是消息」。
  这里用「实际 input − 消息估算」反推静态开销，作为后续优化的归因基线。
"""
import json

from langchain_core.messages import AIMessage

# 中文约 1.45 字/token，英文/数字约 3.6 字符/token（DeepSeek/Qwen 量级，够用即可）
CJK_PER_TOKEN = 1.45
LATIN_PER_TOKEN = 3.6


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    text = str(text)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, round(cjk / CJK_PER_TOKEN + other / LATIN_PER_TOKEN))


def message_text(m) -> str:
    content = getattr(m, "content", "")
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", "")))
            else:
                parts.append(str(x))
        content = "".join(parts)
    extra = ""
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls:
        extra = json.dumps(tool_calls, ensure_ascii=False)
    return str(content or "") + extra


def estimate_messages_tokens(messages) -> int:
    return sum(estimate_text_tokens(message_text(m)) for m in (messages or []))


def split_static_dynamic(messages) -> dict:
    """按每条 AIMessage 的 usage_metadata 反推静态开销（system+工具定义）。

    第 i 条 AIMessage 的 input_tokens 覆盖「它之前的所有消息 + 静态提示」，
    减去 messages[:i] 的估算，就是这一轮的静态开销。
    """
    static = dynamic = calls = 0
    for i, m in enumerate(messages or []):
        if not isinstance(m, AIMessage):
            continue
        usage = getattr(m, "usage_metadata", None) or {}
        try:
            tokens_in = int(usage.get("input_tokens") or 0)
        except Exception:
            tokens_in = 0
        if not tokens_in:
            continue
        dyn = estimate_messages_tokens(messages[:i])
        static += max(0, tokens_in - dyn)
        dynamic += dyn
        calls += 1
    return {
        "calls": calls,
        "static": static,
        "dynamic": dynamic,
        "static_avg": round(static / calls) if calls else 0,
        "static_share": round(static / (static + dynamic) * 100, 1) if (static + dynamic) else 0.0,
    }
