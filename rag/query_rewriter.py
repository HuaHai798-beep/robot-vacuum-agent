# -*- coding: utf-8 -*-
"""多轮查询改写（v2.2 / T11）

问题：用户第二句常常是「那滚刷呢」「它多久换」，直接拿去检索会丢主语。
做法（两级，先便宜后花钱）：
1. **规则门控**：只有出现指代词/省略（"那…呢""它""这个""还有"）或句子过短时才改写；
   大多数单轮问题直接跳过，零额外成本。
2. **LLM 改写**：把最近几轮对话 + 当前句交给小模型，输出一条独立可检索查询；
3. **校验回退**：改写结果必须保留原句的关键字（去掉"那/呢/它"等虚词后至少命中 1 个），
   否则回退原 query —— 宁可不变，也不能改跑偏。
"""
import os
import re

from utils.config_handler import rag_conf
from utils.logger_handler import logger

# 指代词 / 省略式追问的特征
_PRONOUN_HINTS = ("那", "它", "这个", "那个", "上面", "刚才", "还有", "换成", "另一", "这种", "呢")
_STOP = set("那呢它这个个上刚还有换另一种吗的吗？?，。！! ")

REWRITE_PROMPT = """你是检索查询改写器。根据对话历史，把用户最后一句话改写为**独立可检索**的查询。

规则：
1) 补全被省略的主语/对象（例如历史在说"滤网"，用户问"那滚刷呢"→"滚刷多久更换一次"）；
2) 保留原句的关键名词，不要添加历史里没有的新信息；
3) 不超过 30 个汉字，只输出改写结果本身，不要解释、不要引号。

对话历史：
{history}

最后一句话：{q}

改写："""


def rewrite_enabled() -> bool:
    if os.getenv("AGENT_NO_REWRITE") in {"1", "true", "True"}:
        return False
    return bool(rag_conf.get("query_rewrite", True))


def needs_rewrite(query: str, turns: list[str]) -> bool:
    q = (query or "").strip()
    if not turns:
        return False
    if len(q) <= 12:
        return True
    return any(h in q for h in _PRONOUN_HINTS)


def _keywords(text: str) -> set[str]:
    return {ch for ch in (text or "") if ch not in _STOP and ch.strip()}


def _valid(original: str, rewritten: str) -> bool:
    if not rewritten or len(rewritten) > 40:
        return False
    kw = _keywords(original)
    if not kw:
        return False
    # 改写结果必须保留原句至少一个实义词
    return any(k in rewritten for k in kw)


def rewrite_query(query: str, turns: list[str], provider: str | None = None,
                  model: str | None = None) -> tuple[str, bool, bool]:
    """返回 (最终查询, 是否调用了改写, 是否实际改变)。

    turns：最近若干轮 user/assistant 文本（越近越靠后）
    """
    if not rewrite_enabled() or not needs_rewrite(query, turns):
        return query, False, False

    history = "\n".join(f"- {t[:120]}" for t in turns[-4:])
    try:
        from model.factory import build_chat_model
        llm = build_chat_model(max_tokens=80, provider=provider, model=model)
        raw = llm.invoke(REWRITE_PROMPT.format(history=history, q=query)).content or ""
        rewritten = re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’]+$", "", raw.strip().splitlines()[0] if raw.strip() else "")
        rewritten = rewritten[:40]
        if _valid(query, rewritten) and rewritten != query.strip():
            logger.info(f"[rewrite] {query!r} -> {rewritten!r}")
            return rewritten, True, True
        logger.debug(f"[rewrite]结果未通过校验，回退原句：{rewritten!r}")
        return query, True, False
    except Exception as e:
        logger.warning(f"[rewrite]改写失败，回退原 query：{type(e).__name__}: {e}")
        return query, True, False
