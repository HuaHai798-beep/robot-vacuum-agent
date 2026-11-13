# -*- coding: utf-8 -*-
"""检索重排：支持「官方语义重排」与「本地 BM25」两种实现。

背景（v1.7 评测结论）：
    块级指标显示 chunk_recall@5 = 100%、chunk_recall@3 = 87.5%
    —— 正确块总在前 5，只是排不进前 3，属于排序问题。

两种实现：
    1) api  ：DashScope qwen3.7-text-rerank（cross-encoder，语义重排）—— 已验证有效
    2) bm25 ：本地 BM25(jieba) + RRF —— **已证伪**（v1.9 修复了候选-索引错位后重跑，
               结论仍为负收益，见 docs/评测报告.md §15）

切换：config/chroma.yml
    rerank_enabled: true
    rerank_method: api        # api | bm25
    rerank_model: qwen3.7-text-rerank
    rerank_candidates: 10

埋点（v1.9）：所有入口都接受 trace: dict | None，会写入
    rerank_method / rerank_latency(秒) / rerank_calls / rerank_fallback(bool) / rerank_error(str)
    评测脚本据此把「重排耗时/降级」和「模型耗时」分开归因。
"""
import math
import re
import time
from collections import Counter

from langchain_core.runnables import RunnableLambda

from utils.config_handler import chroma_conf
from utils.logger_handler import logger

_RRF_K = 60


def _mark(trace: dict | None, **kw) -> None:
    """把埋点写进 trace；trace 为 None 时静默忽略（生产路径零开销）。"""
    if trace is not None:
        trace.update(kw)


# ============================ 一、官方语义重排 ============================
def api_rerank_search(vector_store, query: str, topk: int = 3, candidates: int = 10,
                      trace: dict | None = None):
    """向量召回 candidates → qwen3.7-text-rerank 重排 → 取 topk"""
    import dashscope

    t0 = time.perf_counter()
    try:
        docs = vector_store.similarity_search(query, k=candidates)
    except Exception as e:  # 向量检索本身失败：交给上层处理
        _mark(trace, rerank_method="api", rerank_calls=0, rerank_fallback=True,
              rerank_error=f"vector_search: {e}", rerank_latency=0.0)
        raise

    if not docs:
        _mark(trace, rerank_method="api", rerank_calls=0, rerank_fallback=False,
              rerank_latency=round(time.perf_counter() - t0, 4))
        return []
    if len(docs) <= topk:
        # 候选不够，重排没有意义，省一次 API 调用
        _mark(trace, rerank_method="api", rerank_calls=0, rerank_fallback=False,
              rerank_latency=round(time.perf_counter() - t0, 4))
        return docs

    model = chroma_conf.get("rerank_model", "qwen3.7-text-rerank")
    try:
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=[d.page_content for d in docs],
            top_n=min(topk, len(docs)),
        )
    except Exception as e:
        logger.warning(f"[rerank]调用失败，降级为向量原序：{type(e).__name__}: {e}")
        _mark(trace, rerank_method="api", rerank_calls=1, rerank_fallback=True,
              rerank_error=f"{type(e).__name__}: {e}",
              rerank_latency=round(time.perf_counter() - t0, 4))
        return docs[:topk]

    if getattr(resp, "status_code", 500) != 200 or not getattr(resp, "output", None):
        logger.warning(f"[rerank]状态码异常({getattr(resp, 'status_code', '?')})，降级为向量原序")
        _mark(trace, rerank_method="api", rerank_calls=1, rerank_fallback=True,
              rerank_error=f"status={getattr(resp, 'status_code', '?')}",
              rerank_latency=round(time.perf_counter() - t0, 4))
        return docs[:topk]

    results = sorted(resp.output.results, key=lambda x: x.relevance_score, reverse=True)
    out = []
    for item in results[:topk]:
        idx = int(item.index)          # index 是「候选里的位置」，不是文档 id
        if 0 <= idx < len(docs):
            out.append(docs[idx])

    _mark(trace, rerank_method="api", rerank_calls=1, rerank_fallback=False,
          rerank_latency=round(time.perf_counter() - t0, 4),
          rerank_scores=[round(float(getattr(x, "relevance_score", 0)), 4) for x in results[:topk]])
    return out or docs[:topk]


# ============================ 二、本地 BM25（对照实验，已证伪） ============================
def tokenize(text: str) -> list[str]:
    """优先 jieba 分词，缺失时退回字符 bigram。"""
    text = (text or "").lower()
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        tokens = re.findall(r"[a-z0-9]+", text)
        for seg in re.findall(r"[\u4e00-\u9fff]+", text):
            tokens.extend([seg] if len(seg) == 1 else [seg[i:i + 2] for i in range(len(seg) - 1)])
        return tokens


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        df = Counter()
        for toks in docs_tokens:
            df.update(set(toks))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(t) for t in docs_tokens]
        self.dl = [len(t) for t in docs_tokens]

    def score(self, query_tokens: list[str], i: int) -> float:
        s = 0.0
        tf, dl = self.tf[i], self.dl[i]
        for t in query_tokens:
            f = tf.get(t)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
            )
        return s


class _IndexCache:
    """全库索引缓存（v1.7 遗留，仅供离线对照实验使用）。

    ⚠️ v1.9 起生产路径不再用它：全库索引的文档序号与「召回候选的顺序」不对应，
    直接 idx.score(i) 会把 A 文档的词频分算到 B 文档头上（v1.7 实验即因此失效）。
    """

    def __init__(self):
        self.key = None
        self.index = None

    def get(self, vector_store) -> BM25:
        data = vector_store.get()
        docs = data.get("documents") or []
        key = hash(tuple(docs))
        if self.key != key:
            self.index = BM25([tokenize(d) for d in docs])
            self.key = key
        return self.index


_CACHE = _IndexCache()


def bm25_rerank_search(vector_store, query: str, topk: int = 3, candidates: int = 10,
                       alpha: float = 0.5, trace: dict | None = None):
    """RRF 融合（alpha 为 BM25 权重）。

    修复（v1.9）：BM25 索引建在**候选集内部**，保证 score(i) 与候选 i 是同一篇文档。
    评测结果：alpha>0 仍低于纯向量（见 docs/评测报告.md §15），故生产默认 api。
    """
    t0 = time.perf_counter()
    vec_docs = vector_store.similarity_search(query, k=candidates)
    if not vec_docs:
        _mark(trace, rerank_method="bm25", rerank_calls=1, rerank_fallback=False,
              rerank_latency=round(time.perf_counter() - t0, 4))
        return []

    # ← 关键修复：在候选集上建索引，序号天然对齐
    local = BM25([tokenize(d.page_content) for d in vec_docs])
    qtokens = tokenize(query)
    bm25_scores = [local.score(qtokens, i) for i in range(len(vec_docs))]

    rrf = {}
    for rank in range(len(vec_docs)):
        rrf[rank] = 1.0 / (_RRF_K + rank + 1)
    bm_order = sorted(range(len(vec_docs)), key=lambda i: bm25_scores[i], reverse=True)
    for rank, pos in enumerate(bm_order):
        rrf[pos] = rrf.get(pos, 0.0) + alpha * 1.0 / (_RRF_K + rank + 1)

    fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    _mark(trace, rerank_method="bm25", rerank_calls=1, rerank_fallback=False,
          rerank_latency=round(time.perf_counter() - t0, 4),
          rerank_alpha=alpha)
    return [vec_docs[i] for i, _ in fused[:topk]]


# ============================ 三、统一入口 ============================
def rerank_search(vector_store, query: str, topk: int = 3, candidates: int = 10,
                  trace: dict | None = None):
    method = (chroma_conf.get("rerank_method") or "api").lower()
    alpha = float(chroma_conf.get("rerank_bm25_alpha", 0.5))
    if method == "bm25":
        return bm25_rerank_search(vector_store, query, topk=topk,
                                  candidates=candidates, alpha=alpha, trace=trace)
    return api_rerank_search(vector_store, query, topk=topk,
                             candidates=candidates, trace=trace)


def make_rerank_retriever(vector_store, topk: int = 3, candidates: int = 10,
                          trace: dict | None = None):
    """返回 LangChain Runnable，可直接 invoke / 参与 chain。"""
    return RunnableLambda(lambda q: rerank_search(vector_store, q, topk=topk,
                                                   candidates=candidates, trace=trace))
