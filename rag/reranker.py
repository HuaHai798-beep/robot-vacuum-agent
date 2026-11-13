# -*- coding: utf-8 -*-
"""本地混合重排：BM25（字符 bigram）+ 向量召回，用 RRF 融合。

为什么需要它：
    评测发现 chunk_recall@5 = 100% 但 chunk_recall@3 = 87.5%
    —— 正确的块**总是能召回到前 5**，只是排不进前 3。这是排序问题，不是召回问题。

为什么不用官方 gte-rerank：
    账号未开通（403 AccessDenied）。本模块零依赖、可离线跑；
    后续开通后只需把 `rerank_search` 换成官方接口，接口签名保持一致。

中文分词：没有 jieba，用「字符 bigram」当 token（对中文短文本效果好且零依赖）。
"""
import math
import re
from collections import Counter

from langchain_core.runnables import RunnableLambda

_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """中文用字符 bigram，英文/数字保留为词。"""
    text = (text or "").lower()
    tokens: list[str] = []
    # 英文/数字词
    for w in re.findall(r"[a-z0-9]+", text):
        tokens.append(w)
    # 中文字符 bigram
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for seg in cjk:
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


class BM25:
    """标准 BM25，用全量知识库统计 idf。"""

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
    """按集合内容缓存 BM25 索引（88 块，构建一次几十毫秒）。"""

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


def rerank_search(vector_store, query: str, topk: int = 3, candidates: int = 10):
    """向量召回 candidates 条 → BM25 对候选打分 → RRF 融合 → 取 topk。"""
    vec_docs = vector_store.similarity_search(query, k=candidates)
    if not vec_docs:
        return []

    idx = _CACHE.get(vector_store)
    qtokens = tokenize(query)
    bm25_scores = [idx.score(qtokens, i) if i < len(idx.tf) else 0.0 for i in range(len(vec_docs))]

    # RRF：向量顺序 + BM25 顺序融合
    rrf = {}
    for rank, _ in enumerate(vec_docs):
        rrf[rank] = rrf.get(rank, 0.0) + 1.0 / (_RRF_K + rank + 1)
    bm_order = sorted(range(len(vec_docs)), key=lambda i: bm25_scores[i], reverse=True)
    for rank, pos in enumerate(bm_order):
        rrf[pos] = rrf.get(pos, 0.0) + 1.0 / (_RRF_K + rank + 1)

    fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    return [vec_docs[i] for i, _ in fused[:topk]]


def make_rerank_retriever(vector_store, topk: int = 3, candidates: int = 10):
    """返回一个 LangChain Runnable，可直接用于 chain 与 retriever.invoke(query)。"""
    return RunnableLambda(lambda q: rerank_search(vector_store, q, topk=topk, candidates=candidates))
