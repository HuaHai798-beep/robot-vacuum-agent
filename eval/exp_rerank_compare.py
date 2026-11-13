# -*- coding: utf-8 -*-
"""块级重排对照实验（v1.9）：修复 BM25 候选-索引错位后重跑。

对比同一批候选（k=10）下的四种排序：
    1) vector     ：纯向量 top3
    2) api        ：qwen3.7-text-rerank 精排 top3
    3) bm25@0.3/0.5/1.0：本地 BM25(jieba)+RRF 融合 top3（索引建在候选集内）

指标：chunk_recall@3 / chunk_MRR@5 / src@3 / hit@3（与主评测口径一致）
输出：eval/exp_rerank_compare.json（可复现，写进报告引用）

用法：python eval/exp_rerank_compare.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_eval import EVAL_DIR, norm, retrieval_query  # noqa: E402
from rag.reranker import api_rerank_search, bm25_rerank_search  # noqa: E402
from rag.vector_store import VectorStoreService  # noqa: E402

ALPHAS = [0.3, 0.5, 1.0]


def rank_metrics(docs: list, case: dict) -> dict:
    """在给定的 docs（已排好序）上算块级/文件级指标"""
    must = case.get("must_contain") or []
    joined = norm(" ".join(d.page_content for d in docs[:3]))
    hit3 = None
    if must:
        hit3 = all(any(norm(alt) in joined for alt in (k if isinstance(k, list) else [k])) for k in must)

    gold_chunk = case.get("gold_chunk")
    chunk_rank = 0
    if gold_chunk:
        for i, d in enumerate(docs, 1):
            if norm(gold_chunk) in norm(d.page_content):
                chunk_rank = i
                break

    gold_src = case.get("gold_source") or ""
    src_rank = 0
    for i, d in enumerate(docs, 1):
        if norm(gold_src) in norm(str((d.metadata or {}).get("source", ""))):
            src_rank = i
            break

    return {
        "hit3": hit3,
        "chunk_hit3": 1 <= chunk_rank <= 3,
        "chunk_mrr5": round(1.0 / chunk_rank, 4) if chunk_rank else 0.0,
        "src3": 1 <= src_rank <= 3,
        "chunk_rank": chunk_rank,
    }


def agg(rows: list) -> dict:
    n = len(rows)
    return {
        "n": n,
        "chunk_recall@3": round(sum(1 for r in rows if r["chunk_hit3"]) / n * 100, 1),
        "chunk_MRR@5": round(sum(r["chunk_mrr5"] for r in rows) / n, 3),
        "src@3": round(sum(1 for r in rows if r["src3"]) / n * 100, 1),
        "hit@3": round(sum(1 for r in rows if r["hit3"]) / n * 100, 1) if any(r["hit3"] is not None for r in rows) else None,
    }


def main():
    cases = [json.loads(l) for l in (EVAL_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [c for c in cases if (c.get("gold_source") or "").endswith(".txt") and c.get("gold_chunk")]
    print(f"检索类用例: {len(cases)} 条")

    vs = VectorStoreService().vector_store
    variants = ["vector", "api"] + [f"bm25@{a}" for a in ALPHAS]
    rows = {v: [] for v in variants}
    t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        q = retrieval_query(case)
        base = vs.similarity_search(q, k=10)

        rows["vector"].append(rank_metrics(base[:3], case))
        rows["api"].append(rank_metrics(api_rerank_search(vs, q, topk=3, candidates=10), case))
        for a in ALPHAS:
            rows[f"bm25@{a}"].append(
                rank_metrics(bm25_rerank_search(vs, q, topk=3, candidates=10, alpha=a), case)
            )
        print(f"  [{i}/{len(cases)}] {case['id']} 完成")

    result = {"meta": {
        "cases": len(cases), "candidates": 10, "topk": 3,
        "alphas": ALPHAS, "elapsed_s": round(time.perf_counter() - t0, 1),
        "note": "v1.9 修复了 BM25 候选-索引错位（v1.7 的结论因此失效，需以本结果为准）",
    }, "variants": {v: agg(rows[v]) for v in variants}}

    out = EVAL_DIR / "exp_rerank_compare.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"{'方案':<12}{'chunk_recall@3':>16}{'chunk_MRR@5':>14}{'src@3':>10}{'hit@3':>10}")
    print("-" * 76)
    for v in variants:
        m = result["variants"][v]
        print(f"{v:<12}{m['chunk_recall@3']:>14}%{m['chunk_MRR@5']:>14}{m['src@3']:>9}%"
              f"{('-' if m['hit@3'] is None else str(m['hit@3']) + '%'):>10}")
    print("=" * 76)
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
