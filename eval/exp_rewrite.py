# -*- coding: utf-8 -*-
"""v2.2 / T11 检索侧 A/B：多轮「拼接 query」 vs 「改写 query」

为什么单独做这个实验：
  agent 级通过率只能说明"最终答对了"，但多轮改写真正的价值是**让检索命中对的块**。
  这里固定检索链路（向量召回 10 → 重排 → top3），只换 query，直接看块级指标。

用法：
    python eval/exp_rewrite.py
输出：
    eval/exp_rewrite.json + 终端对比表
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from utils.path_tool import get_abs_path  # noqa: E402
from rag.vector_store import VectorStoreService  # noqa: E402
from rag.reranker import rerank_search  # noqa: E402
from rag.query_rewriter import rewrite_query  # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"
OUT = ROOT / "eval" / "exp_rewrite.json"


def joined_query(case: dict) -> str:
    msgs = case.get("messages") or []
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    return " ".join(users) if users else case["question"]


def last_user(case: dict) -> str:
    msgs = case.get("messages") or []
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    return users[-1] if users else case["question"]


def prev_turns(case: dict) -> list[str]:
    msgs = case.get("messages") or []
    roles = [m for m in msgs if m.get("role") in {"user", "assistant"}]
    return [m["content"] for m in roles[:-1]]


def evaluate(docs: list, case: dict) -> dict:
    gold = case.get("gold_chunk") or ""
    ranks = [i for i, d in enumerate(docs, 1) if gold and gold in d.page_content]
    rank = min(ranks) if ranks else None
    srcs = [os.path.basename(str((d.metadata or {}).get("source", ""))) for d in docs]
    return {
        "chunk_hit3": rank is not None,
        "chunk_rank": rank,
        "chunk_mrr5": round(1 / rank, 4) if rank else 0.0,
        "src_at3": bool(case.get("gold_source") and case["gold_source"] in srcs),
        "hit3": bool(gold and any(gold in d.page_content for d in docs)),
    }


def main():
    cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [c for c in cases if c.get("messages") and c.get("gold_source", "").endswith(".txt")]
    vs = VectorStoreService()
    k = 3
    cand = 10

    rows = []
    for c in cases:
        base_q = joined_query(c)
        t0 = time.perf_counter()
        new_q, used, changed = rewrite_query(last_user(c), prev_turns(c))
        rewrite_ms = round((time.perf_counter() - t0) * 1000)
        base_docs = rerank_search(vs.vector_store, base_q, topk=k, candidates=cand)
        new_docs = rerank_search(vs.vector_store, new_q, topk=k, candidates=cand)
        rows.append({
            "id": c["id"],
            "base_query": base_q[:60],
            "new_query": new_q[:60],
            "rewrite_used": used,
            "rewrite_changed": changed,
            "rewrite_ms": rewrite_ms,
            "base": evaluate(base_docs, c),
            "new": evaluate(new_docs, c),
        })
        print(f"{c['id']:<5} changed={changed!s:<5} ms={rewrite_ms:>4} | "
              f"旧 rank={rows[-1]['base']['chunk_rank']} 新 rank={rows[-1]['new']['chunk_rank']} | "
              f"{base_q[:24]} -> {new_q[:24]}")

    def agg(key: str) -> tuple:
        b = statistics.mean(1 if r["base"][key] else 0 for r in rows)
        n = statistics.mean(1 if r["new"][key] else 0 for r in rows)
        return round(b * 100, 1), round(n * 100, 1)

    hit_b, hit_n = agg("chunk_hit3")
    src_b, src_n = agg("src_at3")
    mrr_b = statistics.mean(r["base"]["chunk_mrr5"] for r in rows)
    mrr_n = statistics.mean(r["new"]["chunk_mrr5"] for r in rows)
    changed = [r for r in rows if r["rewrite_changed"]]
    ms = statistics.mean(r["rewrite_ms"] for r in rows)

    payload = {
        "n_cases": len(rows),
        "n_rewrite_changed": len(changed),
        "avg_rewrite_ms": round(ms),
        "chunk_recall3": {"base": hit_b, "new": hit_n},
        "chunk_mrr5": {"base": round(mrr_b, 4), "new": round(mrr_n, 4)},
        "src_at3": {"base": src_b, "new": src_n},
        "items": rows,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"多轮用例 {len(rows)} 条 | 实际改写 {len(changed)} 条 | 平均改写耗时 {ms:.0f} ms")
    print(f"块召回@3:  {hit_b}% -> {hit_n}%")
    print(f"块 MRR@5:  {mrr_b:.3f} -> {mrr_n:.3f}")
    print(f"来源@3 :   {src_b}% -> {src_n}%")
    print("=" * 62)
    print("已写入:", OUT)


if __name__ == "__main__":
    main()
