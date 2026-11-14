# -*- coding: utf-8 -*-
"""数据集自检（不联网、不花钱）：新增用例后先跑它，避免把脏数据带进基线。

用法：python eval/tests/test_dataset.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "eval" / "dataset.jsonl"
DATA_DIR = ROOT / "data"

ALLOWED_CATEGORIES = {"retrieval", "tool", "multi_turn", "refuse", "safety", "ambiguous"}
ALLOWED_TOOLS = {"rag_summarize", "get_weather", "get_user_location", "get_user_id",
                 "get_current_month", "fetch_external_data", "fill_context_for_report"}
ALLOWED_INJECT = {"get_weather", "fetch_external_data", "rag_summarize", "get_user_location"}


def check() -> list[str]:
    errs: list[str] = []
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]

    ids = [r.get("id") for r in rows]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    if dup:
        errs.append(f"id 重复：{dup}")

    for r in rows:
        rid = r.get("id", "?")
        if not r.get("question"):
            errs.append(f"{rid} 缺 question")
        if r.get("category") not in ALLOWED_CATEGORIES:
            errs.append(f"{rid} category 非法：{r.get('category')}")
        for t in (r.get("expect_tools") or []):
            if t not in ALLOWED_TOOLS:
                errs.append(f"{rid} expect_tools 含未知工具：{t}")
        if r.get("inject_failure") and r["inject_failure"] not in ALLOWED_INJECT:
            errs.append(f"{rid} inject_failure 非法：{r['inject_failure']}")
        for k in (r.get("must_contain") or []):
            if not isinstance(k, (str, list)):
                errs.append(f"{rid} must_contain 元素类型非法：{k!r}")
        must = r.get("must_contain") or []
        if isinstance(must, str):
            errs.append(f"{rid} must_contain 应为数组")
        if r.get("tags") is not None and not isinstance(r["tags"], list):
            errs.append(f"{rid} tags 应为数组")

        msgs = r.get("messages")
        if msgs:
            if not isinstance(msgs, list) or not msgs:
                errs.append(f"{rid} messages 应为非空数组")
            else:
                if any(m.get("role") not in {"user", "assistant", "system"} for m in msgs):
                    errs.append(f"{rid} messages 含非法 role")
                if msgs[-1].get("role") != "user":
                    errs.append(f"{rid} messages 最后一条必须是 user")
                if msgs[-1].get("content") != r.get("question"):
                    errs.append(f"{rid} question 应与 messages 最后一条 user 内容一致")

        src = r.get("gold_source") or ""
        if src and src not in {"records.csv", "wttr.in"}:
            if not (DATA_DIR / src).exists():
                errs.append(f"{rid} gold_source 文件不存在：{src}")

    print(f"用例数：{len(rows)} | 分类：{dict(Counter(r['category'] for r in rows))} | "
          f"带 tags：{sum(1 for r in rows if r.get('tags'))}")
    return errs


if __name__ == "__main__":
    errors = check()
    if errors:
        print("\n✗ 数据集自检失败：")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)
    print("✓ 数据集自检通过")
