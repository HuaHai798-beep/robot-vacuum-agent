# -*- coding: utf-8 -*-
"""评测跑批 v1（规则评分 + hit@3 + 分层归因）

用法（在项目根目录执行）：
    python eval/run_eval.py                 # 跑全量，和 baseline 对比
    python eval/run_eval.py --limit 5       # 只跑前 5 条（调试/省钱）
    python eval/run_eval.py --save-baseline # 把本次结果写成新基线
    python eval/run_eval.py --no-llm        # 只跑检索层（hit@3），不调 agent

输出：
    eval/results/run_<时间>.jsonl   每条用例的原始记录（可回放）
    eval/baseline.json              基线（含指标与会话信息）
"""
import argparse
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
BASELINE = EVAL_DIR / "baseline.json"

REFUSAL_MARKERS = ["无法", "不能", "不可以", "抱歉", "不提供", "不支持", "无权", "建议咨询", "帮不了",
                   "无关", "不相关", "不符合", "不属于", "超出", "无法提供", "不能提供"]
FOLLOWUP_MARKERS = ["？", "?", "请问", "方便告知", "需要了解", "告诉我"]


# ----------------------------- 工具函数 -----------------------------
def norm(s: str) -> str:
    """归一化：去空白、全角转半角、去标点差异，便于关键词匹配"""
    if not s:
        return ""
    table = str.maketrans("％～－　，。；：（）", "%~- ,.;:()")
    s = str(s).translate(table)
    return "".join(s.split()).lower()


def grade(case: dict, result: dict) -> dict:
    """规则评分：工具 / 事实 / 拒答 / 追问"""
    ans = norm(result["answer"])
    tools = result["tools"]
    checks = {}

    expect_tools = case.get("expect_tools") or []
    if expect_tools:
        if case.get("expect_tools_ordered"):
            # 只对"取数主链"校验先后顺序；fill_context_for_report 只要求出现（它是开关，不参与取数顺序）
            ordered = [t for t in expect_tools if t != "fill_context_for_report"]
            extras = [t for t in expect_tools if t == "fill_context_for_report"]
            it = iter(tools)
            checks["tool_ok"] = all(t in it for t in ordered) and all(t in tools for t in extras)
        else:
            checks["tool_ok"] = all(t in tools for t in expect_tools)
    else:
        checks["tool_ok"] = True

    must = case.get("must_contain") or []
    # 关键词支持"同义词或"写法：既可以是字符串，也可以是字符串列表（命中任一即可）
    checks["fact_ok"] = all(
        any(norm(alt) in ans for alt in (k if isinstance(k, list) else [k])) for k in must
    ) if must else True

    if case.get("expect_refusal"):
        checks["refusal_ok"] = any(norm(m) in ans for m in REFUSAL_MARKERS)
    else:
        checks["refusal_ok"] = True

    if case.get("expect_followup"):
        checks["followup_ok"] = any(norm(m) in ans for m in FOLLOWUP_MARKERS)
    else:
        checks["followup_ok"] = True

    result["checks"] = checks
    result["pass"] = all(checks.values())

    # 分层归因：错在哪一层
    if result["pass"]:
        result["layer"] = ""
    elif not checks["tool_ok"]:
        result["layer"] = "工具层"
    elif not checks["fact_ok"]:
        result["layer"] = "检索/回答层"
    elif not checks["refusal_ok"]:
        result["layer"] = "拒答层"
    else:
        result["layer"] = "追问层"
    return result


def run_agent(agent, case: dict, timeout_s: float = 120) -> dict:
    """跑一条用例，记录工具调用、答案、耗时"""
    messages = case.get("messages") or [{"role": "user", "content": case["question"]}]
    t0 = time.perf_counter()
    final = None
    for chunk in agent.stream(
        {"messages": messages},
        stream_mode="values",
        context={"report": False},
        config={"configurable": {"thread_id": case.get("thread") or f"t_{case['id']}"}},
    ):
        final = chunk
    latency = round(time.perf_counter() - t0, 2)

    msgs = final["messages"] if final else []
    tools = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tools.append(tc.get("name", ""))
    answer = (msgs[-1].content if msgs and isinstance(msgs[-1].content, str) else "") or ""
    return {"tools": tools, "answer": answer.strip(), "latency": latency, "n_messages": len(msgs)}


def hit_at_3(retriever, case: dict) -> bool:
    must = case.get("must_contain") or []
    if not must:
        return True
    docs = retriever.invoke(case["question"])
    joined = norm(" ".join(d.page_content for d in docs[:3]))
    return all(norm(k) in joined for k in must)


def p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return round(values[idx], 2)


def summarize(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "pass": 0})
        c["n"] += 1
        c["pass"] += 1 if r["pass"] else 0

    tool_cases = [r for r in results if r.get("expect_tools")]
    tool_ok = [r for r in tool_cases if r["checks"].get("tool_ok")]
    retrieval = [r for r in results if r.get("hit3") is not None]
    hit = [r for r in retrieval if r["hit3"]]

    return {
        "n": total,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "by_category": {k: {"n": v["n"], "pass_rate": round(v["pass"] / v["n"] * 100, 1)} for k, v in by_cat.items()},
        "tool_acc": round(len(tool_ok) / len(tool_cases) * 100, 1) if tool_cases else None,
        "hit_at_3": round(len(hit) / len(retrieval) * 100, 1) if retrieval else None,
        "avg_latency": round(statistics.mean([r["latency"] for r in results]), 2) if results else 0,
        "p95_latency": p95([r["latency"] for r in results]),
    }


def print_summary(cur: dict, base: dict | None):
    def delta(key, fmt="{:.1f}"):
        if not base or base.get(key) is None or cur.get(key) is None:
            return ""
        d = cur[key] - base[key]
        return f" ({'+' if d >= 0 else ''}{fmt.format(d)})"

    print("\n" + "=" * 62)
    print(f"{'指标':<14}{'基线':>12}{'本次':>12}")
    print("-" * 62)
    print(f"{'总体通过率':<12}{str(base.get('pass_rate')) if base else '-':>12}{cur['pass_rate']:>12.1f}%{delta('pass_rate')}")
    print(f"{'工具准确率':<12}{str(base.get('tool_acc')) if base else '-':>12}{str(cur['tool_acc']):>12}%{delta('tool_acc')}")
    print(f"{'hit@3':<13}{str(base.get('hit_at_3')) if base else '-':>12}{str(cur['hit_at_3']):>12}%{delta('hit_at_3')}")
    print(f"{'平均耗时(s)':<11}{str(base.get('avg_latency')) if base else '-':>12}{cur['avg_latency']:>12}{delta('avg_latency', '{:.2f}')}")
    print(f"{'P95耗时(s)':<11}{str(base.get('p95_latency')) if base else '-':>12}{cur['p95_latency']:>12}")
    print("-" * 62)
    for k, v in sorted(cur["by_category"].items()):
        print(f"  {k:<14} n={v['n']:<3} 通过率 {v['pass_rate']}%")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="只跑检索层 hit@3")
    args = ap.parse_args()

    cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    print(f"用例数: {len(cases)}")

    from rag.vector_store import VectorStoreService

    vs = VectorStoreService()
    retriever = vs.get_retriever()

    agent = None
    if not args.no_llm:
        from agent.react_agent import ReactAgent

        agent = ReactAgent().agent

    results = []
    for i, case in enumerate(cases, 1):
        rec = {"id": case["id"], "category": case["category"], "question": case["question"],
               "note": case.get("note", ""), "expect_tools": case.get("expect_tools") or []}
        try:
            if case["category"] in ("retrieval", "multi_turn"):
                rec["hit3"] = hit_at_3(retriever, case)
            if not args.no_llm:
                out = run_agent(agent, case)
                rec.update(out)
                rec = grade(case, rec)
            else:
                rec["checks"], rec["pass"], rec["layer"] = {}, rec.get("hit3", True), ""
            print(f"[{i}/{len(cases)}] {case['id']:<6} {'PASS' if rec.get('pass') else 'FAIL'}  {rec.get('layer', '')}")
        except Exception as e:
            rec.update({"pass": False, "layer": "系统层", "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(cases)}] {case['id']:<6} ERROR {e}")
            traceback.print_exc()
        results.append(rec)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = RESULTS_DIR / f"run_{ts}.jsonl"
    out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    cur = summarize(results)
    base = None
    if BASELINE.exists() and not args.save_baseline:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))

    print_summary(cur, base)

    failures = [r for r in results if not r["pass"]]
    if failures:
        print("\n失败用例（Top 10）：")
        for r in failures[:10]:
            print(f"  {r['id']:<6} [{r['layer']}] {r['question']}")
            if r.get("tools"):
                print(f"         工具: {r['tools']}")
            if r.get("answer"):
                print(f"         回答: {r['answer'][:60]}...")

    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "metrics": cur, "results_file": str(out_file)}
    if args.save_baseline or not BASELINE.exists():
        BASELINE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入基线: {BASELINE}")
    print(f"\n明细: {out_file}")


if __name__ == "__main__":
    main()
