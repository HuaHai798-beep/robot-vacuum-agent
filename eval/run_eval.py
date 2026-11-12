# -*- coding: utf-8 -*-
"""评测跑批 v2（规则评分 + hit@3 + 分层归因 + 企业级加固）

v2 相比 v1 增加三道保险：
    1) 预检 preflight：开跑前探测一次模型额度，403/欠费立刻退出，不浪费整轮跑批；
    2) 重试退避：403/429/超时 自动重试 3 次（5s/15s/30s），记录重试次数；
    3) 基线元数据：baseline 里记录 model / dataset_sha256 / git_commit，模型不同则拒绝横比。

用法（在项目根目录执行）：
    python eval/run_eval.py
    python eval/run_eval.py --limit 5
    python eval/run_eval.py --baseline eval/baseline_qwen_plus.json --save-baseline
    python eval/run_eval.py --no-llm            # 只跑检索层 hit@3（不消耗对话额度）
    python eval/run_eval.py --skip-preflight    # 跳过预检（调试用）
"""
import argparse
import hashlib
import json
import statistics
import subprocess
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

REFUSAL_MARKERS = ["无法", "不能", "不可以", "抱歉", "不提供", "不支持", "无权", "建议咨询", "帮不了",
                   "无关", "不相关", "不符合", "不属于", "超出", "无法提供", "不能提供"]
FOLLOWUP_MARKERS = ["？", "?", "请问", "方便告知", "需要了解", "告诉我"]

RETRY_HINTS = ("403", "429", "Throttl", "throttl", "AllocationQuota", "timeout", "Timeout", "Connection")


# ----------------------------- 基础工具 -----------------------------
def norm(s: str) -> str:
    if not s:
        return ""
    table = str.maketrans("％～－–—−　，。；：（）", "%~---- ,.;:()")
    return "".join(str(s).translate(table).split()).lower()


def dataset_sha256() -> str:
    return hashlib.sha256(DATASET.read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def current_model() -> str:
    try:
        from utils.config_handler import rag_conf
        return rag_conf["chat_model_name"]
    except Exception:
        return "unknown"


# ----------------------------- 评分 -----------------------------
def grade(case: dict, result: dict) -> dict:
    ans = norm(result["answer"])
    tools = result["tools"]
    checks = {}

    expect_tools = case.get("expect_tools") or []
    if expect_tools:
        if case.get("expect_tools_ordered"):
            ordered = [t for t in expect_tools if t != "fill_context_for_report"]
            extras = [t for t in expect_tools if t == "fill_context_for_report"]
            it = iter(tools)
            checks["tool_ok"] = all(t in it for t in ordered) and all(t in tools for t in extras)
        else:
            checks["tool_ok"] = all(t in tools for t in expect_tools)
    else:
        checks["tool_ok"] = True

    must = case.get("must_contain") or []
    checks["fact_ok"] = all(
        any(norm(alt) in ans for alt in (k if isinstance(k, list) else [k])) for k in must
    ) if must else True

    checks["refusal_ok"] = any(norm(m) in ans for m in REFUSAL_MARKERS) if case.get("expect_refusal") else True
    checks["followup_ok"] = any(norm(m) in ans for m in FOLLOWUP_MARKERS) if case.get("expect_followup") else True

    result["checks"] = checks
    result["pass"] = all(checks.values())
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


def run_agent(agent, case: dict) -> dict:
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
    tools = [tc.get("name", "") for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    answer = (msgs[-1].content if msgs and isinstance(msgs[-1].content, str) else "") or ""
    return {"tools": tools, "answer": answer.strip(), "latency": latency, "n_messages": len(msgs)}


def run_with_retry(fn, *a, retries: int = 3, delays=(5, 15, 30), **kw):
    """可重试错误（403/429/超时）指数退避；其余异常直接抛"""
    last = None
    for attempt in range(retries):
        try:
            out = fn(*a, **kw)
            if attempt and isinstance(out, dict):
                out["retries"] = attempt
            return out
        except Exception as e:
            msg = str(e)
            if not any(h in msg for h in RETRY_HINTS):
                raise
            last = e
            if attempt < retries - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                print(f"      ↻ 可重试错误，{wait}s 后重试（{attempt + 1}/{retries - 1}）：{msg[:80]}")
                time.sleep(wait)
    raise last


def hit_at_3(retriever, case: dict) -> bool:
    must = case.get("must_contain") or []
    if not must:
        return True
    docs = retriever.invoke(case["question"])
    joined = norm(" ".join(d.page_content for d in docs[:3]))
    return all(any(norm(alt) in joined for alt in (k if isinstance(k, list) else [k])) for k in must)


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
    tool_ok = [r for r in tool_cases if (r.get("checks") or {}).get("tool_ok")]
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


# ----------------------------- 主流程 -----------------------------
def preflight(model: str) -> tuple[bool, str]:
    """开跑前探一次额度：花费 1 个 token，避免整轮跑批被 403 打废"""
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
        m = ChatTongyi(model=model, max_tokens=1)
        m.invoke("ping")
        return True, ""
    except Exception as e:
        return False, str(e)[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="只跑检索层 hit@3")
    ap.add_argument("--baseline", default=str(EVAL_DIR / "baseline.json"), help="基线文件路径")
    ap.add_argument("--tag", default="", help="本次跑批的备注标签")
    ap.add_argument("--delay", type=float, default=0.3, help="用例之间的间隔秒数")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = (ROOT / baseline_path) if (ROOT / baseline_path).exists() else Path(args.baseline)

    model = current_model()
    dataset_hash = dataset_sha256()
    commit = git_commit()
    cases_all = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"模型: {model} | 数据集: {len(cases_all)} 条 (sha={dataset_hash}) | commit: {commit}")

    # ① 预检
    if not args.no_llm and not args.skip_preflight:
        ok, err = preflight(model)
        if not ok:
            print("\n✗ 预检失败，已中止（未消耗跑批额度）")
            print(f"  模型: {model}")
            print(f"  原因: {err}")
            if "AllocationQuota" in err or "403" in err:
                print("  处理: 百炼控制台充值/关闭“仅免费额度”，或换 config/rag.yml 里的 chat_model_name")
            sys.exit(2)
        print("✓ 预检通过（额度可用）")

    cases = cases_all
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    print(f"本次用例: {len(cases)}")

    from rag.vector_store import VectorStoreService
    retriever = VectorStoreService().get_retriever()

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
                rec.update(run_with_retry(run_agent, agent, case))
                rec = grade(case, rec)
            else:
                rec["checks"], rec["pass"], rec["layer"] = {}, rec.get("hit3", True), ""
            tag = "PASS" if rec.get("pass") else "FAIL"
            extra = f" retries={rec['retries']}" if rec.get("retries") else ""
            print(f"[{i}/{len(cases)}] {case['id']:<6} {tag}  {rec.get('layer', '')}{extra}")
        except Exception as e:
            rec.update({"pass": False, "layer": "系统层", "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(cases)}] {case['id']:<6} ERROR {str(e)[:120]}")
        results.append(rec)
        if args.delay:
            time.sleep(args.delay)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = RESULTS_DIR / f"run_{ts}.jsonl"
    out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    cur = summarize(results)
    errors = [r for r in results if r.get("layer") == "系统层"]

    # 基线对比（模型不同则拒绝横比）
    base = None
    if baseline_path.exists() and not args.save_baseline:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        base_model = raw.get("model")
        if base_model and base_model != model:
            print(f"\n⚠️ 基线模型为 {base_model}，本次为 {model}：指标不可横比，已忽略基线数值。")
        else:
            base = raw.get("metrics")

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

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "dataset_sha256": dataset_hash,
        "dataset_size": len(cases_all),
        "git_commit": commit,
        "tag": args.tag,
        "errors": len(errors),
        "metrics": cur,
        "results_file": str(out_file),
    }

    if args.save_baseline or not baseline_path.exists():
        if errors:
            print(f"\n✗ 存在 {len(errors)} 条系统级错误，拒绝写入基线（避免污染）")
        else:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n已写入基线: {baseline_path}")

    if errors:
        print(f"\n⚠️ 本轮有 {len(errors)} 条系统级错误（额度/网络），结论仅供参考")
    print(f"\n明细: {out_file}")


if __name__ == "__main__":
    main()
