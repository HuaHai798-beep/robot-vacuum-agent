# -*- coding: utf-8 -*-
"""评测跑批 v4（九指标 + 判据版本 + 成本核算 + 企业级加固）

指标分四层：
    生成层：通过率、忠实度、答案相关性（judge）
    检索层：hit@3（关键词覆盖）、src@1/3/5（来源命中）、MRR@5（文件级）
    工具层：调用准确率、参数准确率
    工程层：P50/P95 延迟、tokens_in/out、单次成本、系统错误数、稳定性（--repeat）

判据版本 JUDGE_VERSION：任何评分逻辑/关键词/归一化变更必须 +1，跨版本分数不可横比。

用法（项目根目录）：
    python eval/run_eval.py --limit 8              # 先验证判据逻辑
    python eval/run_eval.py --no-llm               # 只跑检索层（不耗对话额度）
    python eval/run_eval.py --judge                # 加忠实度 + 答案相关性
    python eval/run_eval.py --repeat 3 --judge     # 稳定性
    python eval/run_eval.py --judge --baseline eval/baseline_qwen_plus_v4.json --save-baseline
"""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
PRICES = EVAL_DIR / "prices.yml"

JUDGE_VERSION = "j5"

REFUSAL_MARKERS = ["无法", "不能", "不可以", "抱歉", "不提供", "不支持", "无权", "建议咨询", "帮不了",
                   "无关", "不相关", "不符合", "不属于", "超出", "无法提供", "不能提供", "不在服务范围",
                   "不在我的服务范围", "服务范围", "不具备", "仅能", "只能", "保密", "无法对外展示"]
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


def load_price(model: str) -> dict:
    try:
        import yaml
        data = yaml.safe_load(PRICES.read_text(encoding="utf-8"))
        return data.get(model, {"input": 0.0, "output": 0.0})
    except Exception:
        return {"input": 0.0, "output": 0.0}


def token_usage(messages) -> tuple[int, int]:
    """返回 (输入 tokens, 输出 tokens)。
    兼容三种来源：
      1) LangChain usage_metadata（input_tokens / output_tokens）
      2) ChatTongyi: response_metadata.token_usage.input_tokens / output_tokens
      3) ChatOpenAI(DeepSeek): prompt_tokens / completion_tokens
    """
    ti = to = 0
    for m in messages or []:
        u = getattr(m, "usage_metadata", None)
        if isinstance(u, dict) and (u.get("input_tokens") or u.get("output_tokens")):
            ti += int(u.get("input_tokens") or 0)
            to += int(u.get("output_tokens") or 0)
            continue
        tu = (getattr(m, "response_metadata", None) or {}).get("token_usage") or {}
        ti += int(tu.get("input_tokens") or tu.get("prompt_tokens") or 0)
        to += int(tu.get("output_tokens") or tu.get("completion_tokens") or 0)
    return ti, to


def arg_match(actual, expect) -> bool:
    if expect == "*":
        return actual not in (None, "")
    if isinstance(expect, dict) and "regex" in expect:
        return bool(re.search(expect["regex"], str(actual or "")))
    return norm(expect) in norm(actual) or norm(actual) == norm(expect)


# ----------------------------- 评分 -----------------------------
def grade(case: dict, result: dict) -> dict:
    ans = norm(result["answer"])
    checks = {}

    expect_tools = case.get("expect_tools") or []
    if expect_tools:
        if case.get("expect_tools_ordered"):
            ordered = [t for t in expect_tools if t != "fill_context_for_report"]
            extras = [t for t in expect_tools if t == "fill_context_for_report"]
            it = iter(result["tools"])
            checks["tool_ok"] = all(t in it for t in ordered) and all(t in result["tools"] for t in extras)
        else:
            checks["tool_ok"] = all(t in result["tools"] for t in expect_tools)
    else:
        checks["tool_ok"] = True

    expect_args = case.get("expect_args") or {}
    if expect_args and checks["tool_ok"]:
        ok = True
        for tool_name, wanted in expect_args.items():
            calls = [c for c in result["tool_calls"] if c["name"] == tool_name]
            if not calls:
                ok = False
                break
            args = calls[0].get("args") or {}
            for field, expect in (wanted or {}).items():
                if not arg_match(args.get(field), expect):
                    ok = False
                    break
            if not ok:
                break
        checks["args_ok"] = ok
    else:
        checks["args_ok"] = True

    must = case.get("must_contain") or []
    checks["fact_ok"] = all(
        any(norm(alt) in ans for alt in (k if isinstance(k, list) else [k])) for k in must
    ) if must else True

    forbidden = case.get("must_not_contain") or []
    checks["no_forbidden"] = not any(norm(k) in ans for k in forbidden)

    checks["refusal_ok"] = any(norm(m) in ans for m in REFUSAL_MARKERS) if case.get("expect_refusal") else True
    checks["followup_ok"] = any(norm(m) in ans for m in FOLLOWUP_MARKERS) if case.get("expect_followup") else True

    result["checks"] = checks
    result["pass"] = all(checks.values())

    if result["pass"]:
        result["layer"] = ""
    elif case.get("expect_refusal") and not checks["refusal_ok"]:
        result["layer"] = "拒答层"
    elif case.get("expect_followup") and not checks["followup_ok"]:
        result["layer"] = "追问层"
    elif not checks["no_forbidden"]:
        result["layer"] = "安全层"
    elif not checks["tool_ok"]:
        result["layer"] = "工具层"
    elif not checks["args_ok"]:
        result["layer"] = "工具参数层"
    elif not checks["fact_ok"]:
        result["layer"] = "检索/回答层"
    else:
        result["layer"] = "其他"
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
    tool_calls = [{"name": tc.get("name", ""), "args": tc.get("args") or {}}
                  for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    answer = (msgs[-1].content if msgs and isinstance(msgs[-1].content, str) else "") or ""
    ti, to = token_usage(msgs)
    return {
        "tools": [c["name"] for c in tool_calls],
        "tool_calls": tool_calls,
        "answer": answer.strip(),
        "latency": latency,
        "tokens_in": ti,
        "tokens_out": to,
        "tokens": ti + to,
        "n_messages": len(msgs),
    }


def run_with_retry(fn, *a, retries: int = 3, delays=(5, 15, 30), **kw):
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
                print(f"      ↻ 可重试错误，{wait}s 后重试：{msg[:70]}")
                time.sleep(wait)
    raise last


def retrieval_metrics(retriever, case: dict, topk: int = 5) -> dict:
    """只对"答案来自知识库 txt"的用例计算检索指标"""
    src_file = (case.get("gold_source") or "")
    if not src_file.endswith(".txt"):
        return {"hit3": None, "src1": None, "src3": None, "src5": None, "mrr5": None}

    docs = retriever.invoke(case["question"])[:topk]
    out = {"hit3": None, "src1": False, "src3": False, "src5": False, "mrr5": 0.0}

    must = case.get("must_contain") or []
    if must:
        joined = norm(" ".join(d.page_content for d in docs[:3]))
        out["hit3"] = all(any(norm(alt) in joined for alt in (k if isinstance(k, list) else [k])) for k in must)

    rank = 0
    for i, d in enumerate(docs, 1):
        if norm(src_file) in norm(str((d.metadata or {}).get("source", ""))):
            rank = i
            break
    out["src1"] = rank == 1
    out["src3"] = 1 <= rank <= 3
    out["src5"] = rank >= 1
    out["mrr5"] = round(1.0 / rank, 4) if rank else 0.0
    return out


JUDGE_PROMPT = """你是严格的质量审核员。只输出 JSON，不要解释。

用户问题：{q}
检索到的参考资料：{ctx}
模型回答：{a}

请判断两件事：
1) faithful：回答中的关键事实是否都能在参考资料中找到依据（有编造的数值/距离/时长/型号即为 false）；
2) relevance：回答是否直接解决了用户问题（5=完整直答，3=答了但跑偏或漏关键点，1=答非所问）。

只输出 JSON：{{"faithful": true/false, "relevance": 1-5, "unsupported": ["无依据的说法"]}}"""


def judge_answer(model_name: str, question: str, context: str, answer: str) -> dict:
    from model.factory import build_chat_model

    llm = build_chat_model(max_tokens=220)
    out = llm.invoke(JUDGE_PROMPT.format(q=question, ctx=context[:6000], a=answer[:3000])).content
    m = re.search(r"\{.*\}", out, re.S)
    data = json.loads(m.group(0)) if m else {}
    try:
        rel = float(data.get("relevance", 0))
    except Exception:
        rel = 0.0
    return {"faithful": bool(data.get("faithful", False)),
            "relevance": rel,
            "unsupported": data.get("unsupported", [])}


def p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return round(values[idx], 2)


def p50(values):
    return round(statistics.median(values), 2) if values else 0.0


def summarize(results: list, price: dict) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["category"], {"n": 0, "pass": 0})
        c["n"] += 1
        c["pass"] += 1 if r["pass"] else 0

    tool_cases = [r for r in results if r.get("expect_tools")]
    args_cases = [r for r in results if r.get("expect_args")]
    hit_cases = [r for r in results if r.get("hit3") is not None]
    src_cases = [r for r in results if r.get("mrr5") is not None]
    faith = [r for r in results if r.get("faithful") is not None]
    rel = [r for r in results if r.get("relevance")]
    toks = [r.get("tokens", 0) for r in results if r.get("tokens")]
    tin = [r.get("tokens_in", 0) for r in results if r.get("tokens")]
    tout = [r.get("tokens_out", 0) for r in results if r.get("tokens")]
    lat = [r.get("latency", 0) for r in results]

    def cost(ti, to):
        return ti / 1000 * price.get("input", 0) + to / 1000 * price.get("output", 0)

    avg_ti = statistics.mean(tin) if tin else 0
    avg_to = statistics.mean(tout) if tout else 0

    return {
        "n": total,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "by_category": {k: {"n": v["n"], "pass_rate": round(v["pass"] / v["n"] * 100, 1)} for k, v in by_cat.items()},
        "tool_call_acc": round(sum(1 for r in tool_cases if (r.get("checks") or {}).get("tool_ok")) / len(tool_cases) * 100, 1) if tool_cases else None,
        "tool_arg_acc": round(sum(1 for r in args_cases if (r.get("checks") or {}).get("args_ok")) / len(args_cases) * 100, 1) if args_cases else None,
        "hit_at_3": round(sum(1 for r in hit_cases if r["hit3"]) / len(hit_cases) * 100, 1) if hit_cases else None,
        "src_at_1": round(sum(1 for r in src_cases if r["src1"]) / len(src_cases) * 100, 1) if src_cases else None,
        "src_at_3": round(sum(1 for r in src_cases if r["src3"]) / len(src_cases) * 100, 1) if src_cases else None,
        "src_at_5": round(sum(1 for r in src_cases if r["src5"]) / len(src_cases) * 100, 1) if src_cases else None,
        "mrr5": round(statistics.mean([r["mrr5"] for r in src_cases]), 3) if src_cases else None,
        "faithfulness": round(sum(1 for r in faith if r["faithful"]) / len(faith) * 100, 1) if faith else None,
        "relevance_avg": round(statistics.mean([r["relevance"] for r in rel]), 2) if rel else None,
        "relevance_pass": round(sum(1 for r in rel if r["relevance"] >= 3.5) / len(rel) * 100, 1) if rel else None,
        "avg_latency": round(statistics.mean(lat), 2) if lat else 0,
        "p50_latency": p50(lat),
        "p95_latency": p95(lat),
        "avg_tokens_in": round(avg_ti),
        "avg_tokens_out": round(avg_to),
        "avg_tokens": round(statistics.mean(toks)) if toks else 0,
        "total_tokens": sum(toks) if toks else 0,
        "cost_per_query": round(cost(avg_ti, avg_to), 5),
        "cost_per_1k": round(cost(avg_ti, avg_to) * 1000, 2),
    }


def print_summary(cur: dict, base: dict | None, judge_version: str):
    def delta(key, fmt="{:.1f}"):
        if not base or base.get(key) is None or cur.get(key) is None:
            return ""
        d = cur[key] - base[key]
        return f" ({'+' if d >= 0 else ''}{fmt.format(d)})"

    rows = [
        ("总体通过率", "pass_rate", "{:.1f}%"),
        ("忠实度", "faithfulness", "{:.1f}%"),
        ("答案相关性(≥3.5)", "relevance_pass", "{:.1f}%"),
        ("相关性均分", "relevance_avg", "{:.2f}"),
        ("检索 hit@3", "hit_at_3", "{:.1f}%"),
        ("来源 src@1", "src_at_1", "{:.1f}%"),
        ("来源 src@3", "src_at_3", "{:.1f}%"),
        ("来源 src@5", "src_at_5", "{:.1f}%"),
        ("MRR@5", "mrr5", "{:.3f}"),
        ("工具调用准确率", "tool_call_acc", "{:.1f}%"),
        ("工具参数准确率", "tool_arg_acc", "{:.1f}%"),
        ("P50/P95延迟", "p95_latency", "{:.2f}"),
        ("平均tokens(入/出)", "avg_tokens", "{:.0f}"),
        ("单次成本(元)", "cost_per_query", "{:.5f}"),
    ]
    print("\n" + "=" * 70)
    print(f"判据版本: {judge_version}    基线判据: {(base or {}).get('judge_version', '-') if base else '-'}")
    print(f"{'指标':<18}{'基线':>12}{'本次':>12}")
    print("-" * 70)
    for label, key, fmt in rows:
        b = base.get(key) if base else None
        v = cur.get(key)
        left = "-" if b is None else fmt.format(b)
        right = "-" if v is None else fmt.format(v)
        print(f"{label:<16}{left:>14}{right:>14}{delta(key)}")
    io = f"{cur['avg_tokens_in']}/{cur['avg_tokens_out']}"
    print(f"{'平均输入/输出tokens':<14}{'-':>14}{io:>14}")
    print("-" * 70)
    for k, v in sorted(cur["by_category"].items()):
        print(f"  {k:<14} n={v['n']:<3} 通过率 {v['pass_rate']}%")
    print("=" * 70)


def preflight(model: str) -> tuple[bool, str]:
    try:
        from model.factory import build_chat_model
        build_chat_model(max_tokens=1).invoke("ping")
        return True, ""
    except Exception as e:
        return False, str(e)[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--baseline", default=str(EVAL_DIR / "baseline.json"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    model = current_model()
    price = load_price(model)
    ds_hash = dataset_sha256()
    commit = git_commit()
    all_cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"判据: {JUDGE_VERSION} | 模型: {model} | 价格: 入{price['input']}/出{price['output']} 元每千token | 数据集: {len(all_cases)} 条 (sha={ds_hash})")

    if not args.no_llm and not args.skip_preflight:
        ok, err = preflight(model)
        if not ok:
            print("\n✗ 预检失败，已中止（未消耗跑批额度）")
            print(f"  原因: {err}")
            sys.exit(2)
        print("✓ 预检通过（额度可用）")

    cases = all_cases
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    print(f"本次用例: {len(cases)} | 重复: {args.repeat} | judge: {'开' if args.judge else '关'}")

    from rag.vector_store import VectorStoreService
    retriever = VectorStoreService().get_retriever()

    agent = None
    if not args.no_llm:
        from agent.react_agent import ReactAgent
        agent = ReactAgent().agent

    results = []
    for i, case in enumerate(cases, 1):
        rec = {"id": case["id"], "category": case["category"], "question": case["question"],
               "note": case.get("note", ""), "expect_tools": case.get("expect_tools") or [],
               "expect_args": case.get("expect_args") or {}}
        try:
            rec.update(retrieval_metrics(retriever, case))
            if not args.no_llm:
                runs = [run_with_retry(run_agent, agent, case) for _ in range(max(1, args.repeat))]
                rec.update(runs[0])
                rec = grade(case, rec)
                if args.repeat > 1:
                    passes = []
                    for out in runs:
                        tmp = dict(out, id=case["id"], category=case["category"],
                                   question=case["question"], expect_tools=case.get("expect_tools") or [],
                                   expect_args=case.get("expect_args") or {})
                        passes.append(grade(case, tmp)["pass"])
                    rec["stable"] = len(set(passes)) == 1
                    rec["run_passes"] = passes
                if args.judge and rec.get("answer"):
                    docs = retriever.invoke(case["question"])[:3]
                    ctx = "\n".join(d.page_content for d in docs)
                    try:
                        j = judge_answer(model, case["question"], ctx, rec["answer"])
                        rec.update(j)
                    except Exception as e:
                        rec["judge_error"] = str(e)[:200]
            else:
                rec["checks"], rec["layer"] = {}, ""
                rec["pass"] = True if rec.get("hit3") is None else bool(rec["hit3"])
            print(f"[{i}/{len(cases)}] {case['id']:<6} {'PASS' if rec.get('pass') else 'FAIL'}  {rec.get('layer','')}  tok={rec.get('tokens_in',0)}/{rec.get('tokens_out',0)}")
        except Exception as e:
            rec.update({"pass": False, "layer": "系统层", "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(cases)}] {case['id']:<6} ERROR {str(e)[:110]}")
        results.append(rec)
        if args.delay:
            time.sleep(args.delay)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = RESULTS_DIR / f"run_{ts}.jsonl"
    out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    cur = summarize(results, price)
    if args.repeat > 1:
        stable = [r for r in results if "stable" in r]
        cur["stability"] = round(sum(1 for r in stable if r["stable"]) / len(stable) * 100, 1) if stable else None
    errors = [r for r in results if r.get("layer") == "系统层"]

    base = None
    if baseline_path.exists() and not args.save_baseline:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        if raw.get("model") and raw["model"] != model:
            print(f"\n⚠️ 基线模型 {raw['model']} ≠ 本次 {model}：不可横比，忽略基线。")
        elif raw.get("judge_version") and raw["judge_version"] != JUDGE_VERSION:
            print(f"\n⚠️ 基线判据 {raw['judge_version']} ≠ 本次 {JUDGE_VERSION}：不可横比，忽略基线。")
        else:
            base = dict(raw.get("metrics") or {}, judge_version=raw.get("judge_version"))

    print_summary(cur, base, JUDGE_VERSION)

    failures = [r for r in results if not r["pass"]]
    if failures:
        print("\n失败用例：")
        for r in failures[:12]:
            print(f"  {r['id']:<6} [{r['layer']}] {r['question'][:36]}")
            if r.get("unsupported"):
                print(f"         无依据: {r['unsupported'][:2]}")
            if r.get("answer"):
                print(f"         回答: {r['answer'][:60]}...")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_version": JUDGE_VERSION,
        "model": model,
        "price": price,
        "dataset_sha256": ds_hash,
        "dataset_size": len(all_cases),
        "git_commit": commit,
        "tag": args.tag,
        "repeat": args.repeat,
        "judge": args.judge,
        "errors": len(errors),
        "metrics": cur,
        "results_file": str(out_file),
    }
    if args.save_baseline or not baseline_path.exists():
        if errors:
            print(f"\n✗ 存在 {len(errors)} 条系统级错误，拒绝写入基线")
        else:
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n已写入基线: {baseline_path}")
    print(f"\n明细: {out_file}")


if __name__ == "__main__":
    main()
