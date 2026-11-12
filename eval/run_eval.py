# -*- coding: utf-8 -*-
"""评测跑批 v3（六类指标 + 判据版本 + 企业级加固）

指标覆盖四层：
    检索层：hit@3（关键词覆盖）、src@3（来源文件命中）
    工具层：调用准确率、参数准确率
    生成层：通过率、忠实度（--judge，LLM-as-Judge）
    工程层：P50/P95 延迟、token 消耗、系统错误数、稳定性（--repeat）

用法（项目根目录）：
    python eval/run_eval.py                       # 全量，和基线对比
    python eval/run_eval.py --limit 8             # 先验证判据逻辑
    python eval/run_eval.py --no-llm              # 只跑检索层（不耗对话额度）
    python eval/run_eval.py --judge               # 加忠实度评分（额外消耗）
    python eval/run_eval.py --repeat 3            # 稳定性：每例跑 3 次
    python eval/run_eval.py --baseline eval/baseline_qwen_plus_v3.json --save-baseline
"""
import argparse
import hashlib
import json
import re
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

# 判据版本：任何"评分逻辑/关键词/归一化"变更都要 +1，跨版本分数不可横比
JUDGE_VERSION = "j4"

REFUSAL_MARKERS = ["无法", "不能", "不可以", "抱歉", "不提供", "不支持", "无权", "建议咨询", "帮不了",
                   "无关", "不相关", "不符合", "不属于", "超出", "无法提供", "不能提供", "不在服务范围"]
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


def total_tokens(messages) -> int:
    """累加所有 AI 消息的 token 用量。
    LangChain 1.x 优先取 usage_metadata；ChatTongyi 实际放在 response_metadata.token_usage。
    """
    n = 0
    for m in messages or []:
        u = getattr(m, "usage_metadata", None)
        if isinstance(u, dict) and u.get("total_tokens"):
            n += int(u["total_tokens"])
            continue
        rm = getattr(m, "response_metadata", None) or {}
        tu = rm.get("token_usage") or {}
        n += int(tu.get("total_tokens") or 0)
    return n


def arg_match(actual, expect) -> bool:
    """expect_args 的匹配规则：
    "*"        → 该参数必须存在
    {"regex":} → 正则匹配
    "abc"      → 归一化后相等或包含
    """
    if expect == "*":
        return actual not in (None, "")
    if isinstance(expect, dict) and "regex" in expect:
        return bool(re.search(expect["regex"], str(actual or "")))
    return norm(expect) in norm(actual) or norm(actual) == norm(expect)


# ----------------------------- 评分 -----------------------------
def grade(case: dict, result: dict) -> dict:
    ans = norm(result["answer"])
    checks = {}

    # 1) 工具调用
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

    # 2) 工具参数
    expect_args = case.get("expect_args") or {}
    if expect_args and checks["tool_ok"]:
        ok = True
        for tool_name, wanted in expect_args.items():
            calls = [c for c in result["tool_calls"] if c["name"] == tool_name]
            if not calls:
                ok = False
                break
            args = calls[0].get("args") or {}
            for field, expect in (wanted if isinstance(wanted, dict) else {k: v for k, v in wanted.items()}).items():
                if not arg_match(args.get(field), expect):
                    ok = False
                    break
            if not ok:
                break
        checks["args_ok"] = ok
    else:
        checks["args_ok"] = True

    # 3) 关键事实
    must = case.get("must_contain") or []
    checks["fact_ok"] = all(
        any(norm(alt) in ans for alt in (k if isinstance(k, list) else [k])) for k in must
    ) if must else True

    # 4) 禁止出现（安全类）
    forbidden = case.get("must_not_contain") or []
    checks["no_forbidden"] = not any(norm(k) in ans for k in forbidden)

    # 5) 行为
    checks["refusal_ok"] = any(norm(m) in ans for m in REFUSAL_MARKERS) if case.get("expect_refusal") else True
    checks["followup_ok"] = any(norm(m) in ans for m in FOLLOWUP_MARKERS) if case.get("expect_followup") else True

    result["checks"] = checks
    result["pass"] = all(checks.values())

    # 分层归因（顺序很重要：行为类问题优先归因）
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
    tool_calls = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            tool_calls.append({"name": tc.get("name", ""), "args": tc.get("args") or {}})
    answer = (msgs[-1].content if msgs and isinstance(msgs[-1].content, str) else "") or ""
    return {
        "tools": [c["name"] for c in tool_calls],
        "tool_calls": tool_calls,
        "answer": answer.strip(),
        "latency": latency,
        "tokens": total_tokens(msgs),
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


def retrieval_hits(retriever, case: dict) -> tuple[bool | None, bool | None]:
    """返回 (hit@3 关键词覆盖, src@3 来源文件命中)
    只有"答案来源是知识库 txt"的用例才参与检索指标；CSV/天气等外部数据源不参与。
    """
    src_file = case.get("gold_source") or ""
    if not src_file.endswith(".txt"):
        return None, None
    docs = retriever.invoke(case["question"])[:3]
    hit_kw = None
    must = case.get("must_contain") or []
    if must:
        joined = norm(" ".join(d.page_content for d in docs))
        hit_kw = all(any(norm(alt) in joined for alt in (k if isinstance(k, list) else [k])) for k in must)
    src = case.get("gold_source")
    hit_src = None
    if src:
        joined_src = " ".join(str((d.metadata or {}).get("source", "")) for d in docs)
        hit_src = norm(src) in norm(joined_src)
    return hit_kw, hit_src


def judge_faithfulness(model_name: str, question: str, context: str, answer: str) -> dict:
    """LLM-as-Judge：判断回答是否完全基于检索到的参考资料"""
    from langchain_community.chat_models.tongyi import ChatTongyi
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是严格的事实核查员。只输出 JSON，不要解释。"),
        ("human", """用户问题：{q}
检索到的参考资料：{ctx}
模型的回答：{a}

判断回答中的关键事实是否都能在参考资料中找到依据。
只输出 JSON：{{"faithful": true/false, "unsupported": ["无依据的说法"]}}"""),
    ])
    llm = ChatTongyi(model=model_name, max_tokens=200)
    out = (prompt | llm).invoke({"q": question, "ctx": context[:6000], "a": answer[:3000]}).content
    m = re.search(r"\{.*\}", out, re.S)
    data = json.loads(m.group(0)) if m else {}
    return {"faithful": bool(data.get("faithful", False)), "unsupported": data.get("unsupported", [])}


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
    args_cases = [r for r in results if r.get("expect_args")]
    hits = [r for r in results if r.get("hit3") is not None]
    srcs = [r for r in results if r.get("src3") is not None]
    faith = [r for r in results if r.get("faithful") is not None]
    toks = [r.get("tokens", 0) for r in results if r.get("tokens")]

    return {
        "n": total,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "by_category": {k: {"n": v["n"], "pass_rate": round(v["pass"] / v["n"] * 100, 1)} for k, v in by_cat.items()},
        "tool_call_acc": round(sum(1 for r in tool_cases if (r.get("checks") or {}).get("tool_ok")) / len(tool_cases) * 100, 1) if tool_cases else None,
        "tool_arg_acc": round(sum(1 for r in args_cases if (r.get("checks") or {}).get("args_ok")) / len(args_cases) * 100, 1) if args_cases else None,
        "hit_at_3": round(sum(1 for r in hits if r["hit3"]) / len(hits) * 100, 1) if hits else None,
        "src_at_3": round(sum(1 for r in srcs if r["src3"]) / len(srcs) * 100, 1) if srcs else None,
        "faithfulness": round(sum(1 for r in faith if r["faithful"]) / len(faith) * 100, 1) if faith else None,
        "avg_latency": round(statistics.mean([r.get("latency", 0) for r in results]), 2) if results else 0,
        "p95_latency": p95([r.get("latency", 0) for r in results]),
        "avg_tokens": round(statistics.mean(toks)) if toks else 0,
        "total_tokens": sum(toks) if toks else 0,
    }


def print_summary(cur: dict, base: dict | None, judge_version: str):
    def delta(key, fmt="{:.1f}"):
        if not base or base.get(key) is None or cur.get(key) is None:
            return ""
        d = cur[key] - base[key]
        return f" ({'+' if d >= 0 else ''}{fmt.format(d)})"

    rows = [
        ("总体通过率", "pass_rate", "{:.1f}%"),
        ("工具调用准确率", "tool_call_acc", "{:.1f}%"),
        ("工具参数准确率", "tool_arg_acc", "{:.1f}%"),
        ("检索 hit@3", "hit_at_3", "{:.1f}%"),
        ("来源 src@3", "src_at_3", "{:.1f}%"),
        ("忠实度", "faithfulness", "{:.1f}%"),
        ("平均耗时(s)", "avg_latency", "{:.2f}"),
        ("P95耗时(s)", "p95_latency", "{:.2f}"),
        ("平均tokens", "avg_tokens", "{:.0f}"),
    ]
    print("\n" + "=" * 66)
    print(f"判据版本: {judge_version}    基线版本: {(base or {}).get('judge_version', '-') if base else '-'}")
    print(f"{'指标':<16}{'基线':>12}{'本次':>12}")
    print("-" * 66)
    for label, key, fmt in rows:
        b = base.get(key) if base else None
        v = cur.get(key)
        left = "-" if b is None else fmt.format(b)
        right = "-" if v is None else fmt.format(v)
        print(f"{label:<14}{left:>12}{right:>12}{delta(key)}")
    print("-" * 66)
    for k, v in sorted(cur["by_category"].items()):
        print(f"  {k:<14} n={v['n']:<3} 通过率 {v['pass_rate']}%")
    print("=" * 66)


def preflight(model: str) -> tuple[bool, str]:
    try:
        from langchain_community.chat_models.tongyi import ChatTongyi
        ChatTongyi(model=model, max_tokens=1).invoke("ping")
        return True, ""
    except Exception as e:
        return False, str(e)[:400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge 忠实度评分（额外消耗）")
    ap.add_argument("--repeat", type=int, default=1, help="稳定性：每例重复次数")
    ap.add_argument("--baseline", default=str(EVAL_DIR / "baseline.json"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    model = current_model()
    ds_hash = dataset_sha256()
    commit = git_commit()
    all_cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"判据版本: {JUDGE_VERSION} | 模型: {model} | 数据集: {len(all_cases)} 条 (sha={ds_hash}) | commit: {commit}")

    if not args.no_llm and not args.skip_preflight:
        ok, err = preflight(model)
        if not ok:
            print("\n✗ 预检失败，已中止（未消耗跑批额度）")
            print(f"  原因: {err}")
            if "AllocationQuota" in err or "403" in err:
                print("  处理: 控制台充值/关闭“仅免费额度”，或改 config/rag.yml 的 chat_model_name")
            sys.exit(2)
        print("✓ 预检通过（额度可用）")

    cases = all_cases
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    print(f"本次用例: {len(cases)} | 重复: {args.repeat} | 忠实度评分: {'开' if args.judge else '关'}")

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
        runs = []
        try:
            rec["hit3"], rec["src3"] = retrieval_hits(retriever, case)
            if not args.no_llm:
                for _ in range(max(1, args.repeat)):
                    out = run_with_retry(run_agent, agent, case)
                    runs.append(out)
                rec.update(runs[0])
                rec = grade(case, rec)
                if args.repeat > 1:
                    passes = []
                    for out in runs:
                        tmp = grade(case, dict(out, id=case["id"], category=case["category"],
                                               question=case["question"], expect_tools=case.get("expect_tools") or []))
                        passes.append(tmp["pass"])
                    rec["stable"] = len(set(passes)) == 1
                    rec["run_passes"] = passes
                if args.judge and rec.get("answer"):
                    docs = retriever.invoke(case["question"])[:3]
                    ctx = "\n".join(d.page_content for d in docs)
                    try:
                        j = judge_faithfulness(model, case["question"], ctx, rec["answer"])
                        rec["faithful"] = j["faithful"]
                        rec["unsupported"] = j["unsupported"]
                    except Exception as e:
                        rec["faithful"] = None
                        rec["judge_error"] = str(e)[:200]
            else:
                rec["checks"], rec["layer"] = {}, ""
                rec["pass"] = True if rec.get("hit3") is None else bool(rec["hit3"])
            flag = "PASS" if rec.get("pass") else "FAIL"
            print(f"[{i}/{len(cases)}] {case['id']:<6} {flag}  {rec.get('layer', '')}  tok={rec.get('tokens', 0)}")
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

    cur = summarize(results)
    if args.repeat > 1:
        stable = [r for r in results if "stable" in r]
        cur["stability"] = round(sum(1 for r in stable if r["stable"]) / len(stable) * 100, 1) if stable else None
    errors = [r for r in results if r.get("layer") == "系统层"]

    base = None
    if baseline_path.exists() and not args.save_baseline:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
        if raw.get("model") and raw["model"] != model:
            print(f"\n⚠️ 基线模型为 {raw['model']}，本次为 {model}：不可横比，已忽略基线数值。")
        elif raw.get("judge_version") and raw["judge_version"] != JUDGE_VERSION:
            print(f"\n⚠️ 基线判据版本 {raw['judge_version']} ≠ 本次 {JUDGE_VERSION}：分数不可横比。")
        else:
            base = dict(raw.get("metrics") or {}, judge_version=raw.get("judge_version"))

    print_summary(cur, base, JUDGE_VERSION)

    failures = [r for r in results if not r["pass"]]
    if failures:
        print("\n失败用例：")
        for r in failures[:12]:
            print(f"  {r['id']:<6} [{r['layer']}] {r['question'][:40]}")
            if r.get("tools"):
                print(f"         工具: {r['tools']}")
            if r.get("answer"):
                print(f"         回答: {r['answer'][:70]}...")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_version": JUDGE_VERSION,
        "model": model,
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
