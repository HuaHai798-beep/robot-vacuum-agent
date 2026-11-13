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

from utils.logger_handler import logger  # noqa: E402

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


def load_rerank_price() -> dict:
    """重排单价（元/1K tokens 或 元/次）。留 0 表示未计价，报告里会标注。"""
    try:
        import yaml
        data = yaml.safe_load(PRICES.read_text(encoding="utf-8")) or {}
        return data.get("rerank", {}) or {}
    except Exception:
        return {}


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
    tool_outputs = [str(m.content) for m in msgs if getattr(m, "type", "") == "tool" and m.content]
    return {
        "tools": [c["name"] for c in tool_calls],
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,     # v1.9：工具返回（报告类用例的数字依据在这里）
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
                logger.error(f"[eval]不可重试错误，直接失败：{type(e).__name__}: {msg[:200]}")
                raise
            last = e
            if attempt < retries - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                logger.warning(f"[eval]可重试错误（{attempt + 1}/{retries}），{wait}s 后重试：{msg[:120]}")
                print(f"      ↻ 可重试错误，{wait}s 后重试：{msg[:70]}")
                time.sleep(wait)
    logger.error(f"[eval]重试 {retries} 次仍失败：{type(last).__name__}: {str(last)[:200]}")
    raise last


def retrieval_query(case: dict) -> str:
    """检索用查询：多轮用例要把历史用户话合并（否则"那它呢"这种查询没法检索）"""
    msgs = case.get("messages")
    if msgs:
        parts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
        q = " ".join(parts).strip()
        if q:
            return q
    return case["question"]


def retrieval_metrics(vector_store, case: dict, topk: int = 5, rerank: bool = False) -> dict:
    """只对"答案来自知识库 txt"的用例计算检索指标。

    v1.9：重排路径带 trace 埋点（方法/耗时/调用次数/是否降级），
    这样 P95 延迟里能区分「模型慢」还是「重排慢」。
    """
    src_file = (case.get("gold_source") or "")
    if not src_file.endswith(".txt"):
        return {"hit3": None, "src1": None, "src3": None, "src5": None, "mrr5": None,
                "rerank_latency": None, "rerank_calls": 0, "rerank_fallback": None,
                "rerank_method": None}

    query = retrieval_query(case)
    trace = {}
    if rerank:
        from rag.reranker import rerank_search
        docs = rerank_search(vector_store, query, topk=topk, candidates=10, trace=trace)
    else:
        docs = vector_store.similarity_search(query, k=topk)
        trace = {"rerank_method": "off", "rerank_calls": 0, "rerank_latency": 0.0, "rerank_fallback": False}
    out = {"hit3": None, "src1": False, "src3": False, "src5": False, "mrr5": 0.0,
           "chunk_hit3": None, "chunk_hit5": None, "chunk_rank": None, "chunk_mrr5": None}

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

    # 块级：gold_chunk（关键句）是否被封装的块命中
    gold_chunk = case.get("gold_chunk")
    if gold_chunk:
        c_rank = 0
        for i, d in enumerate(docs, 1):
            if norm(gold_chunk) in norm(d.page_content):
                c_rank = i
                break
        out["chunk_rank"] = c_rank or None
        out["chunk_hit3"] = 1 <= c_rank <= 3
        out["chunk_hit5"] = c_rank >= 1
        out["chunk_mrr5"] = round(1.0 / c_rank, 4) if c_rank else 0.0
    out.update(trace)          # v1.9：重排埋点（方法/耗时/次数/降级）
    return out


def numeric_check(answer: str, context: str) -> dict:
    """v1.9 规则型忠实度：回答里的「带单位数字事实」是否都能在参考资料里找到。

    和 judge 的区别：judge 是模型打分（有主观性、要花钱）；这里是纯规则（确定性、零成本），
    两者互补——judge 看语义，规则看数字。指标名：number_support_rate。
    """
    try:
        from rag.faithfulness_guard import extract_numeric_facts, _digits, _norm_digits
    except Exception as e:
        logger.warning(f"[numeric_check]后处理模块不可用，跳过数字检查：{type(e).__name__}: {e}")
        return {"numbers_total": None, "numbers_supported": None, "number_support_rate": None}

    if not answer:
        return {"numbers_total": 0, "numbers_supported": 0, "number_support_rate": None}

    facts = extract_numeric_facts(answer)
    if not facts:
        return {"numbers_total": 0, "numbers_supported": 0, "number_support_rate": None}

    ctx_nums = _digits(context)
    ok = sum(1 for f in facts if all(n in ctx_nums for n in f["nums"]))
    unsupported = [f["raw"] for f in facts if any(n not in ctx_nums for n in f["nums"])]
    return {
        "numbers_total": len(facts),
        "numbers_supported": ok,
        "number_support_rate": round(ok / len(facts) * 100, 1),
        "numbers_unsupported": unsupported[:8],
    }


JUDGE_PROMPT = """你是严格的质量审核员。只输出 JSON，不要解释。

用户问题：{q}
检索到的参考资料：{ctx}
模型回答：{a}

请判断两件事：
1) faithful：回答中的关键事实是否都能在参考资料中找到依据（有编造的数值/距离/时长/型号即为 false）；
2) relevance：回答是否直接解决了用户问题（5=完整直答，3=答了但跑偏或漏关键点，1=答非所问）。

只输出 JSON：{{"faithful": true/false, "relevance": 1-5, "unsupported": ["无依据的说法"]}}"""


def judge_answer(question: str, context: str, answer: str,
                 provider: str | None = None, model: str | None = None) -> dict:
    from model.factory import build_chat_model

    llm = build_chat_model(max_tokens=260, provider=provider, model=model)
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


def summarize(results: list, price: dict, rerank_price: dict | None = None) -> dict:
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
    chunk_cases = [r for r in results if r.get("chunk_mrr5") is not None]
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

    # ---- v1.9：重排埋点汇总 ----
    rr = [r for r in results if r.get("rerank_calls")]
    rr_lat = [r["rerank_latency"] for r in results if r.get("rerank_latency") is not None]
    rr_fallback = sum(1 for r in results if r.get("rerank_fallback"))
    rp = rerank_price or {}
    rerank_unit_priced = bool(rp.get("call") or rp.get("input"))

    # ---- v1.9：规则型数字依据率 ----
    num_cases = [r for r in results if r.get("numbers_total")]
    num_total = sum(r.get("numbers_total") or 0 for r in num_cases)
    num_ok = sum(r.get("numbers_supported") or 0 for r in num_cases)

    base_cost = cost(avg_ti, avg_to)
    rerank_cost = float(rp.get("call") or 0.0) * (1 if rr else 0)

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
        "chunk_recall3": round(sum(1 for r in chunk_cases if r["chunk_hit3"]) / len(chunk_cases) * 100, 1) if chunk_cases else None,
        "chunk_recall5": round(sum(1 for r in chunk_cases if r.get("chunk_hit5")) / len(chunk_cases) * 100, 1) if chunk_cases else None,
        "chunk_mrr5": round(statistics.mean([r["chunk_mrr5"] for r in chunk_cases]), 3) if chunk_cases else None,
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
        "cost_per_query": round(base_cost, 5),
        "cost_per_1k": round(base_cost * 1000, 2),
        # v1.9 新增
        "number_support_rate": round(num_ok / num_total * 100, 1) if num_total else None,
        "numbers_total": num_total,
        "numbers_unsupported": num_total - num_ok,
        "rerank_calls": len(rr),
        "rerank_fallbacks": rr_fallback,
        "avg_rerank_latency": round(statistics.mean(rr_lat), 3) if rr_lat else None,
        "p95_rerank_latency": p95(rr_lat) if rr_lat else None,
        "cost_rerank_per_query": round(rerank_cost, 5),
        "cost_total_per_query": round(base_cost + rerank_cost, 5),
        "cost_rerank_priced": rerank_unit_priced,
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
        ("数字依据率(规则)", "number_support_rate", "{:.1f}%"),
        ("答案相关性(≥3.5)", "relevance_pass", "{:.1f}%"),
        ("相关性均分", "relevance_avg", "{:.2f}"),
        ("检索 hit@3", "hit_at_3", "{:.1f}%"),
        ("来源 src@1", "src_at_1", "{:.1f}%"),
        ("来源 src@3", "src_at_3", "{:.1f}%"),
        ("来源 src@5", "src_at_5", "{:.1f}%"),
        ("MRR@5(文件级)", "mrr5", "{:.3f}"),
        ("块级 chunk_recall@3", "chunk_recall3", "{:.1f}%"),
        ("块级 chunk_recall@5", "chunk_recall5", "{:.1f}%"),
        ("块级 chunk_MRR@5", "chunk_mrr5", "{:.3f}"),
        ("工具调用准确率", "tool_call_acc", "{:.1f}%"),
        ("工具参数准确率", "tool_arg_acc", "{:.1f}%"),
        ("P50/P95延迟", "p95_latency", "{:.2f}"),
        ("平均tokens(入/出)", "avg_tokens", "{:.0f}"),
        ("单次成本(元)", "cost_per_query", "{:.5f}"),
        ("单次成本(含重排)", "cost_total_per_query", "{:.5f}"),
        ("重排延迟均值/P95", "avg_rerank_latency", "{:.3f}"),
        ("重排降级次数", "rerank_fallbacks", "{:.0f}"),
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
    ap.add_argument("--judge-provider", default="", help="评审模型 provider（默认同 agent）")
    ap.add_argument("--judge-model", default="", help="评审模型名（默认同 agent），用于交叉评审")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--baseline", default=str(EVAL_DIR / "baseline.json"))
    ap.add_argument("--compare", default="", help="用于对比的历史基线（不写回），默认同 --baseline")
    ap.add_argument("--tag", default="")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--no-rerank", action="store_true", help="强制关闭本地重排")
    ap.add_argument("--rerank", action="store_true", help="强制开启本地重排（默认跟随 chroma.yml）")
    ap.add_argument("--no-numbers", action="store_true", help="关闭规则型数字依据率检查（省一次检索）")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    model = current_model()
    price = load_price(model)
    rerank_price = load_rerank_price()
    ds_hash = dataset_sha256()
    commit = git_commit()
    all_cases = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"判据: {JUDGE_VERSION} | 模型: {model} | 价格: 入{price['input']}/出{price['output']} 元每千token | 数据集: {len(all_cases)} 条 (sha={ds_hash})")

    if not args.no_llm and not args.skip_preflight:
        ok, err = preflight(model)
        if not ok:
            logger.error(f"[eval]预检失败，中止跑批（未消耗额度）：{err}")
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

    from utils.config_handler import chroma_conf, rag_conf
    from rag.vector_store import VectorStoreService
    _vst = VectorStoreService()
    vector_store = _vst.vector_store
    retriever = _vst.get_retriever()

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
            use_rerank = args.rerank or (not args.no_rerank and bool(chroma_conf.get("rerank_enabled")))
            rec.update(retrieval_metrics(vector_store, case, rerank=use_rerank))
            if not args.no_llm:
                from rag import rag_service as _rag_mod
                _rag_mod.take_rag_trace()          # 清空上一个用例的检索轨迹
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
                if rec.get("answer") and (args.judge or not args.no_numbers):
                    # v1.9：评分依据 = 模型实际看到的资料（RAG 上下文 + 所有工具返回）
                    trace = _rag_mod.take_rag_trace()
                    ctx_parts = [t["context"] for t in trace if t.get("context")]
                    rec["eval_ctx_docs"] = sum(int(t.get("n_docs") or 0) for t in trace)
                    tool_outs = [str(t)[:2000] for t in (rec.get("tool_outputs") or [])]
                    rec["eval_ctx_tools"] = len(tool_outs)
                    eval_ctx = "\n".join(ctx_parts + tool_outs)
                    if not eval_ctx:
                        logger.warning(f"[eval] {case['id']} 未捕获检索/工具上下文，回退到独立检索")
                        docs = retriever.invoke(case["question"])[:3]
                        eval_ctx = "\n".join(d.page_content for d in docs)
                    if not args.no_numbers:
                        rec.update(numeric_check(rec["answer"], eval_ctx))
                    if args.judge:
                        try:
                            j = judge_answer(case["question"], eval_ctx, rec["answer"],
                                             args.judge_provider or None, args.judge_model or None)
                            rec.update(j)
                        except Exception as e:
                            rec["judge_error"] = str(e)[:200]
            else:
                rec["checks"], rec["layer"] = {}, ""
                rec["pass"] = True if rec.get("hit3") is None else bool(rec["hit3"])
            print(f"[{i}/{len(cases)}] {case['id']:<6} {'PASS' if rec.get('pass') else 'FAIL'}  {rec.get('layer','')}  tok={rec.get('tokens_in',0)}/{rec.get('tokens_out',0)}")
        except Exception as e:
            rec.update({"pass": False, "layer": "系统层", "error": f"{type(e).__name__}: {e}"})
            logger.error(f"[eval] {case['id']} 用例执行异常：{type(e).__name__}: {e}", exc_info=True)
            print(f"[{i}/{len(cases)}] {case['id']:<6} ERROR {str(e)[:110]}")
        results.append(rec)
        if args.delay:
            time.sleep(args.delay)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = RESULTS_DIR / f"run_{ts}.jsonl"
    out_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    cur = summarize(results, price, rerank_price)
    if args.repeat > 1:
        stable = [r for r in results if "stable" in r]
        cur["stability"] = round(sum(1 for r in stable if r["stable"]) / len(stable) * 100, 1) if stable else None
    errors = [r for r in results if r.get("layer") == "系统层"]

    base = None
    compare_path = Path(args.compare) if args.compare else baseline_path
    if compare_path.exists():
        raw = json.loads(compare_path.read_text(encoding="utf-8"))
        if raw.get("model") and raw["model"] != model:
            print(f"\n⚠️ 基线模型 {raw['model']} ≠ 本次 {model}：不可横比，忽略基线。")
        elif raw.get("judge_version") and raw["judge_version"] != JUDGE_VERSION:
            print(f"\n⚠️ 基线判据 {raw['judge_version']} ≠ 本次 {JUDGE_VERSION}：不可横比，忽略基线。")
        else:
            base = dict(raw.get("metrics") or {}, judge_version=raw.get("judge_version"))

    print_summary(cur, base, JUDGE_VERSION)
    if cur.get("rerank_calls") and not cur.get("cost_rerank_priced"):
        print("⚠️ 重排成本未计价（eval/prices.yml 中 rerank.call / rerank.input 为 0），"
              "当前成本只含对话 tokens，别当全成本用。")

    if args.no_llm and any(r.get("chunk_mrr5") is not None for r in results):
        print("\n逐条检索明细（no-llm 模式）：")
        print(f"{'id':<6}{'hit@3':>7}{'src@3':>7}{'块召回@3':>10}{'块排名':>8}{'块MRR':>8}")
        for r in results:
            if r.get("mrr5") is None:
                continue
            print(f"{r['id']:<6}{str(r.get('hit3')):>7}{str(r.get('src3')):>7}"
                  f"{str(r.get('chunk_hit3')):>10}{str(r.get('chunk_rank')):>8}{str(r.get('chunk_mrr5')):>8}")

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
        "judge_provider": args.judge_provider or "",
        "judge_model": args.judge_model or "",
        "errors": len(errors),
        "metrics": cur,
        # v1.9：把重排配置与单价写进基线，保证「同一份基线 = 同一套检索配置」
        "rerank": {
            "enabled": bool(chroma_conf.get("rerank_enabled")),
            "method": chroma_conf.get("rerank_method"),
            "model": chroma_conf.get("rerank_model"),
            "candidates": chroma_conf.get("rerank_candidates"),
            "alpha": chroma_conf.get("rerank_bm25_alpha"),
        },
        "rerank_price": rerank_price,
        "faithfulness_guard": bool(rag_conf.get("faithfulness_guard", True)),
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
