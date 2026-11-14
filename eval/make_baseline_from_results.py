# -*- coding: utf-8 -*-
"""用「已保存的模型输出」重建完整基线（只重跑判据，不重新调用模型）。

场景：v2.0 冻结前修正了某条用例的 must_contain（期望写错），但没有必要重烧一轮额度。
做法：同 regrade.py，但输出**完整 baseline payload**（含 judge/sha/size/commit），
      这样它仍然是一份合法的基线，并留下 *_regraded.jsonl 供追溯。

用法：
    python eval/make_baseline_from_results.py eval/results/run_xxx.jsonl --baseline eval/baseline_v20.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import (DATASET, JUDGE_VERSION, current_model, grade,  # noqa: E402
                      load_price, load_rerank_price, summarize)


def sha256_of_dataset() -> str:
    return hashlib.sha256(DATASET.read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cases = {c["id"]: c for c in (json.loads(l) for l in
                                  DATASET.read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = [json.loads(l) for l in Path(args.results_file).read_text(encoding="utf-8").splitlines() if l.strip()]

    side = Path(args.results_file).with_suffix(".meta.json")
    meta = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}

    out = []
    for r in rows:
        case = cases.get(r["id"], {})
        rec = dict(r)
        rec["expect_tools"] = case.get("expect_tools") or []
        rec["expect_args"] = case.get("expect_args") or {}
        if r.get("layer") == "系统层":
            out.append(rec)
            continue
        rec = grade({**case, "expect_args": rec["expect_args"]}, rec)
        out.append(rec)

    model = meta.get("model") or current_model()
    price = load_price(model)
    rerank_price = load_rerank_price()
    metrics = summarize(out, price, rerank_price)

    before = sum(1 for r in rows if r.get("pass"))
    after = sum(1 for r in out if r.get("pass"))
    errors = sum(1 for r in out if r.get("layer") == "系统层")

    regraded = Path(args.results_file).with_name(Path(args.results_file).stem + "_regraded.jsonl")
    regraded.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_version": meta.get("judge_version", JUDGE_VERSION),
        "judge": bool(meta.get("judge")),
        "judge_provider": meta.get("judge_provider", ""),
        "judge_model": meta.get("judge_model", ""),
        "model": model,
        "price": price,
        "rerank_price": rerank_price,
        "dataset_sha256": sha256_of_dataset(),
        "dataset_size": len(cases),
        "git_commit": git_commit(),
        "tag": args.tag,
        "errors": errors,
        "note": f"离线重打分基线：模型输出取自 {Path(args.results_file).name}，仅按最终判据重新评分（未重新调用模型）",
        "source_results": str(args.results_file),
        "regraded_results": str(regraded),
        "pass_before_regrade": before,
        "pass_after_regrade": after,
        "metrics": metrics,
    }
    Path(args.baseline).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"判据: {payload['judge_version']} | 用例: {len(out)} | 系统错误: {errors}")
    print(f"重打分前通过: {before}/{len(rows)} → 重打分后: {after}/{len(out)}  ({metrics['pass_rate']}%)")
    print(f"忠实度 {metrics['faithfulness']}% | 数字依据 {metrics['number_support_rate']}% | "
          f"hit@3 {metrics['hit_at_3']}% | chunk_recall@3 {metrics['chunk_recall3']}%")
    print("分类:", {k: v["pass_rate"] for k, v in metrics["by_category"].items()})
    print("已写入基线:", args.baseline)
    print("重打分明细:", regraded)


if __name__ == "__main__":
    main()
