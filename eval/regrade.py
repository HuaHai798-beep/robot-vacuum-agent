# -*- coding: utf-8 -*-
"""离线重打分：用当前判据重新评估某个已保存的 results/*.jsonl

用途：当"判据"（拒答词表、must_contain 等）被修正，但模型额度不足无法重跑时，
     可以直接对**同一批回答**重新评分，避免"为了验证判据再烧一轮额度"。

用法：
    python eval/regrade.py eval/results/run_20261113_1258.jsonl \
        --out eval/baseline_qwen_plus_v4_regraded.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import grade, summarize, load_price, JUDGE_VERSION, DATASET, current_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="", help="源运行使用的模型（默认取当前配置）")
    args = ap.parse_args()

    cases = {c["id"]: c for c in (json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = [json.loads(l) for l in Path(args.results_file).read_text(encoding="utf-8").splitlines() if l.strip()]

    out = []
    for r in rows:
        case = cases.get(r["id"], {})
        rec = dict(r)
        rec["expect_tools"] = case.get("expect_tools") or []
        rec["expect_args"] = case.get("expect_args") or {}
        if r.get("layer") == "系统层":          # 系统错误保持原样
            out.append(rec)
            continue
        rec = grade({**case, "expect_args": rec["expect_args"]}, rec)
        out.append(rec)

    model = args.model or current_model()
    price = load_price(model)
    metrics = summarize(out, price)

    before = sum(1 for r in rows if r.get("pass"))
    after = sum(1 for r in out if r.get("pass"))
    print(f"判据: {JUDGE_VERSION} | 用例: {len(out)}")
    print(f"重打分前通过: {before}/{len(rows)}  →  重打分后: {after}/{len(out)}")
    print(f"通过率: {metrics['pass_rate']}%  忠实度: {metrics['faithfulness']}%  "
          f"安全: {metrics['by_category'].get('safety', {}).get('pass_rate')}%  "
          f"拒答: {metrics['by_category'].get('refuse', {}).get('pass_rate')}%")
    print("分类:", {k: v["pass_rate"] for k, v in metrics["by_category"].items()})

    if args.out:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "judge_version": JUDGE_VERSION,
            "model": model,
            "price": price,
            "note": f"离线重打分：基于 {Path(args.results_file).name} 的模型输出，仅重跑判据，未重新调用模型",
            "source_results": str(args.results_file),
            "pass_before_regrade": before,
            "pass_after_regrade": after,
            "metrics": metrics,
        }
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写入:", args.out)


if __name__ == "__main__":
    main()
