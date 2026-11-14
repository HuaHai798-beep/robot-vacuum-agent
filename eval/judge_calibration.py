# -*- coding: utf-8 -*-
"""Judge 人工校准（v1.9）：把我的 20 条人工标注和 judge 的打分算一致率。

为什么要做：
    LLM-as-Judge 如果不校准，就只是"另一个模型的意见"。企业会问：
    「judge 和人工的一致率多少？」——没有这个数字，忠实度/相关性都不可信。

用法：
    python eval/judge_calibration.py --results eval/results/run_20261113_XXXX.jsonl
输出：
    控制台明细表 + eval/judge_calibration_report.json（写进报告引用）
"""
import argparse
import json
import statistics
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
LABELS = EVAL_DIR / "judge_calibration.json"
REPORT = EVAL_DIR / "judge_calibration_report.json"


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """二分类 Kappa（faithful: 1=忠实, 0=不忠实）"""
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe != 1 else 1.0


def resolve_meta(results_path: Path, baseline: str, cli_v: str, cli_m: str) -> dict:
    """追溯 judge 元数据：sidecar → baseline → results 行 → CLI 参数（v2.0 修复）。

    之前只从 results 行取，而 results 行并不携带这些字段 → 报告里永远是空字符串。
    """
    meta = {}
    side = results_path.with_suffix(".meta.json")
    if side.exists():
        try:
            meta.update({k: v for k, v in json.loads(side.read_text(encoding="utf-8")).items()
                         if k in {"judge_version", "judge_model", "judge_provider", "model"}})
        except Exception as e:
            print(f"[warn] sidecar 解析失败：{type(e).__name__}: {e}")
    if baseline and Path(baseline).exists() and not meta.get("judge_version"):
        try:
            b = json.loads(Path(baseline).read_text(encoding="utf-8"))
            meta.setdefault("judge_version", b.get("judge_version", ""))
            meta.setdefault("judge_model", b.get("judge_model", ""))
            meta.setdefault("model", b.get("model", ""))
        except Exception as e:
            print(f"[warn] baseline 解析失败：{type(e).__name__}: {e}")
    if cli_v:
        meta["judge_version"] = cli_v
    if cli_m:
        meta["judge_model"] = cli_m
    if not meta.get("judge_version"):
        print("[warn] 未能确定 judge_version（sidecar/baseline/CLI 都没有），报告里将留空")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="跑批明细 jsonl")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--baseline", default="", help="用于兜底读 judge 元数据的基线文件")
    ap.add_argument("--judge-version", default="", help="手工指定判据版本（最终兜底）")
    ap.add_argument("--judge-model", default="", help="手工指定评审模型（最终兜底）")
    args = ap.parse_args()

    meta = resolve_meta(Path(args.results), args.baseline, args.judge_version, args.judge_model)

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["items"]
    results = {}
    for line in Path(args.results).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            results[r["id"]] = r

    rows, faith_pairs, rel_pairs = [], [], []
    for lab in labels:
        r = results.get(lab["id"])
        if not r or r.get("faithful") is None or r.get("relevance") is None:
            rows.append({"id": lab["id"], "status": "缺 judge 结果"})
            continue
        jf = 1 if r["faithful"] else 0
        mf = 1 if lab["my_faithful"] else 0
        jr, mr = float(r["relevance"]), float(lab["my_relevance"])
        faith_pairs.append((jf, mf))
        rel_pairs.append((jr, mr))
        rows.append({
            "id": lab["id"],
            "judge_faithful": bool(jf), "my_faithful": bool(mf), "faith_agree": jf == mf,
            "judge_relevance": jr, "my_relevance": mr,
            "rel_diff": round(jr - mr, 2), "rel_pm1": abs(jr - mr) <= 1,
        })

    rel = [r for r in rows if "rel_diff" in r]
    faith = [r for r in rows if "faith_agree" in r]
    if not rel:
        raise SystemExit("没有可比对的样本：检查 results 与 labels 的 id 是否对得上")

    rel_exact = round(sum(1 for r in rel if r["rel_diff"] == 0) / len(rel) * 100, 1)
    rel_pm1 = round(sum(1 for r in rel if r["rel_pm1"]) / len(rel) * 100, 1)
    rel_mae = round(statistics.mean(abs(r["rel_diff"]) for r in rel), 2)
    faith_acc = round(sum(1 for r in faith if r["faith_agree"]) / len(faith) * 100, 1)
    kappa = cohen_kappa(faith_pairs)

    print("=" * 78)
    print(f"人工标注 {len(labels)} 条 | 可比对 {len(rel)} 条 | judge: {meta.get('judge_model') or '?'} / {meta.get('judge_version') or '?'}")
    print("-" * 78)
    print(f"{'id':<6}{'judge忠实':>10}{'人工忠实':>10}{'一致':>6}{'judge相关':>10}{'人工相关':>10}{'差值':>6}")
    for r in rows:
        if "rel_diff" not in r:
            print(f"{r['id']:<6}  (跳过: {r.get('status', '')})")
            continue
        print(f"{r['id']:<6}{str(r['judge_faithful']):>10}{str(r['my_faithful']):>10}"
              f"{'✓' if r['faith_agree'] else '✗':>6}{r['judge_relevance']:>10.0f}{r['my_relevance']:>10.0f}{r['rel_diff']:>+6.1f}")
    print("-" * 78)
    print(f"忠实度一致率: {faith_acc}%   Cohen's kappa: {kappa}")
    print(f"相关性完全一致: {rel_exact}%    ±1 分内一致: {rel_pm1}%    MAE: {rel_mae}")
    print("=" * 78)

    payload = {
        "n_labels": len(labels), "n_compared": len(rel),
        "judge_version": meta.get("judge_version", ""),
        "judge_model": meta.get("judge_model", ""),
        "judge_provider": meta.get("judge_provider", ""),
        "agent_model": meta.get("model", ""),
        "results_file": str(Path(args.results).resolve()),
        "faithfulness": {"agreement": faith_acc, "kappa": kappa},
        "relevance": {"exact": rel_exact, "within_1": rel_pm1, "mae": rel_mae},
        "items": rows,
    }
    Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {args.report}")


if __name__ == "__main__":
    main()
