# -*- coding: utf-8 -*-
"""本地单测：忠实度后处理 + 数字依据率（不联网、不花钱）

v2.1 后修 P1：从"只打印"改成 `def test_*` + assert —— 失败会真的失败。
用法：python eval/tests/test_guard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_all
from rag.faithfulness_guard import extract_numeric_facts, find_unsupported, sanitize_answer
from eval.run_eval import numeric_check

CTX = "身高175-182cm，体重145-165斤，建议尺码2XL。滤网建议每2周清理一次，吸力2000Pa足够。"
ANS = "建议2XL。正常使用2-3年，充放电循环约500次。滤网建议每2周清理一次。价格约1299元。吸力2000Pa足够。"


def test_numeric_facts_extracted():
    """能从回答里抽出带单位的数字事实"""
    facts = [f["raw"] for f in extract_numeric_facts(ANS)]
    assert "2-3年" in facts, f"漏抽 2-3年：{facts}"
    assert "1299元" in facts, f"漏抽 1299元：{facts}"
    assert "2000Pa" in facts, f"漏抽 2000Pa：{facts}"
    assert len(facts) == 5, f"应抽到 5 个数字事实，实际 {facts}"


def test_unsupported_detected():
    """参考资料里没有的数字，必须被判定为无依据"""
    bad = [f["raw"] for f in find_unsupported(ANS, CTX)]
    assert "2-3年" in bad, f"2-3年 应判无依据：{bad}"
    assert "500次" in bad, f"500次 应判无依据：{bad}"
    assert "1299元" in bad, f"1299元 应判无依据：{bad}"
    # 上下文里有的数字不能被误伤
    assert "2000Pa" not in bad, f"2000Pa 有依据，被误判：{bad}"
    assert "2周" not in bad, f"2周 有依据，被误判：{bad}"


def test_sanitize_removes_unsupported():
    """清理后：无依据的句子被删掉，有依据的内容保留"""
    cleaned, dropped = sanitize_answer(ANS, CTX)
    assert "2-3年" not in cleaned, f"无依据内容没删干净：{cleaned}"
    assert "1299元" not in cleaned, f"价格没删干净：{cleaned}"
    assert "2周" in cleaned and "2000Pa" in cleaned, f"有依据内容被误删：{cleaned}"
    assert len(dropped) == 2, f"应删除 2 句，实际 {dropped}"


def test_numeric_check_partial():
    """规则指标：5 个数字事实里 2 个有依据 → 40%"""
    r = numeric_check(ANS, CTX)
    assert r["numbers_total"] == 5, r
    assert r["numbers_supported"] == 2, r
    assert r["number_support_rate"] == 40.0, r
    assert set(r["numbers_unsupported"]) == {"2-3年", "500次", "1299元"}, r


def test_numeric_check_all_supported():
    """全部有依据时，依据率应为 100%"""
    r = numeric_check("建议2XL，滤网每2周清理一次", CTX)
    assert r["numbers_total"] == 1, r
    assert r["numbers_supported"] == 1, r
    assert r["number_support_rate"] == 100.0, r
    assert r["numbers_unsupported"] == [], r


if __name__ == "__main__":
    sys.exit(run_all(globals()))
