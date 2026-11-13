# -*- coding: utf-8 -*-
"""本地单测：忠实度后处理 + 数字依据率（不联网、不花钱）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.faithfulness_guard import extract_numeric_facts, find_unsupported, sanitize_answer
from eval.run_eval import numeric_check

CTX = "身高175-182cm，体重145-165斤，建议尺码2XL。滤网建议每2周清理一次，吸力2000Pa足够。"
ANS = "建议2XL。正常使用2-3年，充放电循环约500次。滤网建议每2周清理一次。价格约1299元。吸力2000Pa足够。"

facts = [f["raw"] for f in extract_numeric_facts(ANS)]
bad = [f["raw"] for f in find_unsupported(ANS, CTX)]
cleaned, dropped = sanitize_answer(ANS, CTX)

print("1) 抽到的数字事实:", facts)
print("2) 无依据:", bad)
print("3) 清理后:", cleaned)
print("4) 删除句:", dropped)
print("5) 规则指标:", numeric_check(ANS, CTX))
print("6) 全部有依据时:", numeric_check("建议2XL，滤网每2周清理一次", CTX))
