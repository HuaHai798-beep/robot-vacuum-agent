# -*- coding: utf-8 -*-
"""忠实度后处理：把「资料里没有的数字事实」揪出来并清掉。

为什么需要它：
    v1.8 的 judge 显示忠实度只有 43.3%，看明细是模型在答对之后**追加常识性数字**
    （"正常使用 2–3 年"、"充放电循环约 500 次"），知识库里根本没有这些数据。
    提示词层面的"禁止编造"约束力有限 → 增加一层**确定性的数字比对后处理**：
        1) 从回答里抽出「带单位的数字事实」（年份/时长/距离/面积/功率/次数/百分比…）
        2) 检查该数字是否出现在检索到的参考资料里
        3) 不在资料里的，整句删掉（并在日志里记录删了什么，便于评测归因）

设计取舍：
    - **只处理数字事实**，不碰描述性句子：数字最好验证，误杀风险最低；
    - **整句删除**而不是改数字：改数字可能造成新的错误事实；
    - 单位白名单里不放"个/级/条"这类结构词，避免把"1 句结论、3 个步骤"误判成事实；
    - 中文数字（"两年"）不处理——正则无法可靠转写，属于已知局限（见 README 局限）。
"""
import re

from utils.logger_handler import logger

# 只认「事实型」单位，避免误杀列表序号/分点数量
_UNIT = r"(?:年|个月|月|周|天|小时|分钟|秒|㎡|平方米|米|cm|厘米|mm|毫米|Pa|kPa|W|瓦|L|升|ml|毫升|kg|公斤|g|克|次|%|元|倍|档)"

# 例：2–3 年 / 1-3个月 / 500 次 / 2000Pa / 50% / 0.5 米
_FACT_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*(?:[-~–—至到]\s*(\d+(?:\.\d+)?))?\s*({_UNIT})",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")


def _digits(text: str) -> set[str]:
    """抽出文本里所有数字串（用于「这个数字在资料里出现过吗」）。"""
    return {d.rstrip("0").rstrip(".") if "." in d else d for d in _DIGIT_RE.findall(text or "")}


def _norm_digits(d: str) -> str:
    d = (d or "").strip()
    return d.rstrip("0").rstrip(".") if "." in d else d


def extract_numeric_facts(text: str) -> list[dict]:
    """抽出「带单位的数字事实」。返回 [{raw, nums, unit}]"""
    out = []
    for m in _FACT_RE.finditer(text or ""):
        nums = [m.group(1)] + ([m.group(2)] if m.group(2) else [])
        out.append({"raw": m.group(0).strip(), "nums": [_norm_digits(n) for n in nums], "unit": m.group(3)})
    return out


def find_unsupported(answer: str, context: str) -> list[dict]:
    """返回回答里「资料未提及」的数字事实列表。"""
    ctx_nums = _digits(context)
    bad = []
    for fact in extract_numeric_facts(answer):
        if any(n not in ctx_nums for n in fact["nums"]):
            bad.append(fact)
    return bad


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?\n])", text or "")
    return [p for p in parts if p is not None]


def sanitize_answer(answer: str, context: str, *, tag: str = "") -> tuple[str, list[str]]:
    """删掉包含「无依据数字」的整句。返回 (清理后的回答, 被删的句子列表)。

    判定粒度是**句子内的数字事实**，不是"这个数字是否出现在坏数字集合里"：
    否则 `2-3 年` 里的 "2" 会误伤"建议 2XL"这类完全正常的句子（v1.9 单测踩到过）。

    若清理后为空（整段都在编数字），保留原文——宁可留一句可能不忠实的回答，
    也不要给用户返回空字符串（空回答本身就是一次事故）。
    """
    if not answer:
        return answer, []
    if not find_unsupported(answer, context):
        return answer, []

    kept, dropped, bad_nums = [], [], set()
    for sent in _split_sentences(answer):
        sent_bad = find_unsupported(sent, context)
        if sent_bad:
            dropped.append(sent.strip())
            bad_nums.update(n for f in sent_bad for n in f["nums"])
        else:
            kept.append(sent)

    if not "".join(kept).strip():
        logger.warning(f"[guard]{tag} 全部句子都含无依据数字，保留原文（长度 {len(answer)}）")
        return answer, []

    cleaned = "".join(kept).strip()
    logger.warning(
        f"[guard]{tag} 删除 {len(dropped)} 句无依据内容（数字 {sorted(bad_nums)}）："
        f"{dropped[0][:40] if dropped else ''}"
    )
    return cleaned, dropped
