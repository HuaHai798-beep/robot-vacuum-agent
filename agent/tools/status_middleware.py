# -*- coding: utf-8 -*-
"""v2.4 Agent 状态栏（Agent Status Bar）。

为什么要有它
  「报告 / 使用数据」链路是固定多步：get_user_id → get_current_month → fill_context_for_report
  → fetch_external_data。但「已经拿到哪几个参数、下一步该调谁」只能靠模型从消息历史里
  现算。模型的强项是检索、不是统计归纳（第二章：上下文窗口是一台只有一半的检索引擎），
  所以步数一多就会跳步、重复调、或者把月份算错。

做法
  每次调模型前，用**代码**把当前状态算好，作为一条 user 消息追加在**请求末尾**：
    - 不写回 state（历史永远只追加，缓存零破坏）；
    - 每轮只有一条最新状态（不会像"持久追加"那样堆陈旧状态）；
    - 放在末尾 → 紧邻即将生成的 token，注意力权重最高。

v2.4 实跑修的两个坑
  1) 状态栏只显示工具算的月份，模型会顺着用 → 改为「用户指定月份」优先（T002 回归）；
  2) 「查某月清洁效率」不需要 fill_context_for_report → 区分 data / report 两种意图。

开关
  config/context.yml: status_enabled / status_max_tokens / status_include_topic
  环境变量 AGENT_NO_STATUS=1 可强制关闭（评测 A/B 用）。
"""
import os
import re

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import HumanMessage

from utils.config_handler import context_conf
from utils.context_budget import estimate_text_tokens
from utils.logger_handler import logger

STATUS_NAME = "agent_status"

# 需要切到「报告提示词」的意图（与 prompts/main_prompt.txt 的报告口径一致）
_REPORT_INTENT = ("报告", "使用情况", "使用记录", "使用数据")
# 需要查「用户使用数据」的意图（不一定切报告提示词）
_DATA_INTENT = ("清洁效率", "耗材状态", "耗材", "记录")
_MATERIALS = ("木地板", "瓷砖", "大理石", "复合地板", "防滑砖", "短毛地毯", "长毛地毯", "混合地面")
_PETS = ("猫", "狗", "宠物")
_USER_ID_RE = re.compile(r"^\d{3,}$")
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_USER_MONTH_RE = re.compile(r"(20\d{2}-\d{2})")


# ----------------------------- 开关 -----------------------------
def status_enabled() -> bool:
    if os.getenv("AGENT_NO_STATUS") in {"1", "true", "True"}:
        return False
    return bool(context_conf.get("status_enabled", True))


# ----------------------------- 解析 state -----------------------------
def _scan_messages(messages):
    """从消息历史里抽取：工具调用次数、最后一次参数、最后一次工具返回。"""
    calls: dict[str, int] = {}
    args: dict[str, dict] = {}
    outputs: dict[str, str] = {}
    tool_rounds = 0

    for m in messages or []:
        tool_calls = getattr(m, "tool_calls", None) or []
        if tool_calls:
            tool_rounds += 1
        for tc in tool_calls:
            name = str((tc or {}).get("name") or "").strip()
            if not name:
                continue
            calls[name] = calls.get(name, 0) + 1
            a = (tc or {}).get("args") or {}
            if isinstance(a, dict) and a:
                args[name] = a
        if getattr(m, "type", "") == "tool":
            name = str(getattr(m, "name", "") or "").strip()
            if name:
                outputs[name] = str(getattr(m, "content", "") or "")
    return calls, args, outputs, tool_rounds


def _user_id(outputs: dict, args: dict) -> str:
    raw = (outputs.get("get_user_id") or "").strip().strip('"')
    if _USER_ID_RE.match(raw):
        return raw
    return str((args.get("fetch_external_data") or {}).get("user_id") or "").strip()


def _month(outputs: dict, args: dict) -> str:
    raw = (outputs.get("get_current_month") or "").strip().strip('"')
    if _MONTH_RE.match(raw):
        return raw
    return str((args.get("fetch_external_data") or {}).get("month") or "").strip()


def _last_user_text(messages) -> str:
    for m in reversed(list(messages or [])):
        if getattr(m, "type", "") == "human" and getattr(m, "name", "") != STATUS_NAME:
            return str(getattr(m, "content", "") or "")
    return ""


def _intent(messages) -> tuple[bool, bool]:
    """返回 (need_report, data_like)。

    need_report：要切报告提示词的场景（报告/使用情况/使用记录/使用数据）；
    data_like  ：要查「用户使用数据」的场景（含用户指定月份、清洁效率、耗材状态）。
    两者分开，因为「查某月清洁效率」不需要 fill_context_for_report，而「生成报告」需要。
    """
    text = _last_user_text(messages)
    need_report = any(k in text for k in _REPORT_INTENT)
    data_like = need_report or bool(_USER_MONTH_RE.search(text)) or any(
        k in text for k in _DATA_INTENT)
    return need_report, data_like


def _topic(messages, limit: int = 4) -> list[str]:
    """主题锚点：面积/材质/宠物/机型/月份。多轮对比追问（M004）丢主题时靠它兜底。"""
    texts = [str(getattr(m, "content", "") or "")
             for m in (messages or [])
             if getattr(m, "type", "") == "human" and getattr(m, "name", "") != STATUS_NAME][-4:]
    facts: list[str] = []
    for t in texts:
        for m in re.finditer(r"(\d{2,3})\s*(?:㎡|平米|平方米|平)", t):
            facts.append(f"{m.group(1)}㎡")
        for w in _MATERIALS + _PETS:
            if w in t:
                facts.append(w)
        for m in re.finditer(r"\d{4}-\d{2}", t):
            facts.append(m.group(0))
        for m in re.finditer(r"(?<![A-Za-z0-9])([A-Za-z]{1,3}\d{2,4}[A-Za-z]?)(?![0-9])", t):
            facts.append(m.group(1).upper())

    seen, out = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[:limit]


def _user_month(messages) -> str:
    """用户在提问里**明确写出**的月份（如 2024-02），比 get_current_month 更权威。"""
    found = _USER_MONTH_RE.findall(_last_user_text(messages))
    return found[-1] if found else ""


def _next_step(need_report: bool, report_mode: bool, user_id: str, month: str,
               data_ready: bool, user_month: str = "") -> str:
    need_month_tool = not month and not user_month
    if not user_id and need_month_tool:
        # 两个都缺时明确写「并行」——模型单靠自己经常串行调，白多一轮往返
        return "并行调用 get_user_id() + get_current_month(offset=-1)"
    if not user_id:
        return "get_user_id()"
    if not month and need_month_tool:
        return "get_current_month(offset=-1)   # 上个月"
    if need_report and not report_mode:
        return (f"fill_context_for_report()   # 用户指定月份 {user_month}，不要调 get_current_month"
                if user_month else "fill_context_for_report()")
    if not data_ready:
        return f"fetch_external_data(user_id={user_id}, month={month})   # 不需要 rag_summarize"
    return "直接输出结论（数据已齐，不要再调工具）"


# ----------------------------- 状态文本 -----------------------------
def build_status_text(messages, report_mode: bool = False, max_rounds: int | None = None) -> str:
    """生成 <agent_status> 文本；不需要注入时返回空串（纯函数，便于单测与评测估算）。"""
    if not status_enabled():
        return ""

    calls, args, outputs, tool_rounds = _scan_messages(messages)
    need_report, data_like = _intent(messages)
    rich = bool(report_mode or data_like)

    # 普通客服单轮（没有多步工具循环）不注入：省 token，也避免无谓干扰
    if not rich and sum(calls.values()) < 2:
        return ""

    max_rounds = int(max_rounds or context_conf.get("status_max_rounds", 6) or 6)
    round_no = tool_rounds + 1
    uid = _user_id(outputs, args)
    month = _month(outputs, args)
    user_month = _user_month(messages)
    user_month_used = bool(user_month) and month != user_month
    if user_month_used:
        month = user_month            # 用户说了月份就以用户为准
    data_ready = "fetch_external_data" in outputs

    if report_mode:
        mode = "report"
    elif need_report:
        mode = "report-pending"
    elif data_like:
        mode = "data"
    else:
        mode = "service"

    lines = ['<agent_status source="framework">']
    lines.append(f"mode: {mode} | round: {round_no}/{max_rounds}")

    if rich:
        month_txt = f"{month}（用户指定）" if user_month_used else (month or "-")
        lines.append(f"user_id: {uid or '-'} | month: {month_txt} | data: "
                     f"{'yes' if data_ready else 'no'}")
        lines.append("next: " + _next_step(need_report, report_mode, uid, month,
                                           data_ready, user_month))

    if calls:
        top = sorted(calls.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
        lines.append("calls: " + " ".join(f"{k}×{v}" for k, v in top))

    if bool(context_conf.get("status_include_topic", True)):
        topic = _topic(messages)
        if topic:
            lines.append("topic: " + " · ".join(topic))

    lines.append("</agent_status>")
    text = "\n".join(lines)

    # 预算裁剪：先砍 topic，再砍 calls，最后硬截断
    budget = int(context_conf.get("status_max_tokens", 120) or 120)
    if estimate_text_tokens(text) > budget:
        text = "\n".join(ln for ln in lines if not ln.startswith("topic: "))
    if estimate_text_tokens(text) > budget:
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("calls: "))
    if estimate_text_tokens(text) > budget:
        text = text[: max(40, budget * 2)] + "\n</agent_status>"

    return text


# ----------------------------- 中间件 -----------------------------
@wrap_model_call
def inject_agent_status(request, handler):
    """在请求末尾追加一条 <agent_status> 的 user 消息（不改 state）。"""
    try:
        if not status_enabled():
            return handler(request)

        report_mode = False
        try:
            report_mode = bool(request.runtime.context.get("report", False))
        except Exception:
            report_mode = False

        text = build_status_text(request.messages, report_mode=report_mode)
        if not text:
            return handler(request)

        messages = list(request.messages) + [HumanMessage(content=text, name=STATUS_NAME)]
        logger.debug(f"[status]注入 {estimate_text_tokens(text)} token（report={report_mode}）")
        return handler(request.override(messages=messages))
    except Exception as e:  # 状态栏是增强项：失败必须降级，不能拖垮主流程
        logger.error(f"[status]注入失败，降级为无状态栏：{type(e).__name__}: {e}", exc_info=True)
        return handler(request)


def build_status_middlewares() -> list:
    if not status_enabled():
        logger.warning("[status]状态栏已关闭（status_enabled=false 或 AGENT_NO_STATUS=1）")
        return []
    logger.info("[status]已启用 Agent 状态栏（末尾注入，不改 state）")
    return [inject_agent_status]


if __name__ == "__main__":  # 手工看一眼输出
    from langchain_core.messages import AIMessage, ToolMessage

    demo = [
        HumanMessage(content="帮我生成上个月的使用报告"),
        AIMessage(content="", tool_calls=[{"name": "get_user_id", "args": {}, "id": "c1"}]),
        ToolMessage(content="1001", name="get_user_id", tool_call_id="c1"),
    ]
    print(build_status_text(demo))

    demo2 = [
        HumanMessage(content="1001 在 2024-02 的清洁效率怎么样？"),
        AIMessage(content="", tool_calls=[{"name": "get_user_id", "args": {}, "id": "c1"}]),
        ToolMessage(content="1001", name="get_user_id", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "get_current_month", "args": {"offset": -1}, "id": "c2"}]),
        ToolMessage(content="2026-10", name="get_current_month", tool_call_id="c2"),
    ]
    print(build_status_text(demo2))
