# -*- coding: utf-8 -*-
"""v2.4 状态栏单测：该注入才注入、字段算得对、用户指定月份优先、预算不超。

用法：python eval/tests/test_status_middleware.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_all  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from agent.tools.status_middleware import build_status_text  # noqa: E402
from utils.context_budget import estimate_text_tokens  # noqa: E402


def _ai(*calls):
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": a or {}, "id": f"c{i}"} for i, (n, a) in enumerate(calls)
    ])


def _tool(content, name, cid="c0"):
    return ToolMessage(content=content, name=name, tool_call_id=cid)


def test_report_progress_next_step():
    """报告场景：拿到 user_id、还没拿月份 → next 必须是 get_current_month。"""
    msgs = [
        HumanMessage(content="帮我生成上个月的使用报告"),
        _ai(("get_user_id", {})),
        _tool("1001", "get_user_id"),
    ]
    text = build_status_text(msgs)
    assert text, "报告场景应当注入状态栏"
    assert "report-pending" in text, text
    assert "user_id: 1001" in text, text
    assert "next: get_current_month" in text, text
    assert "data: no" in text, text


def test_report_ready_after_fetch():
    """参数齐了 → next 变成「直接输出结论」，report 模式标记正确。"""
    msgs = [
        HumanMessage(content="生成报告"),
        _ai(("get_user_id", {})),
        _tool("1001", "get_user_id"),
        _ai(("get_current_month", {"offset": -1})),
        _tool("2024-10", "get_current_month"),
        _ai(("fill_context_for_report", {})),
        _tool("fill_context_for_report已调用", "fill_context_for_report"),
        _ai(("fetch_external_data", {"user_id": "1001", "month": "2024-10"})),
        _tool('{"效率": "90%"}', "fetch_external_data"),
    ]
    text = build_status_text(msgs, report_mode=True)
    assert "mode: report" in text, text
    assert "data: yes" in text, text
    assert "直接输出结论" in text, text
    assert "get_current_month×1" in text, text


def test_user_month_overrides_tool_month():
    """T002 回归：用户写明 2024-02，就不能顺着工具返回的当前月份走。"""
    msgs = [
        HumanMessage(content="1001 在 2024-02 的清洁效率怎么样？"),
        _ai(("get_user_id", {})),
        _tool("1001", "get_user_id"),
        _ai(("get_current_month", {"offset": -1})),
        _tool("2026-10", "get_current_month"),
    ]
    text = build_status_text(msgs)
    assert "2024-02（用户指定）" in text, text
    assert "fetch_external_data(user_id=1001, month=2024-02)" in text, text
    assert "mode: data" in text, text


def test_service_single_turn_not_injected():
    """普通单轮客服不需要状态栏（省 token、避免干扰）。"""
    assert build_status_text([HumanMessage(content="滤网多久换一次？")]) == ""


def test_service_multi_tool_injected():
    """客服进入多步工具循环后才注入，且带工具计数与主题锚点。"""
    msgs = [
        HumanMessage(content="我家 120 平，木地板，养了猫，选哪个型号"),
        _ai(("get_user_location", {})),
        _tool("杭州", "get_user_location"),
        _ai(("get_weather", {"city": "杭州"})),
        _tool('{"text": "晴"}', "get_weather"),
    ]
    text = build_status_text(msgs)
    assert "mode: service" in text, text
    assert "120㎡" in text and "木地板" in text and "猫" in text, text
    assert "get_weather×1" in text, text


def test_status_budget_not_exceeded():
    """预算裁剪生效：超长 topic 也要落在 max_tokens 附近。"""
    long_q = "帮我生成报告：我家 120 平，木地板，养了猫，型号 X1，2024-10，" * 40
    text = build_status_text([HumanMessage(content=long_q), HumanMessage(content=long_q)],
                             report_mode=True)
    assert text
    assert estimate_text_tokens(text) <= 140, f"状态栏超预算：{estimate_text_tokens(text)} token"


def test_repeat_tool_counter():
    """同参重复调用要被计出来（v2.4 新指标的数据源）。"""
    from eval.run_eval import count_repeat_tool_calls

    calls = [
        {"name": "get_current_month", "args": {"offset": -1}},
        {"name": "get_current_month", "args": {"offset": -1}},
        {"name": "get_user_id", "args": {}},
    ]
    assert count_repeat_tool_calls(calls) == 1


if __name__ == "__main__":
    sys.exit(run_all(globals()))
