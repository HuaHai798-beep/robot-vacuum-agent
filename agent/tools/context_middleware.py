# -*- coding: utf-8 -*-
"""v2.3 上下文压缩中间件装配。

两件事：
1) 旧工具结果清理：ContextEditingMiddleware + ClearToolUsesEdit（框架自带，保留 tool_call 配对）
2) 历史摘要：SummarizationMiddleware（框架自带）；**报告模式跳过**——报告需要原始工具数据

开关：环境变量 AGENT_NO_COMPACT=1 时全部关闭（评测 A/B 用）。
"""
import os

from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    SummarizationMiddleware,
)

from utils.config_handler import context_conf
from utils.logger_handler import logger

SUMMARY_PROMPT = """你是对话压缩器。把下面的历史对话压缩成不超过 8 行的中文摘要。

必须保留：
1) 用户给出的面积、地面材质、机型、宠物等约束；
2) 所有数值与单位（寿命、次数、时长、面积、比例）；
3) 月份、用户 ID、已调用工具及其关键结论；
4) 尚未解决的问题。

不要编造，不要给建议，不要把"资料未提及"写成结论。只输出摘要正文。

对话：
{messages}
"""


class ReportAwareSummarization(SummarizationMiddleware):
    """报告模式不做历史摘要：报告要基于原始工具返回写数字。"""

    def before_model(self, state, runtime):
        try:
            if bool(runtime.context.get("report", False)):
                logger.info("[context]报告模式：跳过历史摘要")
                return None
        except Exception:
            pass
        return super().before_model(state, runtime)

    async def abefore_model(self, state, runtime):
        try:
            if bool(runtime.context.get("report", False)):
                return None
        except Exception:
            pass
        return await super().abefore_model(state, runtime)


def build_context_middlewares() -> list:
    """返回需要追加到 create_agent(middleware=[...]) 的中间件列表。"""
    if os.getenv("AGENT_NO_COMPACT") in {"1", "true", "True"}:
        logger.warning("[context]AGENT_NO_COMPACT=1：关闭上下文压缩（A/B 对照）")
        return []
    if not bool(context_conf.get("compact_enabled", True)):
        logger.warning("[context]compact_enabled=false：关闭上下文压缩")
        return []

    middlewares = []

    # 1) 清旧工具结果
    try:
        middlewares.append(ContextEditingMiddleware(edits=[
            ClearToolUsesEdit(
                trigger=int(context_conf.get("clear_tool_trigger", 1800)),
                clear_at_least=int(context_conf.get("clear_at_least", 200)),
                keep=int(context_conf.get("clear_keep", 2)),
                exclude_tools=tuple(context_conf.get("clear_exclude_tools") or ()),
            )
        ]))
        logger.info("[context]已启用旧工具结果清理（clear_tool_trigger=%s）"
                    % context_conf.get("clear_tool_trigger", 1800))
    except Exception as e:
        logger.error(f"[context]工具结果清理初始化失败：{type(e).__name__}: {e}", exc_info=True)

    # 2) 历史摘要（仅非报告模式）
    try:
        from model.factory import build_chat_model
        middlewares.append(ReportAwareSummarization(
            model=build_chat_model(max_tokens=int(context_conf.get("summary_max_tokens", 300))),
            trigger=("messages", int(context_conf.get("summarize_trigger_messages", 8))),
            keep=("messages", int(context_conf.get("summarize_keep_messages", 4))),
            summary_prompt=SUMMARY_PROMPT,
        ))
        logger.info("[context]已启用历史摘要（trigger=%s 条，keep=%s 条）"
                    % (context_conf.get("summarize_trigger_messages", 8),
                       context_conf.get("summarize_keep_messages", 4)))
    except Exception as e:
        logger.error(f"[context]历史摘要初始化失败：{type(e).__name__}: {e}", exc_info=True)

    return middlewares
