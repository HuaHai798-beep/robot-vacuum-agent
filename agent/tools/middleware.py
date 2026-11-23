from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

def _sync_recent_turns(state) -> None:
    """v2.4：把最近轮次写进 ContextVar。

    为什么在工具层还要写一次：v2.4 把全局 list 改成了 ContextVar，
    而工具可能在另一个线程/上下文里执行，before_model 里 set 的值传不过去。
    在 wrap_tool_call 里（与工具同一执行上下文）再 set 一次，保证改写一定能拿到。
    """
    try:
        from utils.context_buffer import set_recent_turns
        msgs = None
        if isinstance(state, dict):
            msgs = state.get("messages")
        if msgs is None:
            msgs = getattr(state, "messages", None)
        turns = [str(m.content) for m in (msgs or [])
                 if getattr(m, "type", "") in {"human", "ai"} and getattr(m, "content", "")]
        set_recent_turns(turns)
    except Exception as e:
        logger.warning(f"[context_buffer]工具层同步最近轮次失败（不影响主流程）：{type(e).__name__}: {e}")


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 工具执行的监控
    _sync_recent_turns(getattr(request, "state", None))
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
            logger.info("[middleware]检测到报告信号灯工具，已置 report=True（后续切报告提示词）")

        return result
    except Exception as e:
        # 工具层异常：记全栈再抛给 Agent 决策（不在这里吞，避免模型误以为工具成功）
        logger.error(f"[tool monitor]工具{request.tool_call['name']}调用失败：{type(e).__name__}: {e}",
                     exc_info=True)
        raise e


@before_model
def log_before_model(
    state: AgentState,        # 整个Agent智能体中的状态记录
    runtime: Runtime,         # 记录了整个执行过程中的上下文信息
):
    # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    # 输出最新一条消息的类型和内容（debug级别日志）
    try:
        last = state["messages"][-1]
        logger.debug(f"[log_before_model]{type(last).__name__} | {last.content}")
    except Exception as e:
        logger.warning(f"[log_before_model]读取最后一条消息失败（不影响主流程）：{type(e).__name__}: {e}")

    # v2.2：把最近几轮 user/assistant 文本放进缓冲，供 RAG 工具做「多轮查询改写」
    try:
        from utils.context_buffer import set_recent_turns
        turns = [str(m.content) for m in state["messages"]
                 if getattr(m, "type", "") in {"human", "ai"} and getattr(m, "content", "")]
        set_recent_turns(turns)
        logger.debug(f"[context_buffer]最近轮次 {len(turns)} 条（保留最后 8 条）")
    except Exception as e:
        logger.warning(f"[context_buffer]写入最近轮次失败（不影响主流程）：{type(e).__name__}: {e}")

    return None

@dynamic_prompt                        # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):    # 动态切换提示词
    try:
        is_report = bool(request.runtime.context.get("report", False))
    except Exception as e:
        logger.warning(f"[middleware]读取 runtime.context 失败，按客服模式处理：{type(e).__name__}: {e}")
        is_report = False

    if is_report:                      # 是报告生成场景，返回报告生成提示词内容
        try:
            return load_report_prompts()
        except Exception as e:
            # 降级：报告提示词加载失败时退回客服提示词，保证还能回答
            logger.error(f"[middleware]报告提示词加载失败，降级为客服提示词：{type(e).__name__}: {e}",
                         exc_info=True)
    try:
        return load_system_prompts()
    except Exception as e:
        logger.error(f"[middleware]系统提示词加载失败，使用兜底提示词：{type(e).__name__}: {e}", exc_info=True)
        return "你是扫地机器人客服，只依据工具返回内容回答，资料未提及的内容不要编造。"