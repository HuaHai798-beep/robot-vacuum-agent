from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 工具执行的监控
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