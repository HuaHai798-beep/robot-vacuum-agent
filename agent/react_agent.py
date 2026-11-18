from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from model.factory import build_chat_model
from utils.config_handler import context_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                    get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from agent.tools.context_middleware import build_context_middlewares


class ReactAgent:
    def __init__(self):
        # v2.3：输出上限（报告 ≤800 字，这里给 1200 token 兜底）
        self.model = build_chat_model(max_tokens=int(context_conf.get("agent_max_tokens", 1200)))
        self.agent = create_agent(
            model=self.model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                  get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch,
                        ToolRetryMiddleware(max_retries=2,
                                            tools=["get_weather", "fetch_external_data"],
                                            on_failure="continue"),
                        *build_context_middlewares()]
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                latest_message = chunk["messages"][-1]
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"
        except Exception as e:
            # 流式过程中出错：记全栈 + 给用户一个可读的降级回复（不把异常栈丢给前端）
            logger.error(f"[agent]stream 执行失败 query={query[:30]}：{type(e).__name__}: {e}", exc_info=True)
            yield "抱歉，这次查询出了点问题，请稍后重试或换个说法。\n"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
