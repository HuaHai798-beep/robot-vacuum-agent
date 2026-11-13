from langchain.agents import create_agent
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                    get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                  get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch]
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