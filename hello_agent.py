# -*- coding: utf-8 -*-
"""最小可运行 Agent 示例（LangChain 1.x 新写法：create_agent）

运行前提：
    1) 已设置环境变量 DASHSCOPE_API_KEY
    2) 已安装 requirements.txt 里的依赖（本项目的 .venv 已经装好）

和旧写法的区别（一句话）：
    旧：create_react_agent + AgentExecutor + 手写 ReAct 提示词
    新：create_agent 一个函数搞定，循环由图引擎驱动，不需要执行器
"""

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool


@tool(description="查询指定城市的天气")
def get_weather(city: str) -> str:
    """演示用的假工具：真实项目里换成天气 API"""
    return "晴天"


# 1) 建智能体：模型 + 工具 + 系统提示词
agent = create_agent(
    model=ChatTongyi(model="qwen-max"),
    tools=[get_weather],
    system_prompt="你是一个聊天助手，可以回答用户问题！",
)

# 2) 调用：输入必须是 {"messages": [...]} 这样的消息列表
res = agent.invoke(
    {"messages": [{"role": "user", "content": "今天深圳天气如何，怎么穿衣？"}]}
)

# 3) 取最终回答：返回值是 dict，答案是 messages 里的最后一条
#    一轮完整对话通常是 4 条：Human → AI(工具调用) → Tool(结果) → AI(最终回答)
for i, msg in enumerate(res["messages"], 1):
    print(f"[{i}] {type(msg).__name__}: {str(msg.content)[:50]}")

print("-" * 50)
print("最终回答：", res["messages"][-1].content)
