from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService

from datetime import datetime
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import os
from utils.logger_handler import logger
import json, urllib.parse, urllib.request

rag = RagSummarizeService()
# 当前用户：环境变量 > config/agent.yml > 兜底默认值
_mock_user = agent_conf.get("mock_user") or {}
USER_ID   = os.getenv("AGENT_USER_ID")   or _mock_user.get("id")   or "1001"
USER_CITY = os.getenv("AGENT_USER_CITY") or _mock_user.get("city") or "杭州"




external_data = {}

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)



@tool(description="获取指定城市的实时天气，返回温度、天气现象和湿度")
def get_weather(city: str) -> str:
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:   # 必有超时
            data = json.loads(r.read().decode("utf-8"))
        cur = data["current_condition"][0]
        return json.dumps({
            "city": city,
            "text": cur["weatherDesc"][0]["value"],
            "temp_c": cur["temp_C"],
            "humidity": f'{cur["humidity"]}%',
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[get_weather]失败：{e}")
        return ""            # 关键：失败返回空串，不要把异常抛给模型


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return USER_CITY

@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return USER_ID

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].strip('"')
                feature: str = arr[1].strip('"')
                efficiency: str = arr[2].strip('"')
                consumables: str = arr[3].strip('"')
                comparison: str = arr[4].strip('"')
                time: str = arr[5].strip('"')

                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }



@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"