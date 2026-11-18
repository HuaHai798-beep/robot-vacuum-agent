from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService

from datetime import datetime
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import os
import time
from utils.logger_handler import logger
import json, urllib.parse, urllib.request

rag = RagSummarizeService()
# 当前用户：环境变量 > config/agent.yml > 兜底默认值
_mock_user = agent_conf.get("mock_user") or {}
USER_ID   = os.getenv("AGENT_USER_ID")   or _mock_user.get("id")   or "1001"
USER_CITY = os.getenv("AGENT_USER_CITY") or _mock_user.get("city") or "杭州"
CURRENT_USER_ID = USER_ID   # 当前登录用户，用于越权校验




external_data = {}

@tool(description="检索知识库（问答/故障/保养/选购）；产品类问题必用")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)



@tool(description="实时天气（温度/现象/湿度）；无依赖时可与其他工具并行调用")
def get_weather(city: str) -> str:
    """v2.3：两次重试 + 失败返回结构化降级标记（而不是空串），让模型知道是"服务不可用"。"""
    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=6) as r:   # 必有超时
                data = json.loads(r.read().decode("utf-8"))
            cur = data["current_condition"][0]
            return json.dumps({
                "city": city,
                "text": cur["weatherDesc"][0]["value"],
                "temp_c": cur["temp_C"],
                "humidity": f'{cur["humidity"]}%',
            }, ensure_ascii=False)
        except Exception as e:
            last = e
            logger.warning(f"[get_weather]第{attempt + 1}次失败：{e}")
            if attempt == 0:
                time.sleep(1)
    logger.error(f"[get_weather]两次均失败，返回降级标记：{last}")
    return json.dumps({
        "error": "service_unavailable",
        "tool": "get_weather",
        "retryable": True,
        "hint": "天气服务暂不可用，无法确认实时天气；不要编造温度/湿度，如实告知用户",
    }, ensure_ascii=False)


@tool(description="当前用户所在城市（纯字符串）")
def get_user_location() -> str:
    return USER_CITY

@tool(description="当前用户 ID（纯字符串）；无依赖时可与其他工具并行调用")
def get_user_id() -> str:
    return USER_ID

@tool(description="获取月份 YYYY-MM；offset=0 当前月，-1 上个月，1 下个月；推荐并行调用")
def get_current_month(offset: int = 0) -> str:
    """v2.3：把「上个月」的日期算术从模型手里收回到工具里（修报告月份口径）。"""
    try:
        offset = int(offset or 0)
    except Exception:
        offset = 0
    now = datetime.now()
    total = now.year * 12 + (now.month - 1) + offset
    year, month = divmod(total, 12)
    return f"{year}-{month + 1:02d}"


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



@tool(description="按用户和月份取使用记录（纯字符串 JSON）；仅限当前用户，无记录返回结构化提示")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    # 越权校验：只允许查询当前登录用户
    if str(user_id).strip() != str(CURRENT_USER_ID):
        logger.warning(f"[fetch_external_data]越权拦截：请求 {user_id}，当前用户 {CURRENT_USER_ID}")
        return "无权查询其他用户的使用记录，我只能查看当前登录用户的数据。"

    try:
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        # v2.3：不再返回空串，给模型一个可判读的"无记录"标记
        return json.dumps({
            "error": "not_found",
            "tool": "fetch_external_data",
            "user_id": str(user_id),
            "month": str(month),
            "hint": "该月没有使用记录，请如实告知用户，不要编造数据",
        }, ensure_ascii=False)


@tool(description="生成个人报告前必须先调用，用于切换到报告提示词")
def fill_context_for_report():
    return "fill_context_for_report已调用"
