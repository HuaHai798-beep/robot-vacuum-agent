import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


def build_chat_model(max_tokens: int = 2000, provider: str | None = None, model: str | None = None):
    """按 config/rag.yml 的 chat_provider 构建对话模型（dashscope | deepseek）

    - dashscope：ChatTongyi（通义百炼）
    - deepseek ：ChatOpenAI 走 DeepSeek 的 OpenAI 兼容接口
    """
    provider = (provider or rag_conf.get("chat_provider") or "dashscope").lower()
    model_name = model or rag_conf["chat_model_name"]

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY 环境变量；设置后需要完全重启 IDE / 终端才会生效"
            )
        return ChatOpenAI(
            model=model_name,                 # 建议 deepseek-chat（reasoner 不适合 tool calling）
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0,
            max_tokens=max_tokens,
        )

    return ChatTongyi(model=model_name, max_tokens=max_tokens)


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return build_chat_model()


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # ⚠️ DeepSeek 不提供 embedding 模型，向量化固定走通义 DashScope
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
