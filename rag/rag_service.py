from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from rag.faithfulness_guard import sanitize_answer
from utils.config_handler import rag_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model

# v1.9：评测用轨迹 —— 记录「模型实际看到的参考资料」。
# 为什么需要：评测如果自己再检索一遍，judge 拿到的上下文和 agent 看到的不一致，
# 会把「有依据的回答」误判成不忠实（v1.9 首次跑批就踩到了：R011 的 2–3 年其实在资料里）。
RAG_TRACE: list[dict] = []


def take_rag_trace() -> list[dict]:
    """取走并清空当次检索轨迹（评测每个用例前调用一次）"""
    out = list(RAG_TRACE)
    RAG_TRACE.clear()
    return out

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    # 1 usage
    def _init_chain(self):
        chain = self.prompt_template |print_prompt |self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)


    # 假设这是RagSummarizeService类下的方法
    def rag_summarize(self, query: str) -> str:
        # 1. 根据query检索相关文档
        try:
            context_docs = self.retriever_docs(query)
        except Exception as e:
            logger.error(f"[rag]检索失败 query={query[:30]}：{type(e).__name__}: {e}", exc_info=True)
            return ""

        context = ""
        counter = 0
        # 2. 拼接检索到的文档为上下文字符串，同时带上元数据
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 2.1 记录模型实际看到的上下文（评测归因用）
        RAG_TRACE.append({"query": query, "context": context, "n_docs": len(context_docs),
                          "sources": [str((d.metadata or {}).get("source", "")) for d in context_docs]})

        if not context.strip():
            logger.warning(f"[rag]检索为空 query={query[:30]}，直接返回未覆盖提示")
            return "资料未覆盖该问题。"

        # 3. 把问题和拼接好的上下文传入大模型链，获取返回结果
        try:
            answer = self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            )
        except Exception as e:
            logger.error(f"[rag]模型调用失败 query={query[:30]}：{type(e).__name__}: {e}", exc_info=True)
            return ""

        # 4. 忠实度后处理（v1.9）：删掉「资料未提及」的数字事实，防止 helpful hallucination
        if bool(rag_conf.get("faithfulness_guard", True)):
            try:
                answer, dropped = sanitize_answer(answer, context, tag=f"rag:{query[:18]}")
                if dropped:
                    answer += "\n（注：以上仅保留参考资料可支撑的内容，其余未经验证的数据已省略）"
            except Exception as e:
                # 后处理是增强能力，失败不能影响主流程：记日志 + 返回原文
                logger.error(f"[rag]忠实度后处理失败，返回原文：{type(e).__name__}: {e}", exc_info=True)
        return answer
if __name__ == '__main__':
     # 初始化RAG服务
     rag = RagSummarizeService()
     # 调用方法，查询小户型适合的扫地机器人
     print(rag.rag_summarize("小户型适合哪些扫地机器人"))