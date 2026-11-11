from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model

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
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        # 2. 拼接检索到的文档为上下文字符串，同时带上元数据
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 3. 把问题和拼接好的上下文传入大模型链，获取返回结果
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
if __name__ == '__main__':
     # 初始化RAG服务
     rag = RagSummarizeService()
     # 调用方法，查询小户型适合的扫地机器人
     print(rag.rag_summarize("小户型适合哪些扫地机器人"))