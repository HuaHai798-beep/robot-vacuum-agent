import argparse
import hashlib
import json
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf, rag_conf
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.qa_splitter import split_documents_qa


class VectorStoreService:
    """知识库向量库服务

    v2.1 后修（P0）：
      1) md5 去重记录里带上「切块参数指纹」：文件没变但参数变了不再静默跳过，而是 WARN 提示 --rebuild；
      2) 新增 `--rebuild`：清空 collection + 清空 md5 记录后全量重建（改 chunk_size / separators 后必须用）；
      3) 新增 `--stats`：不加载、只打印块数与记录数，便于核对。
    """

    def __init__(self):
        self._chroma_kwargs = dict(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        self.vector_store = Chroma(**self._chroma_kwargs)

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # ----------------------------- 基础能力 -----------------------------
    def get_retriever(self):
        k = chroma_conf["k"]
        if chroma_conf.get("rerank_enabled"):
            from rag.reranker import make_rerank_retriever
            return make_rerank_retriever(self.vector_store, topk=k,
                                         candidates=int(chroma_conf.get("rerank_candidates", 10)))
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        """当前 collection 的块数（不触发 embedding）"""
        try:
            return len(self.vector_store.get(include=[])["ids"])
        except Exception as e:
            logger.error(f"[知识库]统计块数失败：{type(e).__name__}: {e}")
            return 0

    # ----------------------------- 切块参数指纹 -----------------------------
    def param_fingerprint(self) -> str:
        """把"决定分块结果"的参数做成短指纹：参数一变，指纹就变。

        只对文件内容做 md5 是不够的：文件没改、但 chunk_size 改了，md5 相同，
        旧实现会当成"已处理"直接跳过，导致"重建库"其实没重建。
        """
        payload = {
            "chunk_size": chroma_conf["chunk_size"],
            "chunk_overlap": chroma_conf["chunk_overlap"],
            "separators": chroma_conf["separators"],
            "embedding_model": rag_conf.get("embedding_model_name", ""),
            # v2.2：切块策略也参与指纹 —— 改 qa_split 必须触发重建，否则库还是老分块
            "qa_split": bool(chroma_conf.get("qa_split", True)),
            "qa_max_chars": int(chroma_conf.get("qa_max_chars", 380)),
            "qa_min_chars": int(chroma_conf.get("qa_min_chars", 80)),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.md5(raw).hexdigest()[:8]

    # ----------------------------- md5 记录读写 -----------------------------
    def _md5_store_path(self) -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    def read_md5_records(self) -> dict:
        """返回 {文件md5: 参数指纹}；旧格式（只有 md5）的指纹记作 '' """
        path = self._md5_store_path()
        if not os.path.exists(path):
            return {}
        records = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    md5_hex, fp = line.split(":", 1)
                    records[md5_hex] = fp
                else:
                    records[line] = ""      # 旧格式：没有参数指纹
        return records

    def _reset_md5_store(self):
        with open(self._md5_store_path(), "w", encoding="utf-8") as f:
            f.write("")

    def _append_md5_record(self, md5_hex: str, fingerprint: str):
        with open(self._md5_store_path(), "a", encoding="utf-8") as f:
            f.write(f"{md5_hex}:{fingerprint}\n")

    # ----------------------------- 重建 -----------------------------
    def reset(self):
        """清空 collection + 清空 md5 记录（--rebuild 时调用）"""
        try:
            self.vector_store.delete_collection()
        except Exception as e:
            # collection 可能本来就不存在
            logger.warning(f"[重建知识库]删除旧 collection 时出现异常（可忽略）：{type(e).__name__}: {e}")
        self.vector_store = Chroma(**self._chroma_kwargs)
        self._reset_md5_store()
        logger.warning("[重建知识库]已清空向量库与 md5 记录，开始全量重建")

    # ----------------------------- 切块 -----------------------------
    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """v2.2：优先按「问句+答句」成块；非问答文档自动退回普通切分。"""
        if bool(chroma_conf.get("qa_split", True)):
            return split_documents_qa(
                documents,
                qa_max_chars=int(chroma_conf.get("qa_max_chars", 380)),
                min_chars=int(chroma_conf.get("qa_min_chars", 80)),
            )
        return self.spliter.split_documents(documents)

    # ----------------------------- 加载 -----------------------------
    def load_document(self, rebuild: bool = False):
        """从数据文件夹读取文件 → 切块 → 入向量库

        :param rebuild: True 时先清空 collection 与 md5 记录（改过切块参数后必须用）
        """
        if rebuild:
            self.reset()

        fingerprint = self.param_fingerprint()
        records = self.read_md5_records()

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        loaded, skipped, warned = 0, 0, 0
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.error(f"[加载知识库]{path}无法计算 md5，跳过")
                continue

            if md5_hex in records:
                recorded_fp = records[md5_hex]
                if recorded_fp == fingerprint:
                    logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                    skipped += 1
                    continue
                # 文件没变、但切块参数变了：必须重建，否则库还是老分块
                logger.warning(
                    f"[加载知识库]{path}文件未变，但切块参数已变"
                    f"（记录指纹 {recorded_fp or '旧格式无指纹'} ≠ 当前 {fingerprint}），"
                    f"已跳过；请执行 `python rag/vector_store.py --rebuild` 重建"
                )
                warned += 1
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self._split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 幂等重入：先按 source 删掉该文件的旧块，再写入。
                # 否则 md5 记录一旦丢失/损坏，同一文件会被重复追加（实测块数 88 → 147）。
                try:
                    self.vector_store.delete(where={"source": path})
                except Exception as e:
                    logger.warning(f"[加载知识库]{path}删除旧块失败（首次入库可忽略）：{type(e).__name__}: {e}")

                self.vector_store.add_documents(split_document)
                self._append_md5_record(md5_hex, fingerprint)
                records[md5_hex] = fingerprint
                loaded += 1
                logger.info(f"[加载知识库]{path} 内容加载成功（{len(split_document)} 块）")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

        logger.info(
            f"[加载知识库]完成：新加载 {loaded} 个文件 | 跳过 {skipped} 个 | "
            f"参数变更告警 {warned} 个 | 当前共 {self.count()} 块 | 指纹 {fingerprint}"
        )
        return {"loaded": loaded, "skipped": skipped, "warned": warned,
                "chunks": self.count(), "fingerprint": fingerprint}


def main():
    ap = argparse.ArgumentParser(description="知识库构建（md5 去重 + 参数指纹 + 重建）")
    ap.add_argument("--rebuild", action="store_true",
                    help="清空向量库与 md5 记录后全量重建（改过 chunk_size/separators 后必须用）")
    ap.add_argument("--stats", action="store_true", help="只打印块数与记录数，不做加载")
    ap.add_argument("--query", default="", help="加载完成后检索该查询（调试用）")
    args = ap.parse_args()

    vs = VectorStoreService()

    if args.stats:
        rec = vs.read_md5_records()
        print(f"块数: {vs.count()} | md5 记录: {len(rec)} 条 | 当前参数指纹: {vs.param_fingerprint()}")
        legacy = [k for k, v in rec.items() if not v]
        if legacy:
            print(f"⚠️ 其中 {len(legacy)} 条为旧格式（无参数指纹），建议执行 --rebuild")
        return

    result = vs.load_document(rebuild=args.rebuild)
    print(f"完成：新加载 {result['loaded']} 个 | 跳过 {result['skipped']} 个 | "
          f"参数变更告警 {result['warned']} 个 | 共 {result['chunks']} 块 | 指纹 {result['fingerprint']}")

    if args.query:
        retriever = vs.get_retriever()
        for i, r in enumerate(retriever.invoke(args.query), 1):
            source = os.path.basename(str((r.metadata or {}).get("source", "")))
            print(f"\n[{i}] {source}\n{r.page_content[:200]}")


if __name__ == "__main__":
    main()
