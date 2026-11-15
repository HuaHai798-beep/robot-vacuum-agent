# -*- coding: utf-8 -*-
"""Q&A 边界切块（v2.2 / T09）

问题：知识库是「问题？/ 答：……」逐行结构，`RecursiveCharacterTextSplitter(200)` 盲切会把
问句和答句切到两个块里 —— 检索命中答句却丢了问题语境，MRR 上不去。

做法（按行解析，而不是按空行切段）：
1. 以「？/? 结尾且不超过 60 字」的行识别问句；
2. 问句之后的行（直到下一个问句/小标题）作为答句，合成 `问：… / 答：…` 一块；
3. 块超过 `qa_max_chars` 时按句拆，**每段都带问句前缀**（contextual header）；
4. 小标题（≤24 字、无标点）作为分隔，触发上一段收口；
5. 非问答内容（选购指南、故障排查条目）按 `min_chars` 合并成 text 块；
6. metadata 增加 `qa_id`（问句哈希）与 `chunk_type`（qa/text），便于块级评测定位。
"""
import hashlib
import re

from langchain_core.documents import Document

_Q_END = ("？", "?")
_PUNCT_END = re.compile(r"[。！？：:，,；;]$")


def _is_question(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 60:
        return False
    if s.startswith(("答：", "答:", "A:", "a:")):
        return False
    return s.endswith(_Q_END)


def _is_heading(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 24:
        return False
    if s.endswith(_Q_END) or s.startswith(("答：", "答:")):
        return False
    return not _PUNCT_END.search(s)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；!?;])", text)
    return [p for p in (x.strip() for x in parts) if p]


def _qa_id(question: str) -> str:
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:8]


def _split_long_answer(question: str, answer: str, max_chars: int, overlap_sents: int = 1) -> list[str]:
    """答句太长：按句切，每段前面都带问句（截断到 40 字，避免前缀吃掉配额）。"""
    head = question.strip()[:40]
    body = answer.strip()
    out, buf = [], head + "\n"
    for sent in _sentences(body):
        if len(buf) + len(sent) > max_chars and len(buf) > len(head) + 2:
            out.append(buf.strip())
            tail = _sentences(buf.strip())[-overlap_sents:] if overlap_sents else []
            buf = head + "\n" + "".join(tail)
        buf += sent
    if len(buf.strip()) > len(head) + 2:
        out.append(buf.strip())
    return out


def split_documents_qa(documents: list[Document], qa_max_chars: int = 380,
                       min_chars: int = 80, overlap_sents: int = 1,
                       text_max_chars: int = 220) -> list[Document]:
    """把 Document 列表切成 Q&A 块（保留原 metadata）。"""
    chunks: list[Document] = []

    for doc in documents:
        text = (doc.page_content or "").replace("\r\n", "\n")
        base_meta = dict(doc.metadata or {})
        cur_q: str | None = None
        ans: list[str] = []
        text_buf: list[str] = []
        pending_heading: str | None = None
        section: str | None = None

        def emit_qa():
            nonlocal cur_q, ans
            if cur_q is None:
                return
            q, a = cur_q, "\n".join(ans).strip()
            cur_q, ans = None, []
            if a:
                a = re.sub(r"^答[：:]\s*", "", a)     # 源文件已带「答：」，避免出现「答：答：」
            merged = f"问：{q}\n答：{a}" if a else f"问：{q}"
            meta = {**base_meta, "qa_id": _qa_id(q), "chunk_type": "qa", "q": q[:60]}
            if section:
                meta["section"] = section
            if len(merged) <= qa_max_chars:
                chunks.append(Document(page_content=merged, metadata=meta))
            else:
                for part in _split_long_answer(f"问：{q}", f"答：{a}", qa_max_chars, overlap_sents):
                    chunks.append(Document(page_content=part, metadata=dict(meta)))

        def emit_text():
            nonlocal text_buf, pending_heading
            if pending_heading:
                text_buf.insert(0, pending_heading)
                pending_heading = None
            if text_buf:
                body = "\n".join(text_buf).strip()
                if body:
                    chunks.append(Document(page_content=body,
                                           metadata={**base_meta, "chunk_type": "text"}))
                text_buf = []

        for raw in text.splitlines():
            s = raw.strip()
            if not s:
                continue
            if _is_question(s):
                emit_qa()
                if pending_heading:
                    section = pending_heading
                    pending_heading = None
                emit_text()
                cur_q, ans = s, []
                continue
            if cur_q is not None:
                ans.append(re.sub(r"^答[：:]\s*", "", s) if s.startswith(("答：", "答:")) else s)
                continue
            if _is_heading(s):
                emit_text()
                pending_heading = s
                continue
            if pending_heading:
                text_buf.append(pending_heading)
                pending_heading = None
            text_buf.append(s)
            if sum(len(x) for x in text_buf) >= max(min_chars, text_max_chars):
                emit_text()

        emit_qa()
        emit_text()

    return chunks
