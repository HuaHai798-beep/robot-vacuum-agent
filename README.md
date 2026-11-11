# 扫地机器人智能客服 Agent

基于 **LangChain 1.x + LangGraph + Chroma + 通义千问** 的 RAG + Agent 客服项目，覆盖「知识库检索 → 工具调用 → 多步推理 → 评测」的完整链路。

> 本项目为个人学习/作品集项目，知识库内容为演示用途，不代表任何品牌官方说法。

## 架构（六层）

```
配置层 config/*.yml ──┐
工具层 utils/*.py ────┤
模型层 model/factory.py ─→ chat_model(qwen-max) / embed_model(text-embedding-v4)
RAG  层 rag/*.py ─────┤   vector_store: 切块/入库/md5 去重/检索
Agent 层 agent/*.py ──┤   create_agent + 中间件（工具监控 / 动态提示词）
评测层 eval/*.py ─────┘   25 条用例 + 规则评分 + hit@3 + 分层归因
```

## 目录

```
agent/          智能体编排（react_agent.py）与工具（tools/agent_tools.py、middleware.py）
config/         rag.yml / chroma.yml / agent.yml / prompts.yml
data/           知识库文件（5 个 txt + 1 个扫描件 PDF）与外部记录 records.csv
eval/           评测集、跑批脚本、基线、结果
model/          模型工厂
prompts/        系统提示词、RAG 总结提示词、报告提示词
rag/            向量库与 RAG 链
utils/          路径、配置、日志、文件解析、提示词加载
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 设置 API Key（Windows 用户变量，设置后重开终端/IDE）
set DASHSCOPE_API_KEY=sk-xxxx

# 1) 建库：读取 data/ 下文件 → 切块 → 向量化入 Chroma
python rag/vector_store.py

# 2) 跑一个 Agent 问答
python agent/react_agent.py

# 3) 跑评测（自动与 baseline 对比）
set PYTHONPATH=%CD%
python eval/run_eval.py
```

## 评测基线（实测）

| 指标 | 数值 |
|---|---|
| 评测集 | 25 条（检索 15 / 工具链 4 / 拒答 3 / 追问 2 / 多轮 1） |
| 知识库 | 88 个文本块，来自 5 个知识文件 |
| 总体通过率 | 72.0% |
| 检索 hit@3 | 87.5% |
| 工具调用准确率 | 80.0% |
| 拒答通过率 | 100%（3/3） |
| 延迟 | 平均 10.37s / P95 19.72s |

失败归因（待修复）：工具漏调（模型凭常识跳过检索）、月份推理错误、回答概括失真、信息不足不追问、编造未返回的地名。

> 详细的失败分析与判据迭代过程见 [docs/评测报告.md](docs/评测报告.md)。

## 关键设计

1. **信号灯工具 + 动态提示词**：空工具 `fill_context_for_report` 由中间件写入 `runtime.context["report"]`，`@dynamic_prompt` 据此在「客服」与「报告」两套提示词间切换。
2. **配置与提示词外置**：模型名、切块参数、提示词路径全在 yml/txt 中，改行为不用改代码。
3. **路径锚定**：所有路径经 `get_abs_path()` 锚定项目根，避免"跟着工作目录跑"（曾因此产生过两份向量库）。
4. **评测驱动优化**：规则评分（工具命中 + 关键事实）+ hit@3 + 分层归因，输出可回放的明细。

## Roadmap

- [ ] P0 提示词强化：涉及产品知识必须先检索、禁止编造、信息不足先追问
- [ ] P0 `get_current_month()` 改为真实系统时间
- [ ] P1 RAG 优化：`k`/`chunk_size` 调参 + rerank，用同一套评测对比
- [ ] P2 上下文管理：工具结果截断 + 历史摘要 + token 预算
- [ ] P3 记忆管理：checkpointer + 用户画像 + thread 隔离
- [ ] P4 MCP：自研业务 MCP Server 接入
