# 扫地机器人智能客服 Agent

[![version](https://img.shields.io/github/v/tag/HuaHai798-beep/robot-vacuum-agent?label=version)](https://github.com/HuaHai798-beep/robot-vacuum-agent/tags)

基于 **LangChain 1.x + LangGraph + Chroma + 通义千问** 的 RAG + Agent 客服项目，覆盖「知识库检索 → 工具调用 → 多步推理 → 评测」的完整链路。

> 本项目为个人学习/作品集项目，知识库与外部记录均为演示数据，不代表任何品牌官方说法。

## 架构（六层）

```
配置层 config/*.yml ──┐
工具层 utils/*.py ────┤
模型层 model/factory.py ─→ chat_model(qwen-max) / embed_model(text-embedding-v4)
RAG  层 rag/*.py ─────┤   vector_store: 切块/入库/md5 去重/检索
Agent 层 agent/*.py ──┤   create_agent + 中间件（工具监控 / 动态提示词）
评测层 eval/*.py ─────┘   26 条用例 + 规则评分 + hit@3 + 分层归因
```

## 目录

```
agent/          智能体编排（react_agent.py）与工具（tools/agent_tools.py、middleware.py）
config/         rag.yml / chroma.yml / agent.yml / prompts.yml
data/           知识库文件（5 个 txt + 1 个扫描件 PDF）与外部记录 records.csv
docs/           评测报告等文档
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

## 评测结果

### v1.1（当前，26 条）

| 指标 | 数值 |
|---|---|
| 评测集 | 26 条（检索 15 / 工具链 5 / 拒答 3 / 追问 2 / 多轮 1） |
| 知识库 | 88 个文本块，来自 5 个知识文件 |
| 总体通过率 | **73.1%**（19/26） |
| 工具链类 | **100%**（5/5） |
| 检索类 | 66.7%（10/15） |
| 拒答类 | 100%（3/3） |
| 追问类 | 0%（0/2） |
| 检索 hit@3 | 87.5% |
| 工具调用准确率 | **90.5%** |
| 延迟 | 平均 10.14s / P95 20.31s |

### v1.0（基线，25 条）

| 指标 | 数值 |
|---|---|
| 总体通过率 | 72.0%（18/25） |
| 工具链类 | 75.0%（3/4） |
| 检索 hit@3 | 87.5% |
| 工具调用准确率 | 80.0% |
| 延迟 | 平均 10.37s / P95 19.72s |

> ⚠️ **口径说明**：v1.1 新增了报告链路用例 T005，数据集 25 → 26 条，**总体通过率跨版本不可直接比较**。
> 可对比的事实：① 工具链类 3/4 → 5/5、工具准确率 80.0% → 90.5%（含判据放宽）；② 报告生成链路由「不可用」变为「可用」（v1.0 因月份随机 + `datetime` 未导入而失败）。
> 基线文件：`eval/baseline_v0.json`（v1.0）、`eval/baseline.json`（v1.1）。

失败归因（当前 7 条）：模型凭常识跳过检索（R010、R015）、检索未命中并自行补全（R012）、回答遗漏关键步骤或术语（R001、R003）、信息不足不追问（A001、A002）。

> 详细的失败分析、判据迭代与复现方式见 [docs/评测报告.md](docs/评测报告.md)。

## 版本记录

| 版本 | 内容 | 通过率 |
|---|---|---|
| `v1.0-baseline` | 首个可复现基线（25 条） | 72.0% |
| `v1.1-report-chain` | mock 改配置注入、月份改真实时间、新增报告用例 T005、判据放宽 | 73.1%（26 条） |

## 关键设计

1. **信号灯工具 + 动态提示词**：空工具 `fill_context_for_report` 由中间件写入 `runtime.context["report"]`，`@dynamic_prompt` 据此在「客服」与「报告」两套提示词间切换。
2. **配置与提示词外置**：模型名、切块参数、提示词路径全在 yml/txt 中，改行为不用改代码。
3. **路径锚定**：所有路径经 `get_abs_path()` 锚定项目根，避免"跟着工作目录跑"（曾因此产生过两份向量库）。
4. **评测驱动优化**：规则评分（工具命中 + 关键事实）+ hit@3 + 分层归因，输出可回放的明细。

## Roadmap

- [x] P0 `get_current_month()` 改为真实系统时间；用户/城市改为配置注入（评测可复现）
- [ ] P0 提示词强化：涉及产品知识必须先检索、禁止编造、信息不足先追问
- [ ] P1 RAG 优化：`k`/`chunk_size` 调参 + rerank，用同一套评测对比
- [ ] P2 上下文管理：工具结果截断 + 历史摘要 + token 预算
- [ ] P3 记忆管理：checkpointer + 用户画像 + thread 隔离
- [ ] P4 MCP：自研业务 MCP Server 接入
