# 扫地机器人智能客服 Agent

[![version](https://img.shields.io/github/v/tag/HuaHai798-beep/robot-vacuum-agent?label=version)](https://github.com/HuaHai798-beep/robot-vacuum-agent/tags)

基于 **LangChain 1.x + LangGraph + Chroma + 通义千问** 的 RAG + Agent 客服项目，覆盖「知识库检索 → 工具调用 → 多步推理 → 评测」的完整链路。

> 本项目为个人学习/作品集项目，知识库与外部记录均为演示数据，不代表任何品牌官方说法。

## 架构（六层）

```
配置层 config/*.yml ──┐
工具层 utils/*.py ────┤
模型层 model/factory.py ─→ chat_model(qwen-plus / qwen-max) / embed_model(text-embedding-v4)
RAG  层 rag/*.py ─────┤   vector_store: 切块/入库/md5 去重/检索
Agent 层 agent/*.py ──┤   create_agent + 中间件（工具监控 / 动态提示词）
评测层 eval/*.py ─────┘   26 条用例 + 规则评分 + hit@3 + 分层归因 + 预检/重试
```

## 目录

```
agent/          智能体编排（react_agent.py）与工具（tools/agent_tools.py、middleware.py）
config/         rag.yml / chroma.yml / agent.yml / prompts.yml
data/           知识库文件（5 个 txt + 1 个扫描件 PDF）与外部记录 records.csv
docs/           评测报告、Release 说明、待开 Issue
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

# 3) 跑评测（开跑前自动预检额度，失败即中止）
set PYTHONPATH=%CD%
python eval/run_eval.py
```

## 评测结果

> ⚠️ **三个数字不可横比**：模型不同（qwen-max / qwen-plus）、判据版本不同（是否支持破折号与同义词）、数据集版本不同（25 / 26 条）。仓库同时保留每轮的 `eval/results/*.jsonl` 与各自的 baseline 文件。

| 版本 | 模型 | 用例 | 通过率 | hit@3 | 工具准确率 | 说明 |
|---|---|---|---|---|---|---|
| v1.0 基线 | qwen-max | 25 | 72.0% | 87.5% | 80.0% | 首个可复现基线 |
| v1.1 报告链路 | qwen-max | 26 | 73.1% | 87.5% | 90.5% | 新增 T005、mock 配置化 |
| **v1.2 P0 提示词** | qwen-max | 26 | **88.5%** | 81.2% | 100% | 强制检索/禁编造/先追问（结果见 `results/run_20261113_0047.jsonl`）|
| **v1.2 评测加固** | **qwen-plus** | 26 | **96.2%** | 93.8% | 100% | 判据修正 + 评测脚本加固（`eval/baseline_qwen_plus.json`）|

当前默认模型为 `qwen-plus`（`config/rag.yml`）。切换回 `qwen-max` 只需改一行，但**基线必须按模型分开记录**。

### 评测工程加固（v1.2）

1. **预检 preflight**：开跑前花 1 个 token 探测额度，403/欠费立即中止（曾因免费额度耗尽跑废一整轮）；
2. **重试退避**：403/429/超时自动重试 3 次（5s/15s/30s），并记录 `retries`；
3. **基线元数据**：`baseline` 内记录 `model` / `dataset_sha256` / `git_commit` / `errors`，模型或数据集不同则拒绝横比；
4. **防空基线**：存在系统级错误时拒绝写入基线。

> 详细的失败分析、判据迭代与复现方式见 [docs/评测报告.md](docs/评测报告.md)。

## 版本记录

| 版本 | 内容 | 通过率 |
|---|---|---|
| `v1.0-baseline` | 首个可复现基线（25 条） | 72.0% |
| `v1.1-report-chain` | mock 改配置注入、月份改真实时间、新增 T005、判据放宽 | 73.1%（26 条） |
| `v1.2-p0-eval-hardening` | P0 提示词强化 + 评测脚本加固 + 切换 qwen-plus | 96.2%（qwen-plus）/ 88.5%（qwen-max） |

## 关键设计

1. **信号灯工具 + 动态提示词**：空工具 `fill_context_for_report` 由中间件写入 `runtime.context["report"]`，`@dynamic_prompt` 据此在「客服」与「报告」两套提示词间切换。
2. **配置与提示词外置**：模型名、切块参数、提示词路径全在 yml/txt 中，改行为不用改代码。
3. **路径锚定**：所有路径经 `get_abs_path()` 锚定项目根，避免"跟着工作目录跑"（曾因此产生过两份向量库）。
4. **评测驱动优化**：规则评分（工具命中 + 关键事实）+ hit@3 + 分层归因，输出可回放的明细。
5. **评测工程化**：预检额度、失败重试、基线元数据（模型/数据集哈希/commit），让"分数"可复现、可比、可追责。

## Roadmap

- [x] P0 `get_current_month()` 改真实时间；用户/城市改配置注入（评测可复现）
- [x] P0 提示词强化：涉及产品知识必须先检索、禁止编造、信息不足先追问
- [x] 评测脚本加固：预检 / 重试退避 / 基线元数据
- [ ] P0 超范围请求要**明确拒答**（当前 F001 只回"未找到相关资料"，属于弱拒答）
- [ ] P1 RAG 优化：`k`/`chunk_size` 调参 + rerank，用同一套评测对比
- [ ] P2 上下文管理：工具结果截断 + 历史摘要 + token 预算
- [ ] P3 记忆管理：checkpointer + 用户画像 + thread 隔离
- [ ] P4 MCP：自研业务 MCP Server 接入
