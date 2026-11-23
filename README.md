# 扫地机器人智能客服 Agent

[![version](https://img.shields.io/github/v/tag/HuaHai798-beep/robot-vacuum-agent?label=version&v=2)](https://github.com/HuaHai798-beep/robot-vacuum-agent/tags)

基于 **LangChain 1.x + LangGraph + Chroma + DeepSeek/通义千问** 的 RAG + Agent 客服项目，覆盖「知识库检索 → 工具调用 → 多步推理 → 上下文工程 → 评测」的完整链路。

> 个人学习 / 作品集项目：知识库与外部记录均为演示数据，不代表任何品牌官方说法。

**当前版本 v2.4**：48 条评测集 **100% 通过**、忠实度 **93.8%**、P95 **5.71s**、单次成本 **0.0078 元**；
核心机制是「状态栏（代码维护任务状态）+ 静态前缀（缓存友好）+ 评测闸门（不许退化）」。

---

## 这个项目解决什么问题

客服 Agent 的难点不是"会不会调用模型"，而是三件事：

1. **答得准**：必须基于知识库与用户数据答，不能编（RAG + 数字依据校验）；
2. **走得完**：报告类请求要按 `取用户 → 取月份 → 切报告模式 → 取使用记录 → 写报告` 多步走完，不能跳步/重复调；
3. **说得清**：每次改动都要能用数字证明变好了（48 条用例 / 九类指标 / 基线闸门）。

三条分别对应下面的 RAG 层、Agent 层、评测层。

---

## 架构（六层）

```
配置层 config/*.yml ──┐  rag / chroma / agent / prompts / context
工具层 utils/*.py ────┤  路径锚定 / 配置加载 / 日志 / 文件解析 / token 预算 / 轮次缓冲
模型层 model/factory.py ─→ chat_model(deepseek-chat | qwen-plus 可切换) / embed_model(text-embedding-v4)
RAG  层 rag/*.py ─────┤  切块入库 / md5 去重 / 官方重排 / 多轮查询改写 / 数字依据校验
Agent 层 agent/*.py ──┤  create_agent + 中间件（工具监控 / 动态提示词 / 状态栏 / 上下文压缩）
评测层 eval/*.py ─────┘  48 条用例 · 九类指标 · 分层归因 · 发布闸门 · 结果可回放
```

## 目录

```
agent/    react_agent.py（编排）；tools/ 下 agent_tools.py（7 个工具）、middleware.py（监控/信号灯）、
          status_middleware.py（状态栏）、context_middleware.py（压缩）
config/   rag.yml / chroma.yml / agent.yml / prompts.yml / context.yml
data/     5 个知识 txt + 1 个 PDF，外部记录 data/external/records.csv
model/    模型工厂（provider 可切换）
prompts/  系统提示词 / RAG 总结 / 报告提示词
rag/      vector_store.py / rag_service.py / reranker.py / query_rewriter.py / faithfulness_guard.py
utils/    路径、配置、日志、文件、提示词、context_budget、context_buffer
eval/     dataset.jsonl（48 条）/ run_eval.py / baselines/ / results/ / tests/ / exp_*.py
docs/     RELEASE_v*.md、评测报告.md、v2.x_交付说明.pdf、待开Issue.md
```

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%
set DASHSCOPE_API_KEY=sk-xxxx      # 嵌入模型
set DEEPSEEK_API_KEY=sk-xxxx       # 对话模型（默认 provider）

python rag/vector_store.py         # 1) 建库（md5 去重，重复跑只会跳过）
python agent/react_agent.py        # 2) 跑一次问答
python eval/run_eval.py            # 3) 跑评测（自动预检额度）
python eval/run_eval.py --judge --judge-model deepseek-reasoner \
  --gate eval/baselines/baseline_v22.json     # 4) 带闸门的发布校验

# 关掉状态栏做 A/B：set AGENT_NO_STATUS=1
# 关掉上下文压缩做 A/B：python eval/run_eval.py --no-compact
```

---

## 评测体系

### 三条不可横比纪律

① 跨模型（deepseek-chat ↔ qwen-plus）；② 跨判据版本（j1→j6）；③ 跨数据集（25/26/30/48 条）。
基线文件记录 `model / judge_version / dataset_sha256 / git_commit`，脚本发现不一致**拒绝横比**。

### 当前基线（48 条，j6，deepseek-chat）

| 指标 | v2.2 | v2.3 | **v2.4（当前）** |
|---|---|---|---|
| 通过率 | 100% | 100% | **100%** |
| 忠实度（judge） | 83.3% | 89.6% | **93.8%** |
| 数字依据率（规则） | 88.4% | 91.6% | 90.2% |
| 检索 hit@3 / 块级 recall@3 | 92.6% / 92.6% | 持平 | 92.6% / 92.6% |
| 工具调用 / 参数准确率 | 100% / 100% | 100% / 100% | **100% / 100%** |
| tokens_in / out | 2647 / 322 | 2622 / 258 | **2674 / 256** |
| P50 / P95 延迟 | 3.86 / 11.52s | 3.79 / 7.96s | **3.56 / 5.71s** |
| 单次成本（含重排） | 0.00828 元 | 0.00772 元 | **0.00781 元** |
| 静态提示占比 | — | 91.3% | **92.2%** |
| 重复工具调用 / 报告链轮次 | — | — | **0 次 / 4.0 轮** |
| Prompt Cache 命中率 | — | — | **80.5%** |

### 指标口径

| 层 | 指标 | 口径 |
|---|---|---|
| 生成 | 通过率 | 用例内全部断言（工具/参数/事实/拒答/安全）同时满足 |
| 生成 | 忠实度 | LLM judge（deepseek-reasoner，与 agent 不同模型）看**模型实际看到的资料**打分 |
| 生成 | 数字依据率 | 规则：回答里带单位的数字事实必须能在工具返回中找到（确定性、零成本） |
| 检索 | hit@3 / src@3 / chunk_recall@3 / chunk_MRR@5 | 关键词覆盖 / 来源文件命中 / 块级召回与排序 |
| 工具 | 调用 / 参数准确率 | 该调的调了、不该调的不调；参数逐字段校验 |
| 工程 | P50/P95、tokens、成本、缓存命中 | 价格表 `eval/prices.yml`；缓存命中取 DeepSeek `prompt_cache_hit_tokens` |
| 上下文 | 静态占比、状态栏规模、重复调用、报告轮次 | `utils/context_budget.py` + 状态栏中间件 |

### 子集与故障注入

- `--tags` / `--ids` 跑子集：`multi_turn`、`refuse`、`failure` 等 9 个 subset；
- `inject_failure` 模拟工具不可用（测"不编造"），`inject_pad` 注入超长返回（测上下文压缩）；
- 补充集 `eval/dataset_v23_supp.jsonl` 单独验收，**不进主基线**（避免改 sha）。

---

## 最近三个版本

### v2.4 状态栏 + 并发隔离（2026-11-23，`v2.4-status-bar`，不重建基线）

| 改动 | 结果 |
|---|---|
| `<agent_status>` 请求末尾注入（代码算状态，不写 history） | 报告链 **5 → 4 轮**，重复调用 **0** |
| 用户指定月份优先（T002 回归） | 工具参数准确率回到 100% |
| 全局 list → `ContextVar`（P0 并发串台） | 两会话并发不串数据 |
| 4 个新指标（重复调用 / 报告轮次 / 状态栏规模 / 缓存命中） | 见上表 |

> 详见 [docs/RELEASE_v2.4.md](docs/RELEASE_v2.4.md)、[docs/v2.4_交付说明.pdf](docs/v2.4_交付说明.pdf)。

### v2.3 上下文预算与工具稳定性（不重建基线）

静态/动态 token 拆分（**静态占 91.3%**）+ 历史摘要（报告模式跳过）+ 旧工具结果清理 + 错误契约/重试 + 月份工具化 + 输出约束。
忠实度 89.6%（+6.3pp）、P95 7.96s（−31%）、成本 −6.8%。

### v2.2 检索质量（基线版本）

Q&A 边界切块（88→130 块）+ 多轮查询改写（门控/LLM/回退）+ 发布闸门 + 检索侧 A/B（block recall 87.5%→100%）。

---

## v1.x 里程碑（细节见评测报告）

| 版本 | 一句话 | 关键数字 |
|---|---|---|
| v1.0–v1.2 | 首个基线 → 报告链路 → P0 提示词（强制检索/禁编造/先追问） | 72% → 88.5% |
| v1.3–v1.4 | 六指标 → 九指标（来源/MRR/相关性/成本 + 越权拦截） | 忠实度 40%→70%、安全 50%→100% |
| v1.5–v1.6 | 双 provider 选型 + 提示词精简与交叉评审 | tokens_in −46%、成本 −37% |
| v1.7–v1.8 | 块级指标 → 官方语义重排 | recall@3 87.5%→100%、MRR 0.969 |
| v1.9 | 修 BM25 错位 bug（结论反转）+ 数字依据率 + 忠实度后处理 | 数字依据 89.6% |

完整迭代过程、失败分析、判据演进见 **[docs/评测报告.md](docs/评测报告.md)**（§1–§20）。

---

## 关键设计

1. **状态栏而不是"提示词要求"**：报告链的「第几步 / 参数齐没齐 / 下一步」由代码算，作为请求末尾一条消息注入；**不写回 state**，历史只追加 → 缓存零破坏、无陈旧状态（第二章"上下文窗口是一半检索"的落地）。
2. **信号灯工具 + 动态提示词**：空工具 `fill_context_for_report` 触发中间件写 `runtime.context["report"]`，`@dynamic_prompt` 据此在「客服」与「报告」两套提示词间切换。
3. **提示词/配置/路径外置且锚定**：模型名、切块参数、提示词全在 yml/txt；所有路径过 `get_abs_path()`，避免"库跟着工作目录跑"。
4. **评测驱动、口径不可横比**：判据版本 + 数据集 sha + 模型名写入基线；发布前跑闸门（通过率不降、tokens ≤ +10%、忠实度 ≥ 基线−2pp）。
5. **正确性优先于微优化**：并发串台用 `ContextVar` 修；工具失败有结构化降级标记；状态栏注入失败自动降级为"无状态栏"。

---

## 版本记录

| tag | 主题 |
|---|---|
| `v1.0-baseline` → `v1.9` | 基线 → 九指标 → 多 provider → 重排 → 数字依据（见评测报告） |
| `v2.0-eval-baseline` | 评测基座：48 条数据集冻结 + 子集 + 故障注入 + S002 口径 |
| `v2.1-metrics` | 度量可信：judge v2（j6）+ 横比闸门 + 重排成本入账 |
| `v2.2-retrieval` | 检索质量：Q&A 边界切块 + 多轮改写 + 发布闸门 |
| `v2.3-context` | 上下文与稳定性：静态/动态拆分 + 压缩 + 错误契约 |
| `v2.4-status-bar` | 状态栏 + 并发隔离 + 4 个新指标 |

## Roadmap

- [x] 评测基座 / 检索质量 / 上下文压缩 / 状态栏（v2.0–v2.4，均不重建基线）
- [ ] **v2.5（口径变更版）**：`get_report_data` 复合工具（报告 6→2 轮，目标 tokens_in −30%）+ checkpointer 记忆 / 用户画像 / thread 隔离；需更新 T001/T005 期望并重建基线
- [ ] **v2.6**：自研业务 MCP Server 接入（工单查询 / 多 server 路由 / 超时降级）
- [ ] 评测演进：judge v3（拒答题型豁免）+ 并发/长上下文用例扩到 60 条

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/评测报告.md](docs/评测报告.md) | 20 节完整评测史：判据迭代、人工校准、逐版本归因 |
| [docs/RELEASE_v2.4.md](docs/RELEASE_v2.4.md) | v2.4 逐项改动、子集验收、delta、已知问题 |
| [docs/v2.4_交付说明.pdf](docs/v2.4_交付说明.pdf) | 交付说明（3 页，可直接给面试官看） |
| [docs/待开Issue.md](docs/待开Issue.md) | 待开工单与优先级 |
| [eval/README.md](eval/README.md) | 评测脚本用法、基线纪律、常见问题 |
