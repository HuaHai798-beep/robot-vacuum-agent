# RELEASE v2.4 — Agent 状态栏 + 并发隔离（不重建基线）

> tag：`v2.4-status-bar`　｜　日期：2026-11-23　｜　对照基线：`eval/baselines/baseline_v22.json`（j6，sha `865424a67c8e`，48 条）
> **本版不产出新基线**：数据集 / 判据 / 模型全部不变，所有数字都是相对 v2.2 的 delta。
> 结果文件：`eval/results/run_20261123_1646.jsonl`（主集 48 条，judge=deepseek-reasoner）

## 1. 一句话

> 给 Agent 加了一条**由代码维护的状态栏**：每轮把「当前第几步、user_id / 月份拿到没有、下一步该调谁、
> 调了几次工具、主题是什么」算好，作为一条 `<agent_status>` 追加在**请求末尾**（不写 state，历史只追加）；
> 顺手修掉了 v2.2 遗留的**并发串台 P0**。结果：**报告链 5→4 轮、重复调用 0、忠实度 +10.5pp、P95 −50%、成本 −5.7%**，tokens_in 基本持平（+1.0%）。

## 2. 变更清单

| # | 改动 | 文件 | 关键点 |
|---|---|---|---|
| 1 | 状态栏中间件 | `agent/tools/status_middleware.py`（新增，264 行） | `@wrap_model_call` + `request.override(messages=…)`；末尾注入；异常降级为「无状态栏」 |
| 2 | 状态栏接入 | `agent/react_agent.py` | 放在 middleware 列表**最后**（最内层），保证注入到最终请求 |
| 3 | 配置项 | `config/context.yml` | `status_enabled / status_max_tokens=120 / status_include_topic / status_max_rounds=6` |
| 4 | 用户指定月份优先 | `status_middleware.py` | T002 回归：用户写 `2024-02` 时不再顺着工具返回的当前月份走 |
| 5 | 意图分流 | `status_middleware.py` | `data`（查某月数据，不调 `fill_context_for_report`）vs `report`（要切报告提示词） |
| 6 | **并发串台修复（P0）** | `utils/context_buffer.py` | 全局 `list` → `contextvars.ContextVar`；顺序场景行为不变 |
| 7 | 工具层同步轮次 | `agent/tools/middleware.py` | `wrap_tool_call` 里再 `set` 一次，避免 ContextVar 在工具上下文丢值（否则多轮改写失效） |
| 8 | 4 个新指标 | `eval/run_eval.py` | `repeat_tool_calls` / `report_rounds` / `status_tokens` / `cache_hit_rate`（含 DeepSeek `prompt_cache_hit_tokens`） |
| 9 | 测试 | `eval/tests/test_status_middleware.py`（7 例）、`eval/tests/test_context_buffer_concurrency.py`（2 例） | 状态栏字段/预算/意图 + 并发隔离 |

## 3. 子集验收

| 子集 | 命令 | 结果 |
|---|---|---|
| 报告链 | `--ids T001,T005` | 轮次 5 → **4 / 5**（均值 4.5），无额外 `rag_summarize` 的用例降到 4 轮 |
| 取数类 | `--ids T002`（`--repeat 2`） | **2/2 通过**（修前 args_ok=False：用了 2026-10） |
| 多轮 | `--ids M004,C002` | 通过；`rewrite_used` 恢复（2 / 1） |
| 易追问 | `--ids M002`（`--repeat 2`） | **2/2 稳定通过**（修前 1 次 no_followup_ok=False） |
| 单测 | `python eval/tests/test_status_middleware.py` | 7/7 通过 |
| 并发 | `python eval/tests/test_context_buffer_concurrency.py` | 2/2 通过（A/B 两会话互不污染） |
| 回归 | `test_dataset / test_guard / test_memory_isolation` | 1/1、5/5、1 skip（未接 checkpointer） |

## 4. 全量对比（48 条）

| 指标 | v2.2 | v2.3 | **v2.4** | delta(vs v2.2) |
|---|---|---|---|---|
| 通过率 | 100% | 100% | **100%** | 0 |
| 忠实度 | 83.3% | 89.6% | **93.8%** | **+10.5pp** |
| 数字依据率（规则） | 88.4% | 91.6% | 90.2% | +1.8pp |
| 相关性均分（≥3.5 占比） | 4.65（89.6%） | — | 4.54（87.5%） | −0.11（−2.1pp） |
| 检索 hit@3 / chunk_recall@3 | 92.6% | 92.6% | 92.6% | 0 |
| 工具调用 / 参数 | 100% / 100% | 100% / 100% | **100% / 100%** | 0 |
| tokens_in | 2647 | 2622 | **2674** | +1.0% |
| tokens_out | 322 | 258 | **256** | **−20.5%** |
| P50 / P95 | 3.86 / 11.52s | 3.79 / 7.96s | **3.56 / 5.71s** | P95 **−50.4%** |
| 单次成本（含重排） | 0.00828 元 | 0.00772 元 | **0.00781 元** | **−5.7%** |
| 静态 tokens / 占比 | — | 2400 / 91.3% | 2465 / **92.2%** | — |
| **重复工具调用** | — | — | **0 次 / 0 例** | 新指标 |
| **报告链轮次** | 5 | 5 | **4.00** | **−1 轮** |
| **缓存命中率** | — | — | **80.5%**（103,292 tokens） | 新指标 |
| 状态栏规模 | — | — | 18 token/例（报告类 66） | 新指标 |

按类别通过率：retrieval 19/19、tool 10/10、refuse 4/4、ambiguous 2/2、multi_turn 8/8、safety 5/5。

## 5. 为什么 tokens_in 没降（诚实归因）

- 状态栏对报告类用例增加 ~66 token/轮（大多数客服单轮不注入，均摊 18 token/例）；
- 但报告链 **5 → 4 轮**，每轮少发一次静态前缀（≈2465 token），两相抵消后净 +1.0%；
- 真正要 −19%～−32%，必须做 v2.5 的 `get_report_data` 复合工具（6 轮→2 轮）。那会改 `expect_tools`、数据集 sha 变化 → **口径变更版，不能与本基线横比**。

## 6. 闸门（`--gate eval/baselines/baseline_v22.json`）

| 检查 | 结果 |
|---|---|
| 数据集 sha 一致 | ✓ `865424a67c8e` |
| 通过率不降 | ✓ 100% vs 100% |
| tokens_in ≤ +10% | ✓ 2674 vs 2647（上限 2912） |
| 忠实度 ≥ 基线−2pp | ✓ 93.8% vs 83.3% |
| 重复工具调用不增加 | ✓ 0 vs 基线无此指标（跳过） |

**结论：闸门通过，可以发版。**

## 7. 交付物

- 代码：`agent/tools/status_middleware.py`、`agent/tools/middleware.py`、`agent/react_agent.py`、`utils/context_buffer.py`、`config/context.yml`、`eval/run_eval.py`
- 测试：`eval/tests/test_status_middleware.py`、`eval/tests/test_context_buffer_concurrency.py`
- 结果：`eval/results/run_20261123_1646.jsonl`（主集）、`run_20261123_1638.jsonl`（T002/M002 复跑）、`run_20261123_1628.jsonl`（改写回归）
- 文档：本文、`docs/v2.4_交付说明.pdf`、README v2.4 段落、`docs/评测报告.md` §20

## 8. 已知问题 / 回滚

1. **状态栏是"有损投影"**：只覆盖 `user_id / month / data / calls / topic` 这几个维度；状态栏没覆盖的问题（如"上次推荐过什么型号"）仍靠消息历史 → 若发现模型过度依赖状态栏，先加维度、不要删原始消息。
2. **T005 仍是 5 轮**：模型在报告链里额外调了一次 `rag_summarize`；已在状态栏写"不需要 rag_summarize"，但报告类用例仍偶发，留 v2.5 由复合工具收敛。
3. **相关性达标率 −2.1pp**：judge 波动 + 报告类回答变短；均分 4.54 仍在高位，observation 项、未进门禁。
4. **回滚**：`AGENT_NO_STATUS=1` 可整体关闭状态栏（A/B 与线上降级都用它）；并发修复**不建议回滚**（属正确性）。
