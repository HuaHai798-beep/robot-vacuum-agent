# RELEASE v2.3 — 上下文预算与工具稳定性（不重建基线）

> tag：`v2.3-context`　｜　日期：2026-11-18　｜　对照基线：`eval/baselines/baseline_v22.json`（j6，sha `865424a67c8e`）
> **本版不产出新基线**：数据集/判据/模型不变，所有数字都是相对 v2.2 的 delta。

## 1. 一句话

> 先量出"每轮 91% 的输入 token 是静态提示"，再把历史摘要、工具清理、错误契约、月份工具和输出上限补上——**忠实度 +6.3pp、P95 −31%、成本 −6.8%**，但输入 token 只降 0.9%（原因见 §4）。

## 2. 子集验收

| 子集 | 命令 | 结果 |
|---|---|---|
| 报告链 | `--ids T001,T005` | 月份修复，忠实度转正 |
| 多轮 | `--tags multi_turn` | 8/8 通过 |
| 降级 | `--tags failure` | X001/X002 通过 |
| 补充集 | `--dataset eval/dataset_v23_supp.jsonl` | C005/L001/X003/X004 **4/4 通过** |

## 3. 全量对比（48 条）

| 指标 | v2.2 | v2.3 | delta |
|---|---|---|---|
| 通过率 | 100% | 100% | 0 |
| 忠实度 | 83.3% | **89.6%** | +6.3pp |
| 数字依据率 | 88.4% | 91.6% | +3.2pp |
| tokens_in | 2647 | 2622 | −0.9% |
| tokens_out | 322 | **258** | −19.9% |
| P50/P95 | 4.27/11.52s | **3.79/7.96s** | −31% |
| 成本（含重排） | 0.00828 | **0.00772** | −6.8% |
| 静态占比 | — | 91.3% | 新指标 |

## 4. 为什么没达到 −19% tokens_in

- 静态 token（system + 7 个工具定义）≈ **2395/例**，占 **91.3%**；
- 报告链路固定 **5 次**模型往返（`get_user_id → get_current_month → fill_context_for_report → fetch_external_data → 回答`），每轮重发静态部分；
- 在**不改工具契约**（不合并工具）的前提下，轮数锁死 → 输入 token 没有下降空间。

**结论**：−19%～−32% 必须走 v2.4 `get_report_data` 复合工具（6 轮→2 轮），届时 `expect_tools` 变化、数据集 sha 变化，属"口径变更版"，不能与本基线横比。

## 5. 闸门

| 检查 | 结果 |
|---|---|
| 数据集 sha 一致 | ✓ |
| 通过率不降 | ✓ |
| tokens_in ≤ +10% | ✓ |
| 忠实度 ≥ 基线−2pp | ✓ |

## 6. 交付物

- 代码：`utils/context_budget.py`、`agent/tools/context_middleware.py`、`config/context.yml`、`agent/tools/agent_tools.py`、`agent/react_agent.py`、两份 prompt、`eval/run_eval.py`
- 用例：`eval/dataset_v23_supp.jsonl`
- 文档：`docs/评测报告.md` §19、`docs/v2.3_交付说明.pdf`
- 结果：`eval/results/run_20261118_1959.jsonl`（主集）、`run_20261118_1951.jsonl`（补充集）

## 7. 已知问题

1. M004 仍不忠实（对比式追问丢主题词）→ v2.4；
2. S001/S004 被 judge 误判（把客服身份当事实主张）→ judge v3 拒答题型豁免；
3. 补充集忠实度 50%（C005 未给型号、L001 混淆最高值）→ 补充集不进门禁。
