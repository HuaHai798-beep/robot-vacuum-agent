# eval/ 评测目录说明

> 本目录是项目的**评测体系**：数据集 → 跑批 → 评分 → 基线 → 回归。
> 一条纪律先记住：**模型、判据版本、数据集 hash 三者任一变化，基线不可横比。**

## 文件清单

| 文件 | 作用 | 是否可删 |
|---|---|---|
| `dataset.jsonl` | 评测集（30 条，5 类：检索/工具/拒答/追问/多轮），每条带期望工具、关键事实、gold 来源 | ❌ 核心资产 |
| `run_eval.py` | 跑批主脚本：检索指标 + 端到端 + 规则评分 + LLM judge + 成本 + 基线对比 | ❌ |
| `regrade.py` | 离线重打分：用当前判据重评历史 `results/*.jsonl`，**不重新调模型、不花钱** | ✅ 有用则留 |
| `judge_calibration.py` | 人工标注 vs LLM judge 的一致率计算 | ✅ |
| `judge_calibration.json` | 20 条人工标注（ground truth，含 faithful/relevance/理由） | ✅ 校准依据 |
| `judge_calibration_report.json` | 校准报告（n=20，一致率、逐条差异） | ✅ 面试可引用 |
| `exp_rerank_compare.py` / `.json` | 重排 A/B 实验：BM25 vs 官方语义重排，产出结论 | ✅ 实验证据 |
| `prices.yml` | 模型单价表（元/1K tokens），用于成本核算；rerank 可按次或按 token | ✅ |
| `tests/test_guard.py` | 忠实度后处理 + 数字依据率的**离线单测**（不联网、不花钱） | ✅ |
| `baselines/` | 9 个历史基线快照（见下表），供追溯"某次改动的对照物" | ✅ |
| `baseline.json` | **legacy**：CLI 默认路径，缺 `model` / `judge_version` 字段 → 与当前模型不可横比（脚本会警告并忽略） | ⚠️ 保留兼容 |
| `results/` | 每次跑批的原始明细（工具调用、答案、token、耗时、检索指标），可回放、可重打分 | 建议定期归档 |

## 当前该用哪个基线做对比？

```powershell
# 当前有效的基线（deepseek-chat + 判据 j5 + 30 条）
python eval/run_eval.py --baseline eval/baselines/baseline_v19.json
```

`baseline.json` 只作为默认路径存在，**不要拿它做对比**（缺元数据，脚本会提示不可横比）。

## 历史基线一览

| 文件 | 版本 | 模型 | 判据 | 条数 | 通过率 | hit@3 | 关键改动 |
|---|---|---|---|---|---|---|---|
| `baselines/baseline_v0.json` | v1.0 | （未记录） | — | 25 | 72.0% | 87.5% | 初版：规则评分 + 25 条用例 |
| `baseline.json` | v1.1 | （未记录） | — | 26 | 73.1% | 87.5% | 补 1 条用例 |
| `baselines/baseline_qwen_plus.json` | v1.2 | qwen-plus | — | 26 | 96.2% | 93.8% | 判据修正（破折号归一化 + 同义词或） |
| `baselines/baseline_qwen_plus_v3.json` | v1.3 | qwen-plus | j4 | 30 | 86.7% | 93.8% | 六类指标（安全/忠实度加入）→ 数字变低但更真实 |
| `baselines/baseline_qwen_plus_v4_regraded.json` | v1.4 | qwen-plus | j5 | 30 | 83.3% | 93.8% | 判据升级后**离线重打分** |
| `baselines/baseline_deepseek_v1_regraded.json` | v1.5 | deepseek-chat | j5 | 30 | 100% | 93.8% | 接入 DeepSeek，双 provider |
| `baselines/baseline_deepseek_v2.json` | v1.6 | deepseek-chat | j5 | 30 | 96.7% | 93.8% | 提示词精简 + 交叉评审 |
| `baselines/baseline_deepseek_v3.json` | v1.7 | deepseek-chat | j5 | 30 | 96.7% | 93.8% | 块级指标 + BM25 实验（结论：无效） |
| `baselines/baseline_deepseek_v4.json` | v1.8 | deepseek-chat | j5 | 30 | 100% | **100%** | 官方语义重排 `qwen3.7-text-rerank` |
| `baselines/baseline_v19.json` | v1.9 | deepseek-chat | j5 | 30 | 96.7% | **100%** | 修 BM25 错位 bug + 成本埋点 + judge 校准 |

> 86.7% → 83.3% 不是变差：**判据从 j4 升到 j5 更严**，同一批答案重打分自然下降。

## 常用命令

```powershell
cd /d D:\开发app\pycharm\Agent项目
set DASHSCOPE_API_KEY=sk-xxxx
set PYTHONPATH=%CD%

# 1) 只测检索层（不花对话额度，秒级；只看 hit@3 / 块召回 / MRR）
python eval/run_eval.py --no-llm

# 2) 端到端 + LLM judge（默认基线 = baseline.json，建议显式指定当前基线）
python eval/run_eval.py --judge --baseline eval/baselines/baseline_v19.json

# 3) 稳定性：同一条重复跑 N 次，看结果是否一致
python eval/run_eval.py --repeat 2

# 4) 确认变好后再写基线（no-llm 结果会被拒绝写入）
python eval/run_eval.py --judge --baseline eval/baselines/baseline_v19.json --save-baseline

# 5) 判据升级后，用旧答案离线重打分
python eval/regrade.py eval/results/run_20261114_0006.jsonl

# 6) judge 校准 / 重排实验 / 离线单测
python eval/judge_calibration.py --results eval/results/run_20261114_0006.jsonl
python eval/exp_rerank_compare.py
python eval/tests/test_guard.py
```

## 判读顺序（很重要）

```
① 先看检索层：hit@3 / 块召回@3 / 块 MRR —— 检索不行，后面全白搭
② 再看工具层：工具命中率 / 参数正确率
③ 再看回答层：规则事实命中、数字依据率、judge 忠实度/相关性
④ 最后看成本与延迟：tokens、P50/P95
```

## 三条纪律

1. **不可横比**：`model` / `judge_version` / `dataset_sha256` 任一不同，就不要直接比通过率（脚本会警告并忽略基线）。
2. **no-llm 的通过率是假的**：非检索类用例在该模式下直接判 True；只看检索指标，且不允许写基线（`run_eval.py` 已加防呆）。
3. **判据也是代码**：判据改动会改变历史分数，必须升 `JUDGE_VERSION`，并用 `regrade.py` 重算旧结果，否则时间线不可信。

## 目录结构

```
eval/
├─ dataset.jsonl                  # 30 条用例
├─ run_eval.py                    # 跑批主脚本
├─ regrade.py                     # 离线重打分
├─ judge_calibration.py/.json/_report.json
├─ exp_rerank_compare.py/.json
├─ prices.yml
├─ tests/test_guard.py
├─ baselines/                     # 9 个历史基线
│  ├─ baseline_v0.json … baseline_v19.json
├─ baseline.json                  # legacy 默认路径（不可横比）
└─ results/                       # 每次跑批明细（可回放/重打分）
```
