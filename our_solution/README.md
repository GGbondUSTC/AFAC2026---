# 自有方案：稀疏反馈自动实验控制

本方案基于 baseline 的工程目标重新实现，并在多轮 A 榜反馈后形成当前主链路。设计原则是串行、可复现、可审计，并把每次提交结果落到版本台账。

当前官方 A 榜最佳为 `v32`：总分 `0.6278`，分类 `0.7590`，推荐 `0.4966`。`v32` 保留 v31 的 top80 多路召回与 masked-history LightGBM 训练，但只在 `seq_len>80` 段进行保守插入：冻结 top5、从 rank6 起最多插入 2 个候选。它把 v31 的候选召回增益转化为官方推荐 `+0.0002`、总分 `+0.0001`（相对 v29/v30）。

当前根目录 `prediction.zip` 为 `v32`。后续 v33/v34 仅对 `>80` 段进行 top3-frozen LightGBM blend：v33 回退到推荐 `0.4963`，v34 收缩改动后回到 `0.4964`，均未超过 v32，因此没有替代当前包。

## 核心差异

- 默认离线可运行；可用 `--use_qwen_agent` 让 Qwen API 额外提出候选配置，再由本地验证筛选。
- 基础链路不强依赖 PyTorch；如果本地存在 PyTorch，则启用零历史用户神经塔和后续推荐候选。
- 分类任务使用图平滑、label propagation、Ridge fallback，并在 `v21` 通过少量高置信 pair-gate 改动提升到官方 `0.7590`。
- 推荐任务使用可解释混合排序器：全局目标热度、历史 item 到目标 item 的转移、重复购买倾向、用户匿名特征分组、item 匿名特征先验、零历史神经融合和中长历史尾部重排。
- 当前默认推荐搜索首位为官方最佳候选 `v32_ranker_gate_gt80_f5_i6_l8_ins2`；`v29` count-only 重排仍是其基线，v31/v33/v34 保留为召回与 gate 校准探针。
- `v27_medium_long_conditional_repeat_gate` 保留为失败探针：它在 v24 尾部重排上加入 repeat concentration gate 与 last/suffix/user-group 条件先验，但 3 split vs v24 的 exact-weighted mean delta 为 `-0.000878`。
- 每个候选配置都进行内部验证，真实记录配置、反馈、下一步策略和耗时，输出 `trajectory_B1.json` / `trajectory_B2.json`。

## 运行方式

在项目根目录或 `our_solution` 目录均可运行：

```powershell
python .\our_solution\run.py --budget 5 --seed 42
```

默认读取：

- 分类数据：`../A分类/A1.npz`
- 分类模板：`../A分类/sample_submission.csv`
- 推荐数据：`../A推荐`

指定 B 榜数据时：

```powershell
python .\our_solution\run.py `
  --cls_data <B1.npz> `
  --cls_sample <B1_sample_submission.csv> `
  --rec_data <B2_rec_data_dir> `
  --output_dir .\our_solution\output_b `
  --budget 5 `
  --time_limit 7200
```

## 输出

```text
our_solution/output/
  submission/
    A1.csv
    A2.csv
    trajectory_B1.json
    trajectory_B2.json
  prediction.zip
  classification_result.json
  recommendation_result.json
  run_summary.json
```

`prediction.zip` 只包含 `A1.csv` 和 `A2.csv`，适合 A 榜预测提交。B 榜需要同时提交 `trajectory_B1.json` 与 `trajectory_B2.json`。

## 方法摘要

### 产品分类

1. 从 `.npz` 还原 CSR 邻接矩阵和节点特征。
2. 构造稀疏特征：
   - 原始节点特征行归一化；
   - 1-hop / 2-hop 图平滑特征；
   - 基于训练标签的 label propagation 特征；
   - 入度、出度、总度等结构特征。
3. 串行尝试 Ridge、SGD、LogisticRegression 等线性模型。
4. 根据分层验证集 Accuracy 选择最佳配置。
5. 用全部公开训练节点重训并生成 `A1.csv`。

### 产品推荐

1. 按 `target_iid` 分层划分内部验证集。
2. 串行尝试多组排序器权重：
   - 目标 item 全局热度；
   - 用户历史 item 到目标 item 的转移统计；
   - 历史重复倾向；
   - 用户匿名类别特征分组统计；
   - item 匿名类别特征先验；
   - 是否排除历史 item；
   - 零历史用户 PyTorch 用户特征神经塔；
   - 短历史保守 rerank；
   - `seq_len>=21` 中长历史 top1-frozen tail rerank。
3. 根据 NDCG@10 选择最佳配置。
4. 用全部训练集重建排序器并生成 `A2.csv`。

## 验证与探针工具

```text
our_solution/src/validation.py
our_solution/tools/v19_multisplit_eval.py
our_solution/tools/v19_grid_eval.py
our_solution/tools/v23_long_history_grid_eval.py
our_solution/tools/data_audit_rec_segments.py
our_solution/tools/classification_grid_eval.py
```

当前经验：

- 短历史神经塔和 v19 count/Bayes rerank 离线不稳，不作为默认提交主线。
- `v24` 中长历史尾部重排已经线上转化。
- `v25` 和 v26 级别的微小离线增益没有可见线上转化。
- `v29` 将 v25 的中长历史尾部重排改为 count-only 信号，在 v17 基座三 split exact-weighted delta 为 `+0.001509 / min +0.001298`，官方推荐从 `0.4962` 提升到 `0.4964`。
- `v31` 的 seed42 candidate recall@80 从 v29 reference 的 `0.903206` 提升到 `0.963009`，但自由 LightGBM 重排线上回退到推荐 `0.4950`；召回增益不能直接当作排序增益。
- `v32` 对 v29 的五 split exact-weighted delta mean/min 为 `+0.000155/0.000000`，只改 271 行 A2、top1 不变，并上线达到推荐 `0.4966`。
- v33/v34 表明只冻结 top3 仍不足以让半自由 tail blend 稳定；后续先审计候选来源、rank gap、支持度和 v32 gate 的接受条件。

## 已知取舍

- 该方案优先保证稳健和可复现，不追求复杂深度模型。
- 推荐任务中训练目标 item 覆盖范围可能远小于候选集，混合排序器会自然偏向训练目标分布，并用 item 特征先验补足候选排序。
- 若复赛数据中目标分布更均匀，可以扩大 `candidate_configs()` 中的 user/item 特征权重搜索空间。
- A 榜后续提分应先看 `docs/DATA_FIRST_IMPROVEMENT_PLAN.md`，以 v32 为基线；没有候选质量或分段证据时，不要提交微小的 rerank 改动。

## Qwen API

主预测流程默认不调用 Qwen，避免线上复现受 API 可用性影响。需要试 API 时显式打开：

```powershell
python .\our_solution\run.py --budget 5 --seed 42 --use_qwen_agent --qwen_rounds 3 --version vN --note "qwen-assisted config search"
```

当前接入方式不是让 LLM 直接预测标签或商品，而是让 Qwen 基于已验证指标提出少量候选配置；代码会对白名单参数做范围清洗，再用现有内部验证决定是否采用。`src/qwen_client.py` 提供可选客户端，`src/qwen_agent.py` 负责 JSON 解析与调用报告。支持两种 `.env` 格式：

```text
QWEN_API_KEY=your-key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

也兼容单行 raw key 文件。建议使用标准 `KEY=value` 格式，便于后续扩展。
