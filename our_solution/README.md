# 自有方案：稀疏反馈自动实验控制

本方案基于 baseline 的工程目标重新实现，但不复用其 PyTorch dense GNN / GRU4Rec 训练核心。设计原则是轻量、串行、可复现、可审计。

## 核心差异

- 不依赖外部 LLM 或未备案服务，默认离线可运行。
- 不依赖 PyTorch，避免复现环境缺少 GPU 或 torch 时失败。
- 分类任务使用稀疏特征工程 + 线性模型搜索，自动适配 A/B 榜不同特征维度和类别数。
- 推荐任务使用可解释混合排序器：全局目标热度、历史 item 到目标 item 的转移、重复购买倾向、用户匿名特征分组和 item 匿名特征先验。
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
   - 是否排除历史 item。
3. 根据 NDCG@10 选择最佳配置。
4. 用全部训练集重建排序器并生成 `A2.csv`。

## 已知取舍

- 该方案优先保证稳健和可复现，不追求复杂深度模型。
- 推荐任务中训练目标 item 覆盖范围可能远小于候选集，混合排序器会自然偏向训练目标分布，并用 item 特征先验补足候选排序。
- 若复赛数据中目标分布更均匀，可以扩大 `candidate_configs()` 中的 user/item 特征权重搜索空间。

## Qwen API

主预测流程默认不调用 Qwen，避免线上复现受 API 可用性影响。`src/qwen_client.py` 提供可选客户端，支持两种 `.env` 格式：

```text
QWEN_API_KEY=your-key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

也兼容单行 raw key 文件。建议使用标准 `KEY=value` 格式，便于后续扩展。
