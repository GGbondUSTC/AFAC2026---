# 外部高价值方案调研：Kaggle / TAAC2026 / GitHub

调研时间：2026-06-18

当前本赛题状态（已结合 2026-06-19 后续线上验证更新）：

- 官方最佳基线：`v32`，A 榜 `0.6278 / 0.7590 / 0.4966`，根目录提交包已恢复为 v32。
- `v31` 验证了多路召回能显著提高候选 recall，但自由 LightGBM 重排线上回退；`v32` 仅在 `seq_len>80` 段冻结 top5、从 rank6 起最多插入两个候选，成功获得小幅正向转化。
- `v33`/`v34` 的 top3-frozen 半自由 blend 仍低于 v32，说明当前重点不是扩大改动范围，而是提高候选质量与 gate 校准的分段证据。

## 参考源

### Kaggle 推荐与点击类比赛

- H&M Personalized Fashion Recommendations 银牌方案：`https://github.com/Wp-Zhang/H-M-Fashion-RecSys`
  - 两类召回策略 + 多个排序模型；不同召回策略的结果差异较大，融合能提高稳健性。
- H&M 方案复盘：`https://ajisamudra.medium.com/silver-medal-solution-on-kaggle-h-m-personalized-fashion-recommendations-a0878e1eae63`
  - 明确使用“候选召回 + 排序”两阶段架构；召回包括近期流行、历史购买、共同购买、相似价格、TFR retrieval。
- OTTO 3rd place：`https://github.com/TheoViel/kaggle_otto_rs`
  - 经典 candidate extraction + reranker pipeline。
- OTTO co-visitation notebook：`https://github.com/nlztrk/OTTO-Multi-Objective-Recommender-System/blob/main/0.%20Covisitation.ipynb`
  - 基于会话内共同出现关系构造 co-visitation matrices。
- Santander Product Recommendation 2nd place write-up：`https://medium.com/kaggle-blog/santander-product-recommendation-competition-2nd-place-winners-solution-write-up-3384f2a34d5b`
  - 时间/季节验证、按产品建模、lag features、概率后处理、MAP 位置优化、LB 过拟合反思。
- Outbrain Click Prediction solution：`https://github.com/alexeygrigorev/outbrain-click-prediction-kaggle`
  - FTRL/SVM、MTV target encoding、XGB/ET、FFM、pairwise ranker 组合。
- 推荐系统竞赛 TOP 开源方案汇总：`https://github.com/ChuanyuXue/Recommender-Systems-Competition-TopSolutions`

### TAAC2026 / 腾讯广告算法大赛

- lizuju/TAAC-2026：`https://github.com/lizuju/TAAC-2026`
  - 单模型、单 checkpoint、单推理流程；主要增益来自全局时间特征、Focal Loss、user dense 分组 projector、step-level checkpoint selection。
- Puiching-Memory/TAAC_2026：`https://github.com/Puiching-Memory/TAAC_2026`
  - 工程化实验工作区；强调统一 tokenization、可堆叠 backbone、流式数据管线、动态增强、平台一致性。
- YodesYang/KDDCup2026-TencentAds-UniRec：`https://github.com/YodesYang/KDDCup2026-TencentAds-UniRec`
  - Top 5.1% industrial track；强调 time-aware sequence buckets、public-tail-oriented validation、多任务正则、辅助验证窗口和 LB 相关性分析。
- zhangzucheng/taac2026_rec_to_312：`https://github.com/zhangzucheng/taac2026_rec_to_312`
  - 从 baseline 到 0.82969：语义化序列分段、long/short window、请求时间特征、DoubleHash 高基数 embedding、dense-only EMA、label smoothing、sparse embedding cold-restart。
- Rongfeng-Guo/Rank51-TAAC2026-KDDCUP：`https://github.com/Rongfeng-Guo/Rank51-TAAC2026-KDDCUP`
  - 文件说明中保留 recent-window、overlap-tail、周期时间特征、序列 hour/day side-info、mean-DIN query pooling、UserDenseGroupProjector、大基数字段 hash、focal loss 等实现点。

### 图半监督分类

- Correct and Smooth：`https://openreview.net/forum?id=8E1-f3VhX1o`
  - 用浅层模型 + label propagation 的 error correlation / prediction correlation，在多个 transductive node classification benchmark 上达到或接近 GNN，参数和耗时更低。
- Feature Propagation：`https://github.com/twitter-research/feature-propagation`
  - 用图扩散处理缺失特征；对本赛题分类任务的缺失/稀疏属性可作为候选。

## 对本赛题最有价值的迁移点

### 1. 推荐任务应继续推进“多路召回 + 学习排序”，但以 v32 的保守 gate 为基线

外部证据：

- H&M 和 OTTO 的高分方案都不是单一路径排序，而是先用多种召回生成候选，再用排序模型学习交互。
- 我们 v29/v30 只是在 v17/v24 生成的 top25 里做尾部重排，实际没有扩大候选召回空间；v30 只改 213 行且 top1 不变，线上持平是合理结果。

可落地方案：

1. 生成候选池 `top80` 左右，而不是只在 top25 内换序：
   - v17/v29 规则模型 topN。
   - 历史重复候选：用户历史中出现过的 item，按 count/recency/repeat concentration。
   - item-item co-visitation：从 `item_seq_raw -> target_iid`、last item -> target、suffix2 -> target、suffix3 -> target 统计。
   - 用户匿名特征分组热度：`u_cat_* -> target_iid`。
   - item 匿名特征相似候选：同 `item.csv` 匿名属性的 target prior / popularity。
   - 长历史专用候选：只对 `seq_len>=21` 使用更大 pool，短历史保持保守。

2. 构造候选级特征并训练 LightGBM/XGBoost ranker：
   - base rank、base score proxy、是否来自 v29 top10/top25/top50。
   - count / recency / last equality / max_count / repeat concentration。
   - last item target prior、suffix2 target prior、suffix3 target prior。
   - user group prior、item group prior、user-item group cross prior。
   - target popularity、target item feature bucket、history length bin。
   - 候选来源 flags：repeat、co-visitation、group-pop、base-model。

3. 训练数据来自多个 masked validation split：
   - 对每个训练行隐藏 target，生成候选，target 在候选中标 1，其余候选标 0。
   - 先用 LambdaRank / pairwise ranker；若本地依赖麻烦，先用二分类 `predict_proba` 后按概率排序。
   - 评估必须同时报告 recall@80、NDCG@10、exact-length weighted NDCG、与 v29 的 row-level delta。

为什么优先级最高：

- 当前推荐差距约 `0.00979`，单纯重排 v29 top25 的天花板太低；但 v31 已证明扩大召回后必须限制排序器的作用范围。
- 如果 target 不在现有 top25，任何 `alpha/count_weight` 都不可能修复。

### 2. 建立更像线上分布的推荐验证，不再只看 seed42 或单一 masked split

外部证据：

- TAAC2026 工业赛道方案明确强调 public-tail-oriented validation、辅助验证窗口和 leaderboard-correlation analysis。
- Santander 复盘明确指出 public/private split 下持续追小分容易过拟合；很多看似合理的小改动不会转化。

可落地方案：

1. 固定一个 `v32_official_guard`：
   - 新候选必须和 v32 比，而不是只和 v17/v24/v29 比。
   - 记录 mean/min/positive-splits。
   - 对推荐微改，门槛提高到：相对 v32 mean >= `+0.0005` 且 min >= `0`；否则不提交。

2. 分段验证至少覆盖：
   - exact history length：`0/1/2/3/4-10/11-20/21-30/31-80/>80`。
   - repeat concentration：低/中/高。
   - target popularity：head/mid/tail。
   - user group：匿名用户特征高支持组合。

3. 对每个候选输出三张表：
   - 哪些段提升，哪些段回撤。
   - 改动行数、top1_changed、top10 overlap。
   - changed rows 的 base rank 分布：只在 rank9/10 互换通常线上不可见。

### 3. 引入 co-visitation / suffix transition，而不是只用历史 count

外部证据：

- OTTO 方案的核心之一是会话 co-visitation，按点击/购物车/订单关系构造候选矩阵。
- H&M 召回也用“共同购买”作为规则候选。

本赛题映射：

- `A推荐/train.csv` 的 `item_seq_raw` 和 `target_iid` 本质上就是“历史序列 -> 下一目标 item”。
- 现有 v22-v30 使用了历史 count/recency，但对“看到 A 后下一个常买 B”这种转移关系还不够强，尤其是 target 不在用户历史时。

可落地实验：

1. 统计 `last1 -> target`、`last2 suffix -> target`、`last3 suffix -> target`。
2. 加 shrinkage：
   - `score = log_lift * support_weight`
   - support 低于 20/50 的 key 不用。
   - 用 global target popularity 平滑。
3. 将 co-visitation 只用于候选召回，不直接覆盖 top1。
4. 再由 ranker 或 segment gate 决定是否进入 top10。

### 4. 用匿名用户/物品特征做“分组 projector”的规则版

外部证据：

- TAAC2026 多个方案都强调 sparse/dense feature tokenization、user dense 分组 projector、RankMixer-style non-sequence tokens。
- Outbrain 方案使用 mean target value、类别交叉和 pairwise ranker。

本赛题映射：

- 我们没有原始语义，但有 `user.csv` / `item.csv` 的匿名特征。
- 当前推荐主要依赖行为序列，用户/物品匿名特征只是零历史或轻量 prior，利用不充分。

可落地实验：

1. 建立高支持 group prior：
   - user 单列：`u_cat_i -> target`
   - user pair：`u_cat_i,u_cat_j -> target`
   - item 单列：`target item feature -> target prior`
   - user group x target item feature：高支持交叉。

2. 不要直接全局加权，先用于候选 ranker 特征：
   - `log_odds(group,target)`
   - `lift(group,target)`
   - `support(group,target)`
   - shrink 后的 posterior probability。

3. 对短历史和零历史单独评估：
   - `seq_len=0/1/2/3` 可能更依赖 user/item feature。
   - v15 短历史神经塔失败，说明不能大幅替换；但 LightGBM/ranker 特征可以更保守。

### 5. NDCG/MAP 目标下要做位置级后处理，而不是只看 item score

外部证据：

- Santander 方案对 MAP 做了后处理和排序位置优化；某些产品对存在依赖关系时，概率排序不是最终最优。

本赛题映射：

- 我们当前 top1 一直冻结，避免掉分但也限制上限。
- NDCG@10 对前几位更敏感，若能识别“v29 top1 错但 top2/3 有强信号”的 segment，才可能带来可见线上提升。

可落地实验：

1. 找 v29 在 masked split 中 top1 错、target 在 top2/top3/top5 的行。
2. 训练一个“是否允许 top1 swap”的二分类 gate：
   - 特征：top1/top2 分差、top2 是否历史重复、top2 transition prior、history length、repeat concentration。
3. 只在 gate 高置信时交换 top1/top2。
4. 提交门槛必须严格：top1 swap 是高风险，要求多 split 正向且最小回撤为正。

### 6. 分类任务可以试 Correct & Smooth 风格的残差传播，但优先级低于推荐

外部证据：

- Correct & Smooth 说明浅层模型 + label propagation 的 error/prediction correlation 可以接近或超过 GNN，并且更快。

本赛题映射：

- 我们分类已是 label propagation + Ridge fallback + pair gate，线上分类 `0.7590` 已较稳。
- 仍可小步尝试 C&S：
  - base model 仍用当前 Ridge/LP 输出。
  - 在训练节点上计算 residual/error。
  - 对 residual 做图传播，再修正测试节点 logits。
  - 再做 prediction smoothing。

提交门槛：

- 分类每次只能改少量高置信行，参考 v21 的 pair-gate。
- 10 split 平均提升要明显，最小 split 不能为负。

## 推荐执行顺序

### P0：先做 v32 gate 下的推荐候选召回审计

目标：确认 v31-style top80 候选在 v32 gate 接受/拒绝行上的 target recall、校准度与可安全插入上限。

输出：

- masked split 上 target 是否在 top10/top25/top50/top80。
- 按 history length / repeat concentration / target popularity 分段。
- 如果 top80 recall 明显高于 top25，进入 ranker 路线；如果 top80 也低，先做新召回。

### P1：实现 co-visitation 召回

目标：把 target 拉进候选池，而不是只在已有 top25 内重排。

最小实现：

- `last1 -> target`
- `suffix2 -> target`
- `hist item -> target`
- shrink + min support
- 每路召回 top20，合并去重到 top80。

### P2：实现 LightGBM/XGBoost ranker 或二分类 ranker

目标：学习“什么时候 repeat、transition、user prior、item prior 该生效”。

先不追复杂神经模型；树模型更适合匿名类别统计、support、rank、count 这类 tabular 特征。

### P3：建立 v32 官方 guard

任何新 zip 生成前必须通过：

- 相对 v32 多 split mean >= `+0.0005`。
- 相对 v32 min >= `0`。
- 至少一个明确 segment 有稳定提升。
- 不能只靠 rank9/rank10 互换和小于 300 行改动。

### P4：分类 C&S 小实验

推荐任务出现瓶颈时再做。分类线上差距也有约 `0.00618`，但当前本地验证更容易过拟合；必须保持小步和高置信。

## 不建议继续投入的方向

- 继续围绕 v29/v30 的 `alpha/count_weight/base_pool` 做微调。v30 已证明同类微增线上不可见。
- 复现 v31 那样的自由 LightGBM 重排，或只靠“冻结 top3”放宽 v32 gate；v33/v34 已给出直接的负面线上证据。
- 直接上复杂 Transformer/SSM 长序列模型。TAAC 的成功依赖大规模序列数据和训练标签，本赛题推荐训练规模与提交格式更适合先做候选召回和 ranker。
- 大幅改短历史神经 rerank。v15 已经显示短历史内部验证和线上存在明显错配。
- 没有分段证据的全局 user/item prior。容易在长历史段破坏已经有效的 repeat/transition 信号。
