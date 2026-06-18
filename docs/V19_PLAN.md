# v19 历史提分方案与结论

> 当前状态：该方案已完成 probe，未通过提交门槛，未生成 `versions/v19/` 提交包。当前官方最佳是 `v24`，下一轮主线已转为 `docs/DATA_FIRST_IMPROVEMENT_PLAN.md` 的数据优先审计。

## 复盘结论

- v19 的 exact-length validation 工具和短历史 count/Bayes tail rerank 已实现。
- `v19a_short_len3_only_alpha04` 在 3 split 上为负，保守 grid 最优也接近 0 且存在负 split。
- 结论：短历史 count/Bayes rerank 没有足够证据替代 `v17`，更不能替代后续已验证的 `v24`。
- 后续不要继续围绕 v19 参数微调，除非新的数据审计发现明确的短历史分布错配。

## 核心判断

v19 不继续调 v17/v18 的神经塔强度。

- v15 已证明短历史神经塔在内部 masked/test-weighted validation 上会过拟合，线上推荐从 v14 的 `0.4942` 降到 `0.4923`。
- v18 已证明单 split 看似正收益的短历史增强和零历史 5-seed，在 5 split 下不稳。
- v19 应先升级验证协议，再做低风险、可解释的短历史计数/Bayes rerank。

当前 v19 主线：

```text
exact-length multi-split validation
+ pairwise delta vs v17
+ short-history count/Bayes tail rerank
+ top1 freeze
```

## 优先方向

推荐优先级：

```text
短历史 seq_len=1/3 > 零历史 seq_len=0 > 长历史 seq_len>=4
```

理由：

- 测试集中短历史用户最多：`seq_len=1` 为 `1003`，`seq_len=2` 为 `49`，`seq_len=3` 为 `4425`，合计 `5477/10000`。
- v17 的短历史 top1 不变、top10 弱 rerank 线上小幅正迁移 `+0.0003`，说明短历史可以轻触。
- v18 的零历史 5-seed MLP 不稳，说明继续加 seed/pool/rank weight 已接近饱和。
- `seq_len>=4` 仅 `1008/10000`，且现有中长历史分段 NDCG 已明显高于短段。

## v19 推荐模型

新增一个可解释短历史 reranker：

```text
v17 base prediction
+ last_item -> target count prior
+ last_item + user group -> target count prior
+ suffix2 + user group -> target count prior
+ Bayesian shrinkage
+ freeze top1
+ only rerank positions 2-10
```

候选类建议：

```python
class SegmentBayesShortRerankRecommender:
    """
    v19 candidate:
    - base = v17
    - only touches seq_len 1/3 by default
    - freezes top1
    - uses last-item / suffix / user-group count priors with Bayesian shrinkage
    """
```

核心预测逻辑：

```python
if raw_len not in target_lengths:
    return base_model.predict_row(row, k=10)

base_items = base_model.predict_row(row, k=base_pool)
frozen = base_items[:freeze_top_n]
prior_scores = count_prior_scores(uid, hist, row)
tail = rerank(base_items[freeze_top_n:], prior_scores)
return frozen + tail[: 10 - len(frozen)]
```

首个候选配置：

```python
def v19_short_count_gate_config():
    return {
        "name": "v19_short_count_gate_top1freeze",
        "model": "segment_bayes_short_rerank",
        "base_config": v17_conservative_shortseq_config(),
        "target_lengths": (1, 3),
        "freeze_top_n": 1,
        "base_pool": 20,
        "prior_pool": 30,
        "min_count": 20,
        "shrink_beta": 60,
        "alpha_by_len": {1: 0.035, 3: 0.050},
        "group_specs": (
            ("u_cat_01",),
            ("u_cat_01", "u_cat_02"),
            ("u_cat_01", "u_cat_06"),
        ),
        "suffix_orders": (1, 2),
        "min_top10_overlap": 9,
    }
```

## 验证协议升级

现有 `<=3` 粗 bin 会混合零历史和短历史，需要改成 exact-length：

```python
def exact_length_bin(raw_len: int) -> str:
    if raw_len == 0:
        return "0"
    if raw_len == 1:
        return "1"
    if raw_len == 2:
        return "2"
    if raw_len == 3:
        return "3"
    if raw_len <= 10:
        return "4-10"
    return ">10"
```

v19 验证必须显式对照 v17：

```text
for seed in [42, 43, 44, 45, 46, 47, 48]:
    split train into fit/val
    build v17 baseline
    build v19 candidate
    evaluate:
      A. natural exact-length val: 0/1/2/3/4-10/>10
      B. masked exact-length val
      C. test-length-weighted exact score
      D. pairwise delta vs v17 on identical rows
      E. row diff audit vs v17-style predictions
```

每个 split 至少记录：

```json
{
  "seed": 42,
  "weighted_exact_delta_vs_v17": 0.0,
  "natural_len_delta": {
    "0": 0.0,
    "1": 0.0,
    "2": 0.0,
    "3": 0.0,
    "4-10": 0.0,
    ">10": 0.0
  },
  "top1_changed_by_len": {},
  "changed_rows_by_len": {},
  "top10_overlap_hist": {},
  "bootstrap_ci95": [0.0, 0.0]
}
```

建议新增：

- `our_solution/src/validation.py`
- `our_solution/tools/v19_multisplit_eval.py`

## 候选顺序

| 候选 | 作用范围 | 说明 |
| --- | ---: | --- |
| `v19a_short_len3_only_alpha04` | 只改 `seq_len=3` | 最大短历史段，跳过 len1/2 |
| `v19b_short_len1_3_alpha035_050` | 改 `seq_len=1/3` | top1 冻结，tail rerank |
| `v19c_zero_bayes_shrink_alpha03` | 只改 `seq_len=0` | 新的贝叶斯平滑，不再加 MLP seed |
| `v19d_combo_b_plus_c` | `0/1/3` | 只有 b、c 单独都稳时才评估 |

不建议 v19 首轮改 `seq_len=2`。测试只有 `49` 个用户，收益上限低、验证噪声大。

## 提交门槛

v19 只有满足以下条件才生成提交包：

```text
1. 7 split mean exact-weighted delta vs v17 >= +0.0012
2. 7 split min exact-weighted delta vs v17 >= +0.0005
3. negative_split_count == 0
4. natural seq_len=1 and seq_len=3 delta 均值为正，min 不显著为负
5. seq_len=1/2/3 top1_changed == 0
6. seq_len=2 changed_rows == 0，除非单独验证极强
7. seq_len>=4 changed_rows == 0
8. A2 无重复 item，全部 item 来自 item.csv
9. 与 v17 比较，不与 v1/reference 比较
```

如果收益只有 `+0.0001` 到 `+0.0003`，不提交。

## 分类任务

v19 默认不动分类，继续复用：

```powershell
--reuse_a1 .\versions\v8\submission\A1.csv
```

分类只做离线 probe，不进默认提交。除非满足：

```text
changed rows <= 20 or <= 30
7 split 全正
mean acc delta >= +0.0015
min acc delta >= +0.0005
类别分布不大幅偏移
```

## 执行顺序

```text
Step 1: 实现 exact-length multi-split evaluator
Step 2: 复现 v17 在 7 split 下的 baseline 指标
Step 3: 跑 v19a_short_len3_only_alpha04
Step 4: 如果 v19a 过线，再跑 v19b_short_len1_3_alpha035_050
Step 5: 单独跑 v19c_zero_bayes_shrink_alpha03
Step 6: 只有 b、c 都稳，才跑 combo
Step 7: 分类默认复用 v8/v17 A1
Step 8: 只有全部门槛通过，才生成 v19 prediction.zip
```

推荐生成命令形态：

```powershell
python .\our_solution\run.py `
  --task 2 `
  --reuse_a1 .\versions\v8\submission\A1.csv `
  --budget 1 `
  --seed 42 `
  --version v19 `
  --note "v19 short-history Bayesian count rerank with exact-length multisplit validation"
```
