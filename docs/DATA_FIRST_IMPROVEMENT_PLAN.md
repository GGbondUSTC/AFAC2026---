# Data-First Improvement Plan

## Why This Is Now The Main Route

The current official best is `v24`:

- Total: `0.6276`
- Classification: `0.7590`
- Recommendation: `0.4962`
- Rank at report time: `27`

`v25` improved internal recommendation metrics slightly but tied `v24` online. The post-v25 grid also found only about `+0.00012` versus `v24`, which is too small to justify another submission. The next useful step is data analysis, not more local rerank micro-tuning.

## Current Hypothesis

The public score gap is more likely caused by validation/test distribution mismatch and missed segment-specific data signals than by a single global weight tweak.

Focus on recommendation first:

- Known leader recommendation reference: `0.50639`
- Current recommendation: `0.4962`
- Gap: `0.01019`

Classification is still worth probing, but only with small, high-confidence changes:

- Current classification: `0.7590`
- Known leader classification reference: `0.76518`
- Gap: `0.00618`

## Data To Audit

Recommendation:

```text
A推荐/train.csv
A推荐/test.csv
A推荐/user.csv
A推荐/item.csv
A推荐/metadata.json
```

Classification:

```text
A分类/A1.npz
A分类/sample_submission.csv
```

Generated predictions for comparison:

```text
versions/v24/submission/A2.csv
versions/v25/submission/A2.csv
versions/v24/submission/A1.csv
```

These versioned CSV files are kept locally but are still ignored by Git. Use them for local audits; do not rely on GitHub readers having all generated artifacts unless they reproduce them.

## Recommendation Audit Checklist

1. Compare `train.csv` validation rows against `test.csv` by raw sequence length:
   - exact lengths `0/1/2/3/4-10/11-20/21-30/31-80/>80`;
   - unique item count;
   - repeat ratio;
   - last-item frequency;
   - count concentration of the top historical item.

2. Compare user feature distributions:
   - each `u_cat_*` marginal distribution;
   - high-support pairs such as `u_cat_01+u_cat_02`, `u_cat_01+u_cat_06`, `u_cat_02+u_cat_06`;
   - segments where test support is high but training target support is sparse.

3. Compare item feature distributions:
   - target-side distribution in `train.csv` by `i_cat_01/i_cat_02/i_cat_03/i_bucket_01`;
   - historical item distribution in `test.csv`;
   - mismatch between historical item feature mix and current predicted item feature mix.

4. Audit `v24` predictions by segment:
   - top1, top3, top10 feature distribution;
   - share of recommendations already present in history;
   - history-frequency and recency rank of recommended items;
   - changed rows versus `v22/v25`.

5. Build validation splits that match test structure more closely:
   - exact-length weighting;
   - user-feature stratification;
   - last-item or suffix stratification for medium/long histories;
   - separate zero-history and short-history holdouts.

## Candidate Families To Explore After Audit

Only implement after a clear data diagnostic points to the segment.

- Medium/long history:
  - conditional rerank by `(last item, user group)` and item feature compatibility;
  - adjust history-repeat score based on repeat concentration;
  - guard `top1_changed == 0` until the validation signal is much stronger.

- Zero history:
  - user-feature cluster priors with stronger shrinkage diagnostics;
  - item feature calibration against test user segments;
  - avoid more neural ensemble seeds unless they change segment behavior, not just validation noise.

- Short history:
  - very conservative data-derived features only;
  - require exact-length multi-split evidence;
  - avoid changing `seq_len=2` unless isolated evidence is unusually strong.

- Classification:
  - graph neighborhood confidence audit around the 9 successful `v21` replacements;
  - per-class degree/feature cluster diagnostics;
  - only submit tiny A1 diffs with multi-split positive evidence.

## Submission Gate

Do not generate a new root `prediction.zip` unless the candidate has a real data-backed reason and passes stricter checks:

```text
recommendation mean delta vs v24 >= +0.0008
recommendation min split delta vs v24 > 0
no high-risk short-history negative split
top1_changed == 0 for short-history rows unless separately justified
A2 has 10000 rows and 10 unique valid items per row
A1 has 2751 rows if classification is touched
```

Tiny offline improvements like `v25` are useful for analysis but should not be submitted again without a new signal.

## First Concrete Work Items

1. Add a data audit script under `our_solution/tools/`.
2. Produce a JSON/Markdown segment report under `our_solution/output/` locally.
3. Identify the top 3 segments where `v24` is most likely under-calibrated.
4. Only then implement one candidate scoped to one segment.
5. Compare against `v24`, not just against `v17`.
