# Data-First Improvement Plan

## Why This Is Now The Main Route

The current official best is `v32`:

- Total: `0.6278`
- Classification: `0.7590`
- Recommendation: `0.4966`
- Rank at report time: not recorded

The current local root `prediction.zip` is the verified `v32` package. `v33` and `v34` were submitted after it as long-history LightGBM blend probes, but both scored below v32; the root package was restored to v32.

`v31` demonstrated that the current candidate pool has headroom: its seed-42 recall@80 was `0.963009` versus `0.903206` for the v29 reference. Its broad rerank nevertheless failed online. `v32` recovered a small official gain by using the same recall sources only for a conservative `>80` insert gate. The next useful step is to audit candidate quality and gate calibration, not to make another global or semi-free rerank.

## Current Hypothesis

The public score gap is more likely caused by validation/test distribution mismatch and missed segment-specific data signals than by a single global weight tweak.

Focus on recommendation first:

- Known leader recommendation reference: `0.50639`
- Current recommendation: `0.4966`
- Gap: `0.00979`

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
versions/v29/submission/A2.csv
versions/v32/submission/A2.csv
versions/v33/submission/A2.csv
versions/v34/submission/A2.csv
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

4. Audit `v32` predictions and the v31/v32 candidate set by segment:
   - top1, top3, top10 feature distribution;
   - share of recommendations already present in history;
   - history-frequency and recency rank of recommended items;
   - candidate-source recall and calibration for rows accepted or rejected by the v32 gate;
   - changed rows versus `v29/v32/v33/v34`.

5. Build validation splits that match test structure more closely:
   - exact-length weighting;
   - user-feature stratification;
   - last-item or suffix stratification for medium/long histories;
   - separate zero-history and short-history holdouts.

Current local tooling:

```powershell
python .\our_solution\tools\data_audit_rec_segments.py
```

This writes `our_solution/output/data_audit_rec_segments.json` locally and reports exact-length support coverage for last item, suffix2, user group, repeat ratio, and top historical item share.

## Candidate Families To Explore After Audit

Only implement after a clear data diagnostic points to the segment.

- Medium/long history:
  - analyze v31 candidate sources before changing a ranker: source, support, base-rank gap, and v32-gate acceptance should explain each proposed insertion;
  - test conditional calibration by `(last item, user group)` or item feature compatibility only on a supported segment;
  - preserve at least the v32 top5 for `>80` histories unless evidence specifically supports a higher-risk change;
  - v33/v34 show that even freezing top3 is insufficient protection for a semi-free tail blend.

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
recommendation mean delta vs v32 >= +0.0005 for a normal candidate
recommendation min split delta vs v32 >= 0
no high-risk short-history negative split
top1_changed == 0 for short-history rows; keep v32 top5 fixed for >80 rows unless separately justified
A2 has 10000 rows and 10 unique valid items per row
A1 has 2751 rows if classification is touched
```

Tiny offline improvements are useful for analysis but should not be submitted again without a new signal. v32 transferred because the change was narrow, segment-scoped, and preserved the strong base ranking; use that as the minimum bar for any small-delta submission.

## First Concrete Work Items

1. Done: add a data audit script under `our_solution/tools/`.
2. Done: implement and evaluate a high-recall candidate pool (`v31`) and a conservative gate (`v32`).
3. Next: measure candidate-source precision, rank-gap, and support for accepted versus rejected v32-gate rows.
4. Identify one supported subsegment, then implement one scoped calibration or candidate-source change.
5. Compare against `v32`, not just against `v29` or `v17`.
