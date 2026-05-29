# Block-importance qualitative analysis

Source: `results/leakfree_block_probe.json` (GSM8K-train probe, n=50; naive-merge probe accuracy = 0.440).

## Headline finding

Of 32 transformer blocks in Llama-3.1-8B:

- **12 blocks have POSITIVE importance** (freezing them improves probe accuracy — they carry GSM8K capability that naive merging damages).
- **14 blocks have NEGATIVE importance** (freezing them HURTS probe accuracy — their delta is net-beneficial for GSM8K).
- **6 blocks have zero measurable effect** (at probe resolution n=50).

This is the structural fact behind the negation finding: more than 40% of blocks have deltas that *help* the protected task. Positive-coefficient mergers can't exploit this; they apply the same scalar to helpful and harmful directions alike.

## Depth pattern

| Depth third | Block range | Mean importance | # positive |
|---|---|---|---|
| Early | 0–9 | -0.026 | 3/10 |
| Middle | 10–19 | +0.032 | 7/10 |
| Late | 20–31 | -0.035 | 2/12 |

Middle blocks concentrate the GSM8K-critical capability (7/10 positive; mean importance +0.032). Early and late blocks lean negative — their deltas help on average. This matches the broader interpretability literature: middle layers encode task-relevant abstractions, boundary layers handle input/output adaptation that is more task-specific and may benefit from the adapter's reshaping.

## Top 5 protected blocks (most positive importance)

| Block | Freeze accuracy | Importance |
|---|---|---|
| 11 | 0.520 | +0.080 |
| 13 | 0.500 | +0.060 |
| 16 | 0.500 | +0.060 |
| 26 | 0.500 | +0.060 |
| 8 | 0.480 | +0.040 |

Freezing any of these blocks during merge yields +4 to +8pp on the probe.

## Top 5 helpful blocks (most negative importance)

| Block | Freeze accuracy | Importance |
|---|---|---|
| 30 | 0.280 | -0.160 |
| 0 | 0.300 | -0.140 |
| 29 | 0.320 | -0.120 |
| 4 | 0.360 | -0.080 |
| 21 | 0.380 | -0.060 |

Freezing block 30 *costs* 16pp on the probe — its delta is strongly beneficial. Blocks 0, 4, 29, 30, 31 cluster at the depth boundaries.

## Suggested paper text additions

Two short additions strengthen the paper's framing without claiming a new method:

1. **In Sec. 4 (Results) or App. C (Probing):** add a one-line stat — '14 of 32 blocks (44%) have negative probe importance, i.e., naive-merging them yields better GSM8K than freezing them. Positive-coefficient mergers (TA, TIES, DARE, MagMax) cannot selectively flip these blocks; this is the structural fact behind the negation finding in App. F.'

2. **In App. C, after the probe table:** add the depth pattern — 'Middle blocks concentrate the protected-task capability (7/10 positive in blocks 10–19, mean importance +0.032), while early and late blocks lean negative. The most strongly negative block (block 30) costs 16pp on the probe when frozen.'

