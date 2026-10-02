# Tier B comparison

| mode | n | success | Wilson 95% | mean attempts |
|---|---:|---:|---|---:|
| k1_default | 100 | 0.000 | [0.000, 0.037] | 1.00 |
| k5_random | 100 | 0.600 | [0.502, 0.691] | 4.05 |
| k5_llm | 100 | 0.820 | [0.733, 0.883] | 3.10 |

Tier B proves in-context benefit only if `k5_llm` beats `k5_random` on the same seeds (non-overlapping Wilson intervals or clear gap).
Start theta is intentionally suboptimal (`TIER_B_START_THETA`) because production Tier A defaults already saturate near 100%.
