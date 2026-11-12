# Tier B comparison

| mode | n | success | Wilson 95% | mean attempts |
|---|---:|---:|---|---:|
| k1_default | 10 | 0.500 | [0.237, 0.763] | 1.00 |

Tier B proves in-context benefit only if `k5_llm` beats `k5_random` on the same seeds (non-overlapping Wilson intervals or clear gap).
Start θ is intentionally suboptimal (`TIER_B_START_THETA`) because production Tier A defaults already saturate near 100%.
