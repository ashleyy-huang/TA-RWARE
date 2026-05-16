# Phase 0 Baseline — heuristic CTA

- env_id: `tarware-medium-8agvs-4pickers-partialobs-v1`
- episodes: 10000
- bad_episode threshold: P95 of CTA's own distribution

## Per-episode distribution

| Metric | Mean | Std | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| pick_rate | 54.292 | 6.262 | 54.720 | 61.920 | 64.800 | 72.000 |
| total_clashes | 42.676 | 54.942 | 35.000 | 54.000 | 393.010 | 1437.000 |
| total_stuck | 2.686 | 8.948 | 1.000 | 5.000 | 48.000 | 172.000 |
| episode_length | 500.000 | 0.000 | 500.000 | 500.000 | 500.000 | 500.000 |
| global_return | 39.040 | 5.189 | 39.500 | 45.600 | 47.900 | 53.500 |

## Bad-episode threshold (for future RL comparison)

- τ_clash = P95(total_clashes_CTA) = **54**
- τ_stuck = P95(total_stuck_CTA) = **5**
- CTA bad_episode_ratio @ (clashes > τ_clash) = **0.05** by construction

## Travel time distribution (heavy-tail profiling)

- count: 377025
- mean: 58.703
- std:  24.957
- p50:  53.0
- p95:  107.0
- p99:  134.0
- p99.5: 145.0
- max:  450.0

### Heavy-tail indicators

- p95 / p50 ratio: **2.02** (Gaussian ~1.96; 越大越尖)
- p99 / p50 ratio: **2.53** (Gaussian ~2.58)
- skewness: **1.152** (Gaussian = 0; 正值代表右尾長)
- excess kurtosis: **2.290** (Gaussian = 0; >3 代表 heavy tail)
- Hill α @ top 5% (threshold=107.0, k=18851): **7.035** (power-law exponent；越小尾巴越重；α<2 → 變異數無界)
- Hill α @ top 1% (threshold=134.0, k=3770): **9.832**
