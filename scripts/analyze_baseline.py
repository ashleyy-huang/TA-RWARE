"""Phase 0 baseline analysis — numeric only, no plotting.

Reads a merged run directory (output of `merge_runs.py`) and produces:

- `analysis/baseline_table.md`     — Plan §0-B baseline 表格 + heavy-tail summary
- `analysis/travel_time_stats.json`— heavy-tail 量化指標（mean/std/percentiles/ratios/Hill index）
- `analysis/travel_times.npy`      — flattened raw delivery travel times (for downstream plotting)
- `analysis/travel_times_ccdf.csv` — CCDF table (x, P(X >= x)) for log-log inspection

The script is purely numerical (numpy only); plotting is a separate concern.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_rows(run_dir: Path):
    jsonl = run_dir / "episodes.jsonl"
    if not jsonl.exists():
        sys.exit(f"[analyze_baseline] missing {jsonl}")
    rows = []
    with open(jsonl) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"[analyze_baseline] {jsonl}:{ln} invalid JSON: {e}")
    return rows


def percentile_summary(arr, name):
    arr = np.asarray(arr, dtype=float)
    return {
        "metric": name,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def hill_estimator(x, frac):
    """Hill tail-index estimator on the top `frac` of values.

    Returns alpha_hat such that P(X > x) ~ x^{-alpha} for the tail.
    Smaller alpha → heavier tail. alpha < 2 means infinite variance.
    """
    x = np.sort(np.asarray(x, dtype=float))[::-1]  # descending
    k = max(2, int(len(x) * frac))
    if k >= len(x):
        return None
    top = x[:k]
    threshold = x[k]
    if threshold <= 0:
        return None
    log_ratios = np.log(top / threshold)
    mean_log_ratio = float(np.mean(log_ratios))
    if mean_log_ratio <= 0:
        return None
    return {
        "frac": frac,
        "k": k,
        "threshold": float(threshold),
        "alpha_hat": 1.0 / mean_log_ratio,
    }


def heavy_tail_stats(travel_times):
    x = np.asarray(travel_times, dtype=float)
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    centered = (x - mu)
    # Excess kurtosis = E[(X-mu)^4]/sigma^4 - 3 (0 for Gaussian).
    skewness = float(np.mean(centered ** 3) / (sigma ** 3)) if sigma > 0 else None
    kurtosis = float(np.mean(centered ** 4) / (sigma ** 4) - 3.0) if sigma > 0 else None
    p50 = float(np.percentile(x, 50))
    p95 = float(np.percentile(x, 95))
    p99 = float(np.percentile(x, 99))
    p995 = float(np.percentile(x, 99.5))
    return {
        "count": int(x.size),
        "mean": mu,
        "std": sigma,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "p99.5": p995,
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "p95_over_p50": p95 / p50 if p50 > 0 else None,
        "p99_over_p50": p99 / p50 if p50 > 0 else None,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "hill_top5pct": hill_estimator(x, 0.05),
        "hill_top1pct": hill_estimator(x, 0.01),
    }


def write_ccdf_csv(x, out_path, n_points=200):
    """Write a downsampled CCDF table: log-spaced quantiles of x → P(X >= x)."""
    x_sorted = np.sort(np.asarray(x, dtype=float))
    n = len(x_sorted)
    # log-spaced indices from 1 to n
    idxs = np.unique(
        np.round(np.geomspace(1, n, num=n_points)).astype(int)
    )
    idxs = np.clip(idxs, 1, n) - 1  # 0-based
    rows = []
    for i in idxs:
        v = x_sorted[i]
        # P(X >= v) ≈ (n - i) / n  with i 為當前小於等於 v 的 index
        rank_from_top = n - i
        rows.append((v, rank_from_top / n))
    with open(out_path, "w") as f:
        f.write("x,ccdf\n")
        for v, p in rows:
            f.write(f"{v},{p}\n")


def render_baseline_table(env_id, n_episodes, per_episode_summaries, tau_clash,
                          tau_stuck, bad_threshold_pct, travel_summary, hill5, hill1):
    lines = []
    lines.append(f"# Phase 0 Baseline — heuristic CTA")
    lines.append("")
    lines.append(f"- env_id: `{env_id}`")
    lines.append(f"- episodes: {n_episodes}")
    lines.append(f"- bad_episode threshold: P{bad_threshold_pct} of CTA's own distribution")
    lines.append("")
    lines.append("## Per-episode distribution")
    lines.append("")
    lines.append("| Metric | Mean | Std | P50 | P95 | P99 | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in per_episode_summaries:
        lines.append(
            f"| {s['metric']} | {s['mean']:.3f} | {s['std']:.3f} | "
            f"{s['p50']:.3f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")
    lines.append("## Bad-episode threshold (for future RL comparison)")
    lines.append("")
    lines.append(
        f"- τ_clash = P{bad_threshold_pct}(total_clashes_CTA) = **{tau_clash:.0f}**"
    )
    lines.append(
        f"- τ_stuck = P{bad_threshold_pct}(total_stuck_CTA) = **{tau_stuck:.0f}**"
    )
    lines.append(
        f"- CTA bad_episode_ratio @ (clashes > τ_clash) = "
        f"**{(100 - bad_threshold_pct) / 100:.2f}** by construction"
    )
    lines.append("")
    lines.append("## Travel time distribution (heavy-tail profiling)")
    lines.append("")
    lines.append(f"- count: {travel_summary['count']}")
    lines.append(f"- mean: {travel_summary['mean']:.3f}")
    lines.append(f"- std:  {travel_summary['std']:.3f}")
    lines.append(f"- p50:  {travel_summary['p50']:.1f}")
    lines.append(f"- p95:  {travel_summary['p95']:.1f}")
    lines.append(f"- p99:  {travel_summary['p99']:.1f}")
    lines.append(f"- p99.5: {travel_summary['p99.5']:.1f}")
    lines.append(f"- max:  {travel_summary['max']:.1f}")
    lines.append("")
    lines.append("### Heavy-tail indicators")
    lines.append("")
    lines.append(f"- p95 / p50 ratio: **{travel_summary['p95_over_p50']:.2f}** "
                 "(Gaussian ~1.96; 越大越尖)")
    lines.append(f"- p99 / p50 ratio: **{travel_summary['p99_over_p50']:.2f}** "
                 "(Gaussian ~2.58)")
    lines.append(f"- skewness: **{travel_summary['skewness']:.3f}** "
                 "(Gaussian = 0; 正值代表右尾長)")
    lines.append(f"- excess kurtosis: **{travel_summary['excess_kurtosis']:.3f}** "
                 "(Gaussian = 0; >3 代表 heavy tail)")
    if hill5 is not None:
        lines.append(
            f"- Hill α @ top 5% (threshold={hill5['threshold']:.1f}, k={hill5['k']}): "
            f"**{hill5['alpha_hat']:.3f}** "
            "(power-law exponent；越小尾巴越重；α<2 → 變異數無界)"
        )
    if hill1 is not None:
        lines.append(
            f"- Hill α @ top 1% (threshold={hill1['threshold']:.1f}, k={hill1['k']}): "
            f"**{hill1['alpha_hat']:.3f}**"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Phase 0 baseline numeric analysis.")
    ap.add_argument("--run_dir", required=True, type=Path,
                    help="Merged run dir produced by merge_runs.py")
    ap.add_argument("--bad_threshold_pct", default=95, type=int,
                    help="Percentile of CTA's clash distribution for τ definition")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    rows = load_rows(run_dir)
    cfg_path = run_dir / "merged_config.json"
    env_id = "unknown"
    if cfg_path.exists():
        env_id = json.loads(cfg_path.read_text()).get("env_id", "unknown")

    # Per-episode summaries.
    pick_rate = [r["pick_rate"] for r in rows]
    total_clashes = [r["total_clashes"] for r in rows]
    total_stuck = [r["total_stuck"] for r in rows]
    episode_length = [r["episode_length"] for r in rows]
    global_return = [r["global_return"] for r in rows]
    per_ep = [
        percentile_summary(pick_rate, "pick_rate"),
        percentile_summary(total_clashes, "total_clashes"),
        percentile_summary(total_stuck, "total_stuck"),
        percentile_summary(episode_length, "episode_length"),
        percentile_summary(global_return, "global_return"),
    ]

    # Bad-episode thresholds.
    tau_clash = float(np.percentile(total_clashes, args.bad_threshold_pct))
    tau_stuck = float(np.percentile(total_stuck, args.bad_threshold_pct))

    # Heavy-tail analysis on flattened delivery travel times.
    travel_times = [t for r in rows for t in r.get("delivery_travel_times", [])]
    if not travel_times:
        sys.exit("[analyze_baseline] no delivery_travel_times found in jsonl rows. "
                 "The run is likely pre-schema-fix; rerun heuristic to populate.")
    travel_arr = np.asarray(travel_times, dtype=np.int32)
    travel_summary = heavy_tail_stats(travel_arr)
    hill5 = travel_summary["hill_top5pct"]
    hill1 = travel_summary["hill_top1pct"]

    # Write artifacts.
    np.save(out_dir / "travel_times.npy", travel_arr)
    write_ccdf_csv(travel_arr, out_dir / "travel_times_ccdf.csv")

    stats_payload = {
        "env_id": env_id,
        "n_episodes": len(rows),
        "bad_threshold_pct": args.bad_threshold_pct,
        "tau_clash": tau_clash,
        "tau_stuck": tau_stuck,
        "per_episode": per_ep,
        "travel_time": travel_summary,
    }
    (out_dir / "travel_time_stats.json").write_text(
        json.dumps(stats_payload, indent=2)
    )

    md = render_baseline_table(
        env_id, len(rows), per_ep, tau_clash, tau_stuck,
        args.bad_threshold_pct, travel_summary, hill5, hill1
    )
    (out_dir / "baseline_table.md").write_text(md)

    print(f"[analyze_baseline] wrote {out_dir / 'baseline_table.md'}")
    print(f"[analyze_baseline] wrote {out_dir / 'travel_time_stats.json'}")
    print(f"[analyze_baseline] wrote {out_dir / 'travel_times.npy'} "
          f"({travel_arr.size} ints)")
    print(f"[analyze_baseline] wrote {out_dir / 'travel_times_ccdf.csv'}")
    print()
    print("=" * 60)
    print(md)


if __name__ == "__main__":
    main()
