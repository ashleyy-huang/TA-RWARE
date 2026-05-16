"""Merge parallel heuristic runs into a single run dir (+ optional W&B upload).

Reads each source run's `config.json` and `episodes.jsonl`, validates that the
shards form a clean non-overlapping partition of the episode index space, and
writes a sorted `episodes.jsonl` + `merged_config.json` to a new directory.

Source runs are never modified.
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np


def fail(msg: str) -> None:
    print(f"[merge_runs] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    jsonl_path = run_dir / "episodes.jsonl"
    if not cfg_path.exists():
        fail(f"missing config.json in {run_dir}")
    if not jsonl_path.exists():
        fail(f"missing episodes.jsonl in {run_dir}")
    config = json.loads(cfg_path.read_text())
    rows = []
    with open(jsonl_path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"{jsonl_path}:{ln} invalid JSON: {e}")
    return config, rows


def auto_output_name(first_run_basename: str) -> str:
    # Strip trailing "_{seed}_{mmdd}_{hhmm}" → keep the algo+env stem.
    stem = re.sub(r"_\d+_\d{4}_\d{4}$", "", first_run_basename)
    return f"{stem}_merged_{time.strftime('%m%d_%H%M')}"


def compute_global_metrics(rows):
    out = {}
    all_tt = []
    for r in rows:
        tt = r.get("delivery_travel_times")
        if tt:
            all_tt.extend(tt)
    if all_tt:
        out["travel_time_mean"] = float(np.mean(all_tt))
        out["travel_time_std"] = float(np.std(all_tt))
        out["travel_time_p50"] = float(np.percentile(all_tt, 50))
        out["travel_time_p95"] = float(np.percentile(all_tt, 95))
        out["travel_time_max"] = float(np.max(all_tt))
        out["travel_time_min"] = float(np.min(all_tt))
        out["travel_time_count"] = len(all_tt)

    for field, alias in [("pick_rate", "pick_rate_mean"),
                         ("clash_rate", "clash_rate_mean"),
                         ("stuck_rate", "stuck_rate_mean")]:
        vals = [r[field] for r in rows if field in r]
        if vals:
            out[alias] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge parallel TA-RWARE runs.")
    ap.add_argument("--run_glob", required=True,
                    help="Glob pattern (relative to cwd) for run directories.")
    ap.add_argument("--output_name", default=None)
    ap.add_argument("--output_dir", default="runs")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite non-empty output dir if it already exists.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="tarware")
    args = ap.parse_args()

    matched = sorted(Path(p) for p in glob(args.run_glob))
    matched = [p for p in matched if p.is_dir()]
    if not matched:
        fail(f"no directories matched --run_glob={args.run_glob!r}")
    print(f"[merge_runs] matched {len(matched)} run dir(s):")
    for p in matched:
        print(f"  - {p}")

    configs, all_rows_by_run, source_names = [], [], []
    # [(seed_start, seed_end_exclusive_actual, basename, actual_rows, nominal_count)]
    # seed_end_exclusive uses ACTUAL row count so partial runs don't over-claim
    # coverage and don't trigger false-positive overlap fails against later seeds.
    seed_intervals = []
    for p in matched:
        cfg, rows = load_run(p)
        configs.append(cfg)
        all_rows_by_run.append(rows)
        source_names.append(p.name)
        seed = int(cfg.get("seed", 0))
        nominal = int(cfg.get("num_episodes", len(rows)))
        actual = len(rows)
        if actual != nominal:
            print(f"[merge_runs] WARN: {p.name} has {actual} rows but "
                  f"config.num_episodes={nominal} — using actual count.")
        seed_intervals.append((seed, seed + actual, p.name, actual, nominal))

    # Validate env_id / algo consistency.
    env_ids = {c.get("env_id") for c in configs}
    algos = {c.get("algo") for c in configs}
    if len(env_ids) != 1:
        fail(f"env_id mismatch across runs: {env_ids}")
    if len(algos) != 1:
        fail(f"algo mismatch across runs: {algos}")
    env_id = next(iter(env_ids))
    algo = next(iter(algos))

    # Validate non-overlapping seed intervals (using actual produced ranges).
    ordered = sorted(seed_intervals, key=lambda x: x[0])
    for (s1, e1, n1, *_), (s2, e2, n2, *_) in zip(ordered, ordered[1:]):
        if s2 < e1:
            fail(f"seed intervals overlap: {n1} [{s1},{e1}) vs {n2} [{s2},{e2})")

    # Validate global episode index uniqueness.
    flat_rows = [r for rs in all_rows_by_run for r in rs]
    ep_indices = [r["episode"] for r in flat_rows]
    if len(ep_indices) != len(set(ep_indices)):
        # Find the dupes for a helpful message.
        seen, dup = set(), set()
        for e in ep_indices:
            (dup if e in seen else seen).add(e)
        fail(f"duplicate episode indices across runs: {sorted(dup)[:10]}...")

    flat_rows.sort(key=lambda r: r["episode"])
    total_episodes = len(flat_rows)

    # Resolve output dir.
    out_name = args.output_name or auto_output_name(matched[0].name)
    out_dir = Path(args.output_dir) / out_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        fail(f"output dir {out_dir} exists and is non-empty (use --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write merged jsonl.
    out_jsonl = out_dir / "episodes.jsonl"
    with open(out_jsonl, "w") as f:
        for r in flat_rows:
            f.write(json.dumps(r) + "\n")

    global_metrics = compute_global_metrics(flat_rows)

    merged_config = {
        "env_id": env_id,
        "algo": algo,
        "sources": source_names,
        "seed_ranges": [[s, e] for s, e, *_ in ordered],
        "source_actual_counts": {n: actual for _, _, n, actual, _ in ordered},
        "source_nominal_counts": {n: nominal for _, _, n, _, nominal in ordered},
        "total_episodes": total_episodes,
        "merged_at": dt.datetime.now().isoformat(timespec="seconds"),
        "global_metrics": global_metrics,
    }
    (out_dir / "merged_config.json").write_text(json.dumps(merged_config, indent=2))

    wandb_url = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=out_name,
            group=f"merged__{env_id}",
            tags=["merged", algo, env_id],
            config=merged_config,
        )
        for r in flat_rows:
            wandb.log(r, step=r["episode"])
        # Final summary step so global_metrics show up as W&B values too.
        wandb.log({f"global/{k}": v for k, v in global_metrics.items()})
        for k, v in global_metrics.items():
            wandb.run.summary[f"global/{k}"] = v
        wandb_url = run.url if run is not None else None
        wandb.finish()

    seed_lo = min(s for s, *_ in ordered)
    seed_hi = max(e for _, e, *_ in ordered)
    span = seed_hi - seed_lo
    print()
    print(f"[merge_runs] merged {len(matched)} run(s) → {out_dir}")
    print(f"             episodes: {total_episodes}")
    print(f"             seed coverage: [{seed_lo}, {seed_hi}) "
          f"(span {span}, rows {total_episodes})")
    if span != total_episodes:
        print(f"             [warn] {span - total_episodes} episode(s) missing "
              f"within span — partial run(s) or gap")
    if global_metrics:
        keys = ", ".join(f"{k}={global_metrics[k]:.3f}"
                         for k in ("pick_rate_mean", "travel_time_p95")
                         if k in global_metrics)
        if keys:
            print(f"             global: {keys}")
    if wandb_url:
        print(f"             wandb: {wandb_url}")


if __name__ == "__main__":
    main()
