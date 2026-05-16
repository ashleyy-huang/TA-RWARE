"""Phase 0 baseline heavy-tail visualization.

Reads the outputs of `analyze_baseline.py` and produces 5 PNG figures:

1. travel_time_hist_linear.png       — travel time histogram (linear y)
2. travel_time_hist_logy.png         — travel time histogram (log y)
3. travel_time_ccdf_loglog.png       — travel time CCDF on log-log
4. clash_ccdf_loglog.png             — per-episode total_clashes CCDF on log-log
5. travel_time_qq_exponential.png    — Q-Q plot vs Exponential distribution

Optional W&B upload (`--wandb_resume <run_id>`) attaches the figures as
`wandb.Image` to an existing run so they sit alongside the original
merged metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl_field(jsonl: Path, key: str):
    out = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line)[key])
    return np.asarray(out)


def annotate_percentiles(ax, percentiles, color="firebrick", alpha=0.75):
    ymax = ax.get_ylim()[1]
    for x, label in percentiles:
        ax.axvline(x, ls="--", lw=1.0, color=color, alpha=alpha)
        ax.text(x, ymax, f" {label}", rotation=90, va="top", ha="left",
                color=color, fontsize=9)


def fig_hist_linear(tt, p50, p95, p99, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tt, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("delivery travel time (env steps)")
    ax.set_ylabel("count")
    ax.set_title(f"Travel time distribution (linear, N={len(tt)})")
    annotate_percentiles(ax, [(p50, "p50"), (p95, "p95"), (p99, "p99")])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_hist_logy(tt, p50, p95, p99, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tt, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("delivery travel time (env steps)")
    ax.set_ylabel("count (log)")
    ax.set_title("Travel time distribution (log-y)\n"
                 "straight line on log-y ⇒ exponential decay")
    annotate_percentiles(ax, [(p50, "p50"), (p95, "p95"), (p99, "p99")])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_ccdf_travel(tt, stats, out_path):
    x_sorted = np.sort(tt)
    n = len(x_sorted)
    ccdf = (n - np.arange(n)) / n
    p50, p95, p99 = stats["p50"], stats["p95"], stats["p99"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x_sorted, ccdf, color="steelblue", lw=1.4, label="empirical CCDF")
    annotate_percentiles(ax, [(p50, "p50"), (p95, "p95"), (p99, "p99")],
                         color="firebrick", alpha=0.55)
    hill5 = stats.get("hill_top5pct")
    if hill5:
        alpha_hat = hill5["alpha_hat"]
        thr = hill5["threshold"]
        x_fit = np.geomspace(thr, x_sorted.max(), 60)
        y_fit = 0.05 * (x_fit / thr) ** (-alpha_hat)
        ax.loglog(x_fit, y_fit, ls=":", color="darkorange", lw=2.2,
                  label=f"Hill fit α={alpha_hat:.2f} (top 5%)")
    ax.set_xlabel("x = delivery travel time")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title("Travel time CCDF (log-log)\n"
                 "straight line ⇒ power-law tail; α<2 ⇒ infinite variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_ccdf_clash(clashes, out_path):
    c = clashes[clashes > 0]
    c_sorted = np.sort(c)
    n_c = len(c_sorted)
    ccdf = (n_c - np.arange(n_c)) / n_c
    p50 = float(np.percentile(clashes, 50))
    p95 = float(np.percentile(clashes, 95))
    p99 = float(np.percentile(clashes, 99))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(c_sorted, ccdf, color="indianred", lw=1.4)
    annotate_percentiles(ax, [(p50, "p50"), (p95, "p95"), (p99, "p99")],
                         color="black", alpha=0.55)
    ax.set_xlabel("c = total clashes per episode")
    ax.set_ylabel("P(C ≥ c)")
    ax.set_title(f"Per-episode clash count CCDF (log-log, N={len(clashes)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def fig_qq_exponential(tt, mu, out_path):
    n = len(tt)
    plot_pos = (np.arange(1, n + 1) - 0.5) / n
    theoretical = -mu * np.log(1.0 - plot_pos)
    empirical = np.sort(tt)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(theoretical, empirical, ".", alpha=0.35, ms=2.5,
            color="steelblue", label="empirical")
    lim = float(max(theoretical.max(), empirical.max()))
    ax.plot([0, lim], [0, lim], "--", color="firebrick", lw=1.3,
            label=f"y = x (Exp(λ=1/{mu:.1f}))")
    ax.set_xlabel(f"theoretical quantile (Exponential, mean={mu:.1f})")
    ax.set_ylabel("empirical quantile (travel time)")
    ax.set_title("Q-Q plot vs Exponential\n"
                 "above red line = heavier than exponential tail")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def upload_to_wandb(figs_dir, run_id, project, stats):
    import wandb
    run = wandb.init(id=run_id, resume="must", project=project)
    fig_keys = {
        "travel_time_hist_linear": "phase0/travel_time/hist_linear",
        "travel_time_hist_logy": "phase0/travel_time/hist_logy",
        "travel_time_ccdf_loglog": "phase0/travel_time/ccdf_loglog",
        "clash_ccdf_loglog": "phase0/clash/ccdf_loglog",
        "travel_time_qq_exponential": "phase0/travel_time/qq_exponential",
    }
    log_payload = {
        k: wandb.Image(str(figs_dir / f"{name}.png"))
        for name, k in fig_keys.items()
    }
    for k, v in stats["travel_time"].items():
        if isinstance(v, (int, float)):
            log_payload[f"phase0/heavy_tail/{k}"] = v
            run.summary[f"phase0/heavy_tail/{k}"] = v
    for hill_key in ("hill_top5pct", "hill_top1pct"):
        h = stats["travel_time"].get(hill_key)
        if h:
            run.summary[f"phase0/heavy_tail/{hill_key}_alpha"] = h["alpha_hat"]
    run.summary["phase0/bad_threshold/tau_clash"] = stats["tau_clash"]
    run.summary["phase0/bad_threshold/tau_stuck"] = stats["tau_stuck"]
    wandb.log(log_payload)
    wandb.finish()
    print(f"[plot_baseline] uploaded 5 figures + heavy-tail stats to wandb run {run_id}")


def main():
    ap = argparse.ArgumentParser(description="Phase 0 baseline heavy-tail visualization.")
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--wandb_resume", default=None,
                    help="Existing W&B run id to attach figures to (resume='must')")
    ap.add_argument("--wandb_project", default="tarware")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    figs_dir = analysis_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    tt_path = analysis_dir / "travel_times.npy"
    stats_path = analysis_dir / "travel_time_stats.json"
    if not tt_path.exists() or not stats_path.exists():
        sys.exit("[plot_baseline] missing analyze_baseline outputs — "
                 "run analyze_baseline.py first")

    tt = np.load(tt_path)
    stats = json.loads(stats_path.read_text())
    travel_stats = stats["travel_time"]
    clashes = load_jsonl_field(run_dir / "episodes.jsonl", "total_clashes")

    fig_hist_linear(tt, travel_stats["p50"], travel_stats["p95"],
                    travel_stats["p99"], figs_dir / "travel_time_hist_linear.png")
    fig_hist_logy(tt, travel_stats["p50"], travel_stats["p95"],
                  travel_stats["p99"], figs_dir / "travel_time_hist_logy.png")
    fig_ccdf_travel(tt, travel_stats, figs_dir / "travel_time_ccdf_loglog.png")
    fig_ccdf_clash(clashes, figs_dir / "clash_ccdf_loglog.png")
    fig_qq_exponential(tt, travel_stats["mean"],
                       figs_dir / "travel_time_qq_exponential.png")

    print(f"[plot_baseline] wrote 5 figures to {figs_dir}")
    for p in sorted(figs_dir.glob("*.png")):
        print(f"  - {p.name}")

    if args.wandb_resume:
        upload_to_wandb(figs_dir, args.wandb_resume, args.wandb_project, stats)


if __name__ == "__main__":
    main()
