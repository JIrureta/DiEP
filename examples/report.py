"""Figures and ANALYSIS.md for the DiEP study, against the GA and DPMA.

    python examples/report.py --results results \
        --ga-harness ../GA/MoE_compression_GA/results/harness --dpma-results ../DPMA/results

Every benchmark JSON in this study -- GA, DPMA and DiEP -- has the same schema
and was produced with the same calibration tokens and the same lm-eval
protocol, so one loader reads them all.

Colour: each baseline keeps the colour it has in the GA and DPMA figures
(colour follows the method). The eight categorical slots are all taken by
those methods, and no free hue passes the palette validator against the
baselines drawn here, so DiEP -- the method under study -- is drawn in ink
with a star marker and direct labels: it is identified by lightness, marker
and label, never by hue alone. Its variants share the ink and differ by dash
pattern and marker.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import torch                              # noqa: E402

MC_TASKS = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu",
            "openbookqa", "piqa", "winogrande"]
CHANCE = {"arc_challenge": .25, "arc_easy": .25, "boolq": .5, "hellaswag": .25,
          "mmlu": .25, "openbookqa": .25, "piqa": .5, "winogrande": .5}
MC_CHANCE = sum(CHANCE.values()) / len(CHANCE)
KEEPS = [0.75, 0.5, 0.25]

# The project's colours (validated, all-pairs, light mode) and the neutrals.
COLORS = {"dpma": "#2a78d6", "frequency": "#eb6834", "random": "#1baf7a",
          "ga-perplexity-100-gens": "#4a3aa7"}
INK, MUTED, GRID, DENSE = "#1d1d1b", "#52514e", "#e1e6ec", "#8a8f98"
LABELS = {"dpma-v3": "DPMA v3 (soft export)", "dpma-v3-hard": "DPMA v3 (hard export)",
          "ga-perplexity-100-gens": "GA (best arm)", "frequency": "frequency",
          "random": "random (5 draws, min–max)"}

#: The DiEP run this report is about (``--primary``); its variants are
#: ``<primary>-uniform``, ``-merge``, ``-skip``. ``--previous`` names an earlier
#: DiEP run drawn next to it for comparison (e.g. the 20-step search).
PRIMARY, PREVIOUS = "diep", None


def configure(primary, primary_label, previous=None, previous_label=None):
    global PRIMARY, PREVIOUS
    PRIMARY, PREVIOUS = primary, previous
    LABELS[primary] = primary_label
    LABELS[primary + "-uniform"] = "DiEP, uniform per layer"
    LABELS[primary + "-merge"] = "DiEP + merge (§8)"
    LABELS[primary + "-skip"] = "DiEP + adaptive skip (§7)"
    if previous:
        LABELS[previous] = previous_label


configure("diep", "DiEP (mean of 3 seeds)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--ga-harness", default="../GA/MoE_compression_GA/results/harness")
    parser.add_argument("--dpma-results", default="../DPMA/results")
    parser.add_argument("--primary", default="diep", help="the DiEP method label to report on")
    parser.add_argument("--primary-label", default="DiEP (mean of 3 seeds)")
    parser.add_argument("--previous", default=None,
                        help="an earlier DiEP label to compare against (e.g. diep)")
    parser.add_argument("--previous-label", default="DiEP, earlier run")
    parser.add_argument("--previous-results", default=None,
                        help="that run's results folder (its harness and search logs)")
    return parser.parse_args()


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_harness(*directories) -> dict:
    """(method, requested keep ratio) -> list of records."""
    grouped: dict = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*-1shot.json")):
            if path.name.endswith(".provenance.json"):
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if record.get("mean_mc") is None:
                continue
            key = (record["method"], round(float(record["requested_keep_ratio"]), 3))
            grouped.setdefault(key, []).append(record)
    return grouped


def mean_mc(grouped, method, keep):
    values = [r["mean_mc"] for r in grouped.get((method, keep), [])]
    return (statistics.mean(values), values) if values else (None, [])


def task_value(record, task):
    return record["tasks"].get(task, {}).get("headline", {}).get("value")


def load_searches(runs: Path) -> dict:
    out = {}
    for scores in sorted(runs.glob("seed*/scores.pt")):
        try:     # a search still being written must not take the whole report down
            out[scores.parent.name] = torch.load(scores, map_location="cpu", weights_only=False)
        except (EOFError, RuntimeError, OSError):
            continue
    return out


def load_merge_reports(runs: Path) -> dict:
    """keep -> the merge report written by ``diep prune --merge`` (seed 0)."""
    out = {}
    for path in sorted(runs.glob("seed0/pruned-r*-merge/prune_card.json")):
        card = json.loads(path.read_text())
        if card.get("merge_report"):
            out[round(1 - float(card["sparsity"]), 3)] = card["merge_report"]
    return out


def load_bench(runs: Path) -> dict:
    out = {}
    for path in sorted(runs.glob("seed0/bench-r*.json")):
        sparsity = float(path.stem.removeprefix("bench-r"))
        try:
            out[round(1 - sparsity, 3)] = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.grid": True, "grid.linewidth": .7, "grid.color": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cdd5df",
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 2.0, "lines.markersize": 8,
    })


def _reference_lines(ax, grouped):
    dense, _ = mean_mc(grouped, "dense", 1.0)
    if dense is not None:
        ax.axhline(dense, ls=(0, (6, 3)), lw=1.4, color=DENSE)
        ax.annotate(f"dense {dense:.3f}", (0.77, dense), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8.5, color=MUTED)
    ax.axhline(MC_CHANCE, ls=":", lw=1.4, color=DENSE)
    ax.annotate(f"chance {MC_CHANCE:.3f}", (0.77, MC_CHANCE), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)


def _random_band(ax, grouped):
    lo, hi, xs = [], [], []
    for keep in KEEPS:
        values = [r["mean_mc"] for r in grouped.get(("random", keep), [])]
        if values:
            xs.append(keep); lo.append(min(values)); hi.append(max(values))
    if xs:
        ax.fill_between(xs, lo, hi, color=COLORS["random"], alpha=.22, lw=0,
                        label=LABELS["random"])


def _line(ax, grouped, method, *, color, marker, ls="-", mfc=None, lw=2.0, label=None,
          seed_bars=False, direct=False, bar_offset=0.0):
    points = [(k, mean_mc(grouped, method, k)) for k in KEEPS]
    points = [(k, m, v) for k, (m, v) in points if m is not None]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    ax.plot(xs, ys, ls=ls, lw=lw, color=color, marker=marker, ms=8 if marker != "*" else 12,
            mfc=mfc or color, mec=color, mew=1.4, label=label or LABELS.get(method, method),
            zorder=4)
    if seed_bars:
        # min-max over seeds, wherever a method has more than one run; offset a
        # little so two methods' bars at the same ratio do not sit on each other
        for k, m, v in points:
            if len(v) > 1:
                x = k + bar_offset
                ax.vlines(x, min(v), max(v), color=color, lw=2.4, zorder=5)
                ax.hlines([min(v), max(v)], x - .008, x + .008, color=color, lw=1.6, zorder=5)
    if direct:
        k, m, _ = points[0]
        ax.annotate(LABELS.get(method, method).split(" (")[0], (k, m), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")


def fig_comparison(grouped, outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax in (a, b):
        _random_band(ax, grouped)
        _reference_lines(ax, grouped)

    _line(a, grouped, "frequency", color=COLORS["frequency"], marker="o")
    _line(a, grouped, "ga-perplexity-100-gens", color=COLORS["ga-perplexity-100-gens"], marker="s")
    _line(a, grouped, "dpma-v3-hard", color=COLORS["dpma"], marker="D", seed_bars=True)
    _line(a, grouped, "dpma-v3", color=COLORS["dpma"], marker="D", ls=(0, (5, 2)), mfc="white",
          seed_bars=True, bar_offset=-0.014)
    if PREVIOUS:
        _line(a, grouped, PREVIOUS, color=INK, marker="*", ls=(0, (5, 2)), mfc="white", lw=1.8,
              seed_bars=True, bar_offset=0.028)
    _line(a, grouped, PRIMARY, color=INK, marker="*", lw=2.6, seed_bars=True, bar_offset=0.014)
    a.set_title("DiEP against the other methods")
    a.text(0.02, 0.9, "vertical bars: min–max over seeds, where a method has several runs\n"
           "(DiEP: 3 seeds at every ratio; DPMA v3: 3 seeds at keep 0.5)",
           transform=a.transAxes, ha="left", va="top", fontsize=8, color=MUTED)

    _line(b, grouped, PRIMARY, color=INK, marker="*", lw=2.6, seed_bars=True, direct=True)
    _line(b, grouped, PRIMARY + "-uniform", color=INK, marker="o", ls=(0, (5, 2)), mfc="white",
          direct=True)
    _line(b, grouped, PRIMARY + "-merge", color=INK, marker="^", ls=(0, (1.2, 1.8)), direct=True)
    _line(b, grouped, PRIMARY + "-skip", color=INK, marker="v", ls=(0, (7, 2, 1.5, 2)),
          mfc="white", direct=True)
    b.set_title("DiEP variants (seed 0; DiEP itself is the 3-seed mean)")

    for ax in (a, b):
        ax.set_xlabel("fraction of experts kept")
        ax.set_xticks(KEEPS)
        ax.set_xlim(0.2, 0.8)
        # Below the plot: inside, it would cover the dense reference line.
        ax.legend(frameon=False, fontsize=8.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncols=2)
    a.set_ylabel("mean one-shot accuracy, 8 MC benchmarks")
    fig.suptitle("OLMoE-1B-7B, identical C4 calibration (32 × 1024) and identical lm-eval protocol "
                 "for every line", fontsize=10, color=MUTED, y=0.995)
    fig.tight_layout()
    return save(fig, outdir, "comparison")


def fig_allocation(grouped, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, keep in zip(axes, KEEPS):
        uniform = 64 * keep
        ax.axhline(uniform, ls=(0, (6, 3)), lw=1.4, color=DENSE)
        ax.annotate(f"uniform {uniform:.0f}", (15, uniform), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8.5, color=MUTED)
        for method, color, marker, label in (
                ("ga-perplexity-100-gens", COLORS["ga-perplexity-100-gens"], "s", "GA (best arm)"),
                ("dpma-v3", COLORS["dpma"], "D", "DPMA v3, seed 0"),
                *(((PREVIOUS, DENSE, "*", LABELS[PREVIOUS].split(" (")[0] + ", seed 0"),)
                  if PREVIOUS else ()),
                (PRIMARY, INK, "*", LABELS[PRIMARY].split(" (")[0] + ", seed 0")):
            records = [r for r in grouped.get((method, keep), []) if r.get("replicate", 0) == 0]
            if records and records[0].get("layer_budgets"):
                budgets = records[0]["layer_budgets"]
                ax.plot(range(len(budgets)), budgets, color=color, marker=marker,
                        ms=7 if marker != "*" else 11, lw=1.8, label=label)
        ax.set_title(f"keep {keep:g}  ({int(1024 * keep)} of 1024 experts)")
        ax.set_xlabel("MoE layer")
        ax.set_xticks(range(0, 16, 3))
    axes[0].set_ylabel("experts kept in the layer")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Per-layer budgets each method discovered (floor: 8 for DiEP, the top-k)",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    return save(fig, outdir, "allocation")


def fig_search(searches, outdir):
    if not searches:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    styles = {"seed0": "-", "seed1": (0, (5, 2)), "seed2": (0, (1.2, 1.8))}
    for name, payload in searches.items():
        logs = payload["logs"]
        steps = [l["step"] for l in logs]
        ls = styles.get(name, "-")
        axes[0].plot(steps, [l["ce"] for l in logs], color=INK, ls=ls, label=name)
        axes[1].plot(steps, [l["phi"] for l in logs], color=INK, ls=ls, label=name)
        axes[2].plot(range(len(payload["beta"])), payload["beta"].tolist(), color=INK, ls=ls,
                     marker="o", ms=5, label=name)
        axes[3].plot(steps, [statistics.mean(l["alpha_entropy"]) for l in logs], color=INK,
                     ls=ls, label=name)
        for l in logs:
            if l["target"] == "beta":
                axes[0].axvline(l["step"], color=GRID, lw=1, zorder=0)
    axes[0].set_title("next-token CE (batch of 16)")
    axes[1].set_title("Φ = ‖H′ − H‖_F (per sequence)")
    axes[2].set_title("β per layer, end of search")
    axes[3].set_title("mean entropy of softmax(α) per layer")
    axes[0].set_xlabel("step (grey rules: β-steps)")
    axes[1].set_xlabel("step")
    axes[2].set_xlabel("MoE layer")
    axes[3].set_xlabel("step")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return save(fig, outdir, "search")


def save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------
# ANALYSIS.md
# --------------------------------------------------------------------------
def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def cell(grouped, method, keep):
    m, values = mean_mc(grouped, method, keep)
    if m is None:
        return "—"
    return f"{m:.4f} ±{statistics.stdev(values):.4f}" if len(values) > 1 else f"{m:.4f}"


def write_analysis(grouped, searches, bench, path: Path, merges=None, previous_searches=None):
    lines = ["# DiEP on OLMoE — results against the GA and DPMA", "",
             "Generated by `examples/report.py`. Method: [`../specs.md`](../specs.md) "
             "(Bai et al., NeurIPS 2025); what differs from the paper and why: "
             "[`../README.md`](../README.md).", ""]

    # 1. headline table
    methods = [("dense", "uncompressed"), ("random", "5 random draws"),
               ("frequency", "routing frequency"),
               ("ga-perplexity-100-gens", "GA, best arm"), ("ga-kl", "GA, KL fitness"),
               ("dpma", "DPMA v1"), ("dpma-v3", "DPMA v3, soft export"),
               ("dpma-v3-hard", "DPMA v3, hard export"),
               *(((PREVIOUS, LABELS[PREVIOUS]),) if PREVIOUS else ()),
               (PRIMARY, f"**{LABELS[PRIMARY].split(' (')[0]}**"),
               (PRIMARY + "-uniform", "DiEP, uniform per layer"),
               (PRIMARY + "-merge", "DiEP + merge"), (PRIMARY + "-skip", "DiEP + adaptive skip")]
    lines += ["## 1. Mean one-shot accuracy, 8 multiple-choice benchmarks", "",
              "Every row: OLMoE-1B-7B, calibration `c4-32x1024-seed0.pt`, lm-eval 0.4.12, "
              f"1-shot, headline metric per task. `±` is the standard deviation over seeds "
              f"or random draws; a row without it is one run. Chance is **{MC_CHANCE:.4f}**.", "",
              "| method | | keep 0.75 | keep 0.5 | keep 0.25 |", "|---|---|---|---|---|"]
    for method, note in methods:
        if method == "dense":
            m, _ = mean_mc(grouped, "dense", 1.0)
            lines.append(f"| dense | {note} | {fmt(m)} | {fmt(m)} | {fmt(m)} |")
            continue
        if not any((method, k) in grouped for k in KEEPS):
            continue
        lines.append(f"| `{method}` | {note} | " + " | ".join(cell(grouped, method, k) for k in KEEPS)
                     + " |")
    lines.append("")

    # 2. reading
    lines += ["## 2. Where DiEP sits", ""]
    for keep in KEEPS:
        diep, dv = mean_mc(grouped, PRIMARY, keep)
        if diep is None:
            lines.append(f"- **keep {keep:g}** — DiEP not evaluated yet.")
            continue
        rv = [r["mean_mc"] for r in grouped.get(("random", keep), [])]
        note = []
        if len(rv) > 1:
            mu, sd = statistics.mean(rv), statistics.stdev(rv)
            note.append(f"{(diep - mu) / sd:+.1f} σ from the random mean "
                        f"({'above' if diep > max(rv) else 'not above'} the best random draw)")
        others = {m: mean_mc(grouped, m, keep)[0] for m in
                  ("frequency", "ga-perplexity-100-gens", "dpma-v3", "dpma-v3-hard")}
        others = {m: v for m, v in others.items() if v is not None}
        if others:
            best = max(others, key=others.get)
            note.append(f"best other method `{best}` {others[best]:.4f} "
                        f"(DiEP {diep - others[best]:+.4f})")
        if len(dv) > 1:
            note.append(f"DiEP seed spread {max(dv) - min(dv):.4f} over {len(dv)} seeds")
        at_chance = diep < MC_CHANCE + 0.02
        lines.append(f"- **keep {keep:g}** — DiEP {diep:.4f}; " + "; ".join(note)
                     + (". Near chance: ranks methods weakly." if at_chance else "."))
    lines.append("")

    # 3. per task
    lines += ["## 3. Per task", "",
              "DiEP is the mean over its seeds; the others as in section 1.", ""]
    for keep in KEEPS:
        cols = [("dense", 1.0), ("random", keep), ("frequency", keep),
                ("ga-perplexity-100-gens", keep), ("dpma-v3-hard", keep), ("dpma-v3", keep),
                *(((PREVIOUS, keep),) if PREVIOUS else ()), (PRIMARY, keep)]
        present = [(m, k) for m, k in cols if (m, k) in grouped]
        if (PRIMARY, keep) not in grouped:
            continue
        lines += [f"**keep {keep:g}**", "",
                  "| task | chance | " + " | ".join(m if m != "random" else "random (mean)"
                                                  for m, _ in present) + " |",
                  "|---|---|" + "---|" * len(present)]
        for task in MC_TASKS:
            row = []
            for m, k in present:
                values = [task_value(r, task) for r in grouped[(m, k)]]
                values = [v for v in values if v is not None]
                row.append(fmt(statistics.mean(values), 3) if values else "—")
            lines.append(f"| {task} | {CHANCE[task]:.2f} | " + " | ".join(row) + " |")
        lines.append("")

    # 4. the spec's own checks, adapted
    lines += ["## 4. The specification's checks (§12), on OLMoE", ""]
    rows = []
    for keep in KEEPS:
        g, _ = mean_mc(grouped, PRIMARY, keep)
        u, _ = mean_mc(grouped, PRIMARY + "-uniform", keep)
        g0 = [r["mean_mc"] for r in grouped.get((PRIMARY, keep), []) if r.get("replicate") == 0]
        if u is not None and g0:
            rows.append(f"| {keep:g} | {g0[0]:.4f} | {u:.4f} | {g0[0] - u:+.4f} | "
                        f"{'yes' if g0[0] > u else 'no'} |")
    if rows:
        lines += ["DiEP must beat the uniform-per-layer baseline built from the same scores "
                  "(same seed):", "", "| keep | DiEP (seed 0) | uniform | Δ | beats uniform |",
                  "|---|---|---|---|---|", *rows, ""]
    lines += ["The Mixtral-8x7B acceptance numbers (§12) do not transfer to OLMoE and are not "
              "checked. The qualitative check — a non-uniform learned allocation — is in "
              "`figures/allocation.png`.", ""]

    # 4a. search length (only when an earlier DiEP run is being compared)
    if PREVIOUS:
        lines += steps_section(grouped, searches, previous_searches or {})

    # 4b. plain vs merge
    lines += merge_section(grouped, merges or {})

    # 5. search diagnostics
    if searches:
        lines += ["## 5. Search diagnostics", "",
                  "| seed | steps (α / β) | Φ at step 0 | CE first → last | β range at end | "
                  "softmax(α) entropy at end (max ln 64 = 4.1589) |", "|---|---|---|---|---|---|"]
        for name, payload in searches.items():
            logs, steps = payload["logs"], payload["steps"]
            beta = payload["beta"]
            ent = logs[-1]["alpha_entropy"]
            lines.append(f"| {name} | {steps['total']} ({steps['alpha']} / {steps['beta']}) | "
                         f"{logs[0]['phi']:.1e} | {logs[0]['ce']:.3f} → {logs[-1]['ce']:.3f} | "
                         f"{beta.min():.4f} – {beta.max():.4f} | {min(ent):.4f} – {max(ent):.4f} |")
        lines += ["", "Φ at step 0 should be `sqrt(1e-8) = 1e-4`: the search forward at "
                  "`alpha = 0, beta = 1` is the original model exactly. With 32 calibration "
                  "sequences an epoch is 2 steps, so the paper's 10 epochs are 20 steps here "
                  "(80 in the paper): alpha and beta move very little from their "
                  "initialisation, and the ranking is decided by small differences.", ""]

    # 6. efficiency and skipping
    if bench:
        lines += ["## 6. Efficiency (spec §6.1) and adaptive skipping", "",
                  "Generation, bf16, batch 1, 128-token prompt, 256 new tokens, 20 prompts after 3 "
                  "warm-ups, one L40S. `DiEP forward` is the Python-loop forward skipping needs; "
                  "the skip speedup is measured against it, not against the native kernels.", "",
                  "| keep | memory ratio | full tok/s | pruned tok/s (native) | speedup | "
                  "DiEP forward tok/s | + skip tok/s | skip speedup (same forward) | "
                  "pairs skipped |", "|---|---|---|---|---|---|---|---|---|"]
        for keep in KEEPS:
            b = bench.get(keep)
            if not b or "full_native" not in b:
                continue
            lines.append(
                f"| {keep:g} | {b['memory_ratio']:.3f} | {b['full_native']['tokens_per_s']:.1f} | "
                f"{b['pruned_native']['tokens_per_s']:.1f} | {b['pruned_native']['speedup_vs_full']:.2f}× | "
                f"{b['pruned_diep_forward']['tokens_per_s']:.1f} | "
                f"{b['pruned_skip']['tokens_per_s']:.1f} | "
                f"{b['pruned_skip']['speedup_vs_same_forward']:.2f}× | "
                f"{b['pruned_skip']['skip_fraction']:.1%} |")
        acc = []
        for keep in KEEPS:
            g0 = [r["mean_mc"] for r in grouped.get((PRIMARY, keep), []) if r.get("replicate") == 0]
            s = [r for r in grouped.get((PRIMARY + "-skip", keep), [])]
            if g0 and s:
                frac = s[0].get("diep", {}).get("skip_fraction")
                acc.append(f"| {keep:g} | {g0[0]:.4f} | {s[0]['mean_mc']:.4f} | "
                           f"{s[0]['mean_mc'] - g0[0]:+.4f} | "
                           f"{'—' if frac is None else f'{frac:.1%}'} |")
        if acc:
            lines += ["", "| keep | DiEP (seed 0) | + skip | Δ accuracy | pairs skipped in eval |",
                      "|---|---|---|---|---|", *acc]
        lines.append("")

    lines += ["## 7. Reading this fairly", "",
              "- **Same data, same protocol.** Every method here used the same 32 × 1024 C4 "
              "tokens and the same lm-eval configuration (the test suite checks DiEP's protocol "
              "constants against the GA's file).",
              "- **Not the same compute.** DiEP's search is 20 steps of α/β (minutes); the GA "
              "evaluates ~1,600 candidates; DPMA trains ~20k structural parameters for 900 steps "
              "and repairs dropped experts (absorb/merge), which DiEP does not (pure pruning).",
              "- **Seeds.** DiEP and DPMA v3 at keep 0.5 have three seeds each; everything else "
              "is one run, and random has five draws. A difference smaller than the random "
              "spread is not a result.",
              "- **Compute per token.** DiEP and `dpma-v3-hard` run at most 8 experts per token; "
              "`dpma-v3` (soft export) runs more (9.2 at keep 0.5), so it is not compute-matched.",
              ""]
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def _search_summary(searches) -> dict:
    """Mean over seeds of how far the search moved alpha and beta."""
    if not searches:
        return {}
    rows = []
    for payload in searches.values():
        alpha, beta, logs = payload["alpha"], payload["beta"], payload["logs"]
        rows.append({"steps": payload["steps"], "alpha_absmax": float(alpha.abs().max()),
                     "beta_min": float(beta.min()), "beta_max": float(beta.max()),
                     "entropy_min": min(logs[-1]["alpha_entropy"]),
                     "minutes": logs[-1]["elapsed"] / 60})
    first = rows[0]["steps"]
    return {"seeds": len(rows), "steps": f"{first['total']} ({first['alpha']} α / {first['beta']} β)",
            "alpha_absmax": statistics.mean(r["alpha_absmax"] for r in rows),
            "beta_min": min(r["beta_min"] for r in rows), "beta_max": max(r["beta_max"] for r in rows),
            "entropy_min": min(r["entropy_min"] for r in rows),
            "minutes": statistics.mean(r["minutes"] for r in rows)}


def steps_section(grouped, searches, previous_searches) -> list[str]:
    """The earlier DiEP run against this one: same calibration tokens, same
    protocol, only the number of search epochs differs."""
    lines = [f"## 4a. Search length: `{PREVIOUS}` vs `{PRIMARY}`", "",
             "Identical calibration file, evaluation and pruning; the only difference is the "
             "number of search epochs over the same 32 sequences. The comparison is judged "
             "against the two runs' own seed spreads.", "",
             f"| keep | `{PREVIOUS}` | `{PRIMARY}` | Δ | spread `{PREVIOUS}` | "
             f"spread `{PRIMARY}` | reading |", "|---|---|---|---|---|---|---|"]
    for keep in KEEPS:
        a, av = mean_mc(grouped, PREVIOUS, keep)
        b, bv = mean_mc(grouped, PRIMARY, keep)
        if a is None or b is None:
            lines.append(f"| {keep:g} | {cell(grouped, PREVIOUS, keep)} | "
                         f"{cell(grouped, PRIMARY, keep)} | — | — | — | not both run yet |")
            continue
        sa = max(av) - min(av) if len(av) > 1 else 0.0
        sb = max(bv) - min(bv) if len(bv) > 1 else 0.0
        # Separated if the two seed ranges do not overlap at all.
        separated = min(bv) > max(av) or max(bv) < min(av)
        reading = ("ranges do not overlap: a real difference" if separated
                   else "seed ranges overlap: not distinguishable")
        lines.append(f"| {keep:g} | {cell(grouped, PREVIOUS, keep)} | {cell(grouped, PRIMARY, keep)} "
                     f"| {b - a:+.4f} | {sa:.4f} | {sb:.4f} | {reading} |")
    lines.append("")
    before, after = _search_summary(previous_searches), _search_summary(searches)
    if before and after:
        lines += ["How far the search moved (over seeds):", "",
                  "| | steps | max \|α\| (mean over seeds) | β range | lowest entropy of softmax(α) "
                  "(uniform = 4.1589) | minutes per search |", "|---|---|---|---|---|---|"]
        for name, row in ((PREVIOUS, before), (PRIMARY, after)):
            lines.append(f"| `{name}` | {row['steps']} | {row['alpha_absmax']:.4f} | "
                         f"{row['beta_min']:.4f} – {row['beta_max']:.4f} | {row['entropy_min']:.4f} "
                         f"| {row['minutes']:.1f} |")
        lines.append("")
    # does DPMA still win?
    wins = []
    for keep in KEEPS:
        d, _ = mean_mc(grouped, "dpma-v3", keep)
        h, _ = mean_mc(grouped, "dpma-v3-hard", keep)
        b, _ = mean_mc(grouped, PRIMARY, keep)
        if b is not None and d is not None and h is not None:
            wins.append(f"| {keep:g} | {b:.4f} | {h:.4f} ({h - b:+.4f}) | {d:.4f} ({d - b:+.4f}) |")
    if wins:
        lines += [f"Does DPMA still beat DiEP with the longer search? (DPMA − `{PRIMARY}` in brackets)", "",
                  f"| keep | `{PRIMARY}` | DPMA v3, hard export | DPMA v3, soft export |",
                  "|---|---|---|---|", *wins, ""]
    return lines


def merge_section(grouped, merges) -> list[str]:
    """Plain DiEP against DiEP + the Appendix A.5 merge, paired on seed 0.

    Both come from the same seed-0 scores, so they keep exactly the same experts
    and the same router; the only difference is that the retained experts'
    weights are replaced by softmax(CKA)-weighted averages with their donors.
    """
    lines = ["## 4b. Plain DiEP vs DiEP + merge (Appendix A.5)", "",
             "Paired on seed 0: identical scores, identical surviving experts, identical "
             "router. The merge changes only the weights of retained experts that received "
             "donors. The yardstick is plain DiEP's own spread over 3 seeds.", ""]
    rows = []
    for keep in KEEPS:
        plain0 = [r for r in grouped.get((PRIMARY, keep), []) if r.get("replicate") == 0]
        merged = grouped.get((PRIMARY + "-merge", keep), [])
        plain_all = [r["mean_mc"] for r in grouped.get((PRIMARY, keep), [])]
        if not plain0 or not merged:
            rows.append(f"| {keep:g} | {fmt(plain0[0]['mean_mc']) if plain0 else '—'} | "
                        f"{fmt(merged[0]['mean_mc']) if merged else '—'} | — | — | not both run yet |")
            continue
        delta = merged[0]["mean_mc"] - plain0[0]["mean_mc"]
        spread = (max(plain_all) - min(plain_all)) if len(plain_all) > 1 else None
        verdict = ("—" if spread is None else
                   "outside the seed spread" if abs(delta) > spread else "inside the seed spread")
        rows.append(f"| {keep:g} | {plain0[0]['mean_mc']:.4f} | {merged[0]['mean_mc']:.4f} | "
                    f"{delta:+.4f} | {fmt(spread)} | {verdict} |")
    lines += ["| keep | plain (seed 0) | + merge (seed 0) | Δ | plain seed spread (max − min, 3 seeds) | reading |",
              "|---|---|---|---|---|---|", *rows, ""]

    # per task
    task_rows = []
    for keep in KEEPS:
        plain0 = [r for r in grouped.get((PRIMARY, keep), []) if r.get("replicate") == 0]
        merged = grouped.get((PRIMARY + "-merge", keep), [])
        if plain0 and merged:
            deltas = []
            for task in MC_TASKS:
                a, b = task_value(plain0[0], task), task_value(merged[0], task)
                deltas.append("—" if a is None or b is None else f"{b - a:+.3f}")
            task_rows.append(f"| {keep:g} | " + " | ".join(deltas) + " |")
    if task_rows:
        lines += ["Per-task change, merge − plain (seed 0):", "",
                  "| keep | " + " | ".join(MC_TASKS) + " |", "|---|" + "---|" * len(MC_TASKS),
                  *task_rows, ""]

    # what the merge did
    stat_rows = []
    for keep in KEEPS:
        report = merges.get(keep)
        if not report:
            continue
        groups = [g for layer in report for g in layer.values()]
        if not groups:
            continue
        donors = [len(g["members"]) - 1 for g in groups]
        own = [g["weights"][0] for g in groups]
        receivers = len(groups)
        retained = int(1024 * keep)
        stat_rows.append(f"| {keep:g} | {receivers} of {retained} | {statistics.mean(donors):.2f} | "
                         f"{max(donors)} | {statistics.mean(own):.3f} | {min(own):.3f} |")
    if stat_rows:
        lines += ["What the merge did (seed 0):", "",
                  "| keep | retained experts that received donors | donors per receiver (mean) | "
                  "(max) | weight the receiver keeps on itself (mean) | (min) |",
                  "|---|---|---|---|---|---|", *stat_rows, "",
                  "The weights are `softmax(CKA(e, r))` over the group, with `CKA(r, r) = 1`. "
                  "CKA lies in [0, 1], so the softmax is close to uniform: a receiver with "
                  "several donors keeps only a minority of its own weights.", ""]
    return lines


def main():
    args = parse_args()
    configure(args.primary, args.primary_label, args.previous, args.previous_label)
    style()
    results = Path(args.results)
    dpma = Path(args.dpma_results)
    extra = [Path(args.previous_results) / "harness"] if args.previous_results else []
    grouped = load_harness(results / "harness", args.ga_harness, dpma / "harness",
                           dpma / "v2" / "harness", dpma / "v3" / "harness", *extra)
    searches = load_searches(results / "runs")
    previous_searches = (load_searches(Path(args.previous_results) / "runs")
                         if args.previous_results else {})
    bench = load_bench(results / "runs")
    merges = load_merge_reports(results / "runs")
    outdir = results / "figures"
    fig_comparison(grouped, outdir)
    fig_allocation(grouped, outdir)
    fig_search(searches, outdir)
    write_analysis(grouped, searches, bench, results / "ANALYSIS.md", merges=merges,
                   previous_searches=previous_searches)


if __name__ == "__main__":
    main()
