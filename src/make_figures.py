"""make_figures.py - Figures 2-3 from gate_test_results.csv.
Run from the project folder with the virtual-environment Python: python make_figures.py
"""
import csv
import os
import re
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG: edit these to match your CSV ----------------
CSV_PATH = "experiments/gate_test_results.csv"
COL_TRUE = "path"        # the path contains the true cow's folder, e.g. ...\cow_1\60.jpg
COL_PRED = "predicted"
COL_SCORE = "confidence"
AUTHORISED = {"cow_1", "cow_2"}
UNSEEN = {"cow_44", "cow_45", "cow_46"}
THRESHOLD = 0.30
OUT_DIR = "figures"
MAKE_SWEEP = False                # set True if want a threshold-sweep figure
# -----------------------------------------------------------------------

GREEN, ORANGE, BLUE, GREY = "#009E73", "#E69F00", "#0072B2", "#8A8A8A"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})


def ident(text):
    m = re.search(r"cow_\d+", str(text))
    if not m:
        raise ValueError(f"No cow_N identity found in: {text!r}")
    return m.group(0)


def group_of(true_id):
    if true_id in AUTHORISED:
        return "Authorised"
    if true_id in UNSEEN:
        return "Unseen"
    return "Enrolled, unauthorised"


def load():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t, p = ident(r[COL_TRUE]), ident(r[COL_PRED])
            rows.append({"true": t, "pred": p, "score": float(r[COL_SCORE]), "group": group_of(t)})
    return rows


def is_open(r, tau=THRESHOLD):
    return r["pred"] in AUTHORISED and r["score"] >= tau


def passes(r, tau=THRESHOLD):
    return r["pred"] == r["true"] and r["score"] >= tau


def sanity_checks(rows):
    print(f"Rows: {len(rows)}")
    for g in ("Authorised", "Enrolled, unauthorised", "Unseen"):
        sub = [r for r in rows if r["group"] == g]
        print(f"{g}: n={len(sub)}, gate open={sum(is_open(r) for r in sub)}")
    auth = [r for r in rows if r["group"] == "Authorised"]
    ok = [r["score"] for r in auth if is_open(r) and r["pred"] == r["true"]]
    print(f"Lowest score among correctly accepted authorised images: {min(ok):.3f}")
    enrolled = [r for r in rows if r["group"] != "Unseen"]
    print(f"Median score, authorised: {median(r['score'] for r in auth):.2f}; "
          f"all enrolled test images: {median(r['score'] for r in enrolled):.2f}")
    print(f"Enrolled images correct and >= {THRESHOLD}: "
          f"{sum(passes(r) for r in enrolled)}/{len(enrolled)}")
    imp = [r for r in rows if r["group"] != "Authorised" and r["pred"] in AUTHORISED]
    print("Non-authorised images predicted as an authorised identity:")
    for r in imp:
        print(f"  true={r['true']} pred={r['pred']} score={r['score']:.3f}")
    by_id = {}
    for r in enrolled:
        by_id.setdefault(r["true"], []).append(passes(r))
    all_pass = sorted(i for i, v in by_id.items() if all(v))
    none_pass = sorted(i for i, v in by_id.items() if not any(v))
    print(f"Identities passing every test image: {len(all_pass)}; passing none: {len(none_pass)} {none_pass}")


def fig_scores(rows):
    groups = [("Authorised", GREEN), ("Enrolled, unauthorised", ORANGE), ("Unseen", BLUE)]
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.45, 3.1))
    for ax, (g, col) in zip(axes, groups):
        s = [r["score"] for r in rows if r["group"] == g]
        ax.hist(s, bins=bins, color=col, edgecolor="white", linewidth=0.3)
        ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(f"{g} (n={len(s)})", loc="right", fontsize=7, pad=2)
    axes[-1].set_xlabel("Top-class confidence score")
    axes[-1].set_xlim(0, 1)
    fig.text(0.02, 0.5, "Number of test images", rotation=90, va="center", fontsize=8)
    fig.tight_layout(rect=(0.04, 0, 1, 1), h_pad=0.4)
    save(fig, "fig2_score_distributions")


def fig_identity(rows):
    enrolled = [r for r in rows if r["group"] != "Unseen"]
    by_id = {}
    for r in enrolled:
        by_id.setdefault(r["true"], []).append(passes(r))
    data = [(i, sum(v) / len(v), len(v)) for i, v in by_id.items()]
    data.sort(key=lambda x: (-x[1], int(x[0].split("_")[1])))
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    xs = range(len(data))
    ax.bar(xs, [d[1] * 100 for d in data],
           color=[GREEN if d[0] in AUTHORISED else GREY for d in data], width=0.75)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{d[0].split('_')[1]} (n={d[2]})" for d in data], rotation=90, fontsize=5.5)
    ax.set_xlabel("Enrolled identity (cow number, number of test images)")
    ax.set_ylabel("Test images passing (%)")
    ax.set_ylim(0, 105)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=GREY)],
              labels=["Authorised", "Other enrolled"], frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_identity_pass_rate")


def fig_sweep(rows):
    taus = np.arange(0.30, 0.901, 0.05)
    auth = [r for r in rows if r["group"] == "Authorised"]
    enrolled = [r for r in rows if r["group"] != "Unseen"]
    non_auth = [r for r in rows if r["group"] != "Authorised"]
    acc = [100 * sum(is_open(r, t) for r in auth) / len(auth) for t in taus]
    usable = [100 * sum(passes(r, t) for r in enrolled) / len(enrolled) for t in taus]
    fa = [sum(is_open(r, t) for r in non_auth) for t in taus]
    print("tau  auth_accept%  enrolled_correct_above%  false_accepts")
    for t, a, u, f in zip(taus, acc, usable, fa):
        print(f"{t:.2f}  {a:6.1f}  {u:6.1f}  {f}")
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.plot(taus, acc, marker="o", markersize=3, color=GREEN, label="Authorised images accepted")
    ax.plot(taus, usable, marker="s", markersize=3, color=BLUE, label="Enrolled images correct and above threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Test images (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig4_threshold_sweep")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=300)
    plt.close(fig)
    print("Saved", name)


if __name__ == "__main__":
    data = load()
    sanity_checks(data)
    fig_scores(data)
    fig_identity(data)
    if MAKE_SWEEP:
        fig_sweep(data)
