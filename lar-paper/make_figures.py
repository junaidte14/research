"""Builds the figures and the condensed main table for the Neurocomputing version.
Inputs: tables/*.tex (generated earlier from the result files) and lar_v9_results.json (raw per-run results).
Outputs: figures/*.pdf and tables/tab_main_stab.tex. Run from the project folder: python make_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "tables"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 150,
})
COL = {"ce": "#7f7f7f", "aug": "#E69F00", "lara": "#56B4E9", "larc": "#0072B2", "pos": "#D55E00"}


def rows_of(name):
    """Rows of a generated tabular: list of lists of stripped cell strings (header and rules skipped)."""
    out = []
    for line in (TAB / name).read_text().splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("\\"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        out.append(cells)
    return out


def num(s):
    return float(s.replace("$-$", "-").replace("+", ""))


# ------------------------------------------------------------------ Figure: pairing gap
def fig_pairing():
    rows = rows_of("tab_alignment.tex")
    want = {"SST-2 large": "SST-2", "RT large": "Rotten Tomatoes", "IMDB 5k": "IMDB 5k", "IMDB 20k": "IMDB 20k"}
    data = {}
    for r in rows:
        if r[0] in want:
            data.setdefault(r[0], {})[r[1]] = (num(r[2]), num(r[4]))      # positional, order-preserving
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.1), sharey=True)
    axes = axes.ravel()
    x = np.arange(3)
    for ax, (k, title) in zip(axes, want.items()):
        pos = [data[k][s][0] for s in ("light", "medium", "heavy")]
        ide = [data[k][s][1] for s in ("light", "medium", "heavy")]
        ax.fill_between(x, pos, ide, color=COL["pos"], alpha=0.15, lw=0)
        ax.plot(x, ide, "-o", color=COL["larc"], ms=4, lw=1.8, label="identity pairing")
        ax.plot(x, pos, "--s", color=COL["pos"], ms=4, lw=1.8, label="positional pairing")
        ax.annotate(f"gap {ide[2] - pos[2]:.2f}", (2, (pos[2] + ide[2]) / 2), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color="#444444")
        ax.set_xticks(x, ["light", "medium", "heavy"])
        ax.set_title(title)
        ax.set_ylim(0.52, 1.0)
        ax.set_xlim(-0.25, 2.25)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[0].set_ylabel(r"stability $\rho$")
    axes[2].set_ylabel(r"stability $\rho$")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pairing.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: stability by strength
def parse_main(name):
    out = {}
    for r in rows_of(name):
        m = [re.match(r"([\d.]+)\s*\(([\d.]+)\)", c) for c in r[2:5]]
        if not all(m):
            continue                                   # header row
        out[r[0]] = [(float(a.group(1)), float(a.group(2))) for a in m]
    return out


def fig_strength():
    sets = [("SST-2", "tab_large_main.tex", "LAR-C, $\\lambda{=}0.25$"),
            ("Rotten Tomatoes", "tab_rt_main.tex", "LAR-C, $\\lambda{=}0.25$"),
            ("IMDB 5k", "tab_imdb_main.tex", "LAR-C, $\\lambda{=}0.25$"),
            ("IMDB 20k", "tab_imdb_large_main.tex", "LAR-C, $\\lambda{=}0.25$")]
    series = [("CE only", "ce", "plain fine-tuning", "o"), ("Augmentation", "aug", "augmentation", "s"),
              ("LAR-A (cos, aligned)", "lara", "LAR-A", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.6))
    axes = axes.ravel()
    x = np.arange(3)
    for ax, (title, f, larc_key) in zip(axes, sets):
        d = parse_main(f)
        for key, c, label, mk in series:
            m = np.array([v[0] for v in d[key]]); s = np.array([v[1] for v in d[key]])
            ax.errorbar(x, m, yerr=s, color=COL[c], marker=mk, ms=4, lw=1.5, capsize=2, label=label)
        m = np.array([v[0] for v in d[larc_key]]); s = np.array([v[1] for v in d[larc_key]])
        ax.errorbar(x, m, yerr=s, color=COL["larc"], marker="D", ms=4, lw=1.9, capsize=2, label="LAR-C")
        ax.set_xticks(x, ["light", "medium", "heavy"])
        ax.set_title(title)
        ax.set_xlim(-0.25, 2.25)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[0].set_ylabel(r"stability $\rho$")
    axes[2].set_ylabel(r"stability $\rho$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig_strength.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: forest plot of side measures
def fig_forest():
    rows = [r for r in rows_of("tab_effects_all.tex") if re.search(r"\d\.\d", r[1])]
    heads = ["SST-2", "Rotten Tomatoes", "IMDB 20k\nall seeds", "IMDB 20k\nwithout collapsed runs"]
    labels = ["Stability (Spearman)", "Top-20% overlap", "Integrated-Gradients stability",
              "Clean accuracy", "Prediction flip rate (lower is better)", "Comprehensiveness gain", "Sufficiency gain"]
    groups = [0, 0, 0, 1, 1, 2, 2]      # 0 attribution measures, 1 prediction measures, 2 faithfulness
    gcol = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes_grid = plt.subplots(2, 2, figsize=(5.6, 5.3), sharey=True, sharex=True)
    axes = axes_grid.ravel()
    for j, ax in enumerate(axes):
        for i, r in enumerate(rows):
            cell = r[j + 1]
            star = "^{*}" in cell
            vals = [float(v) for v in re.findall(r"[-+]?\d+\.\d+", cell.replace("$-$", "-"))]
            est, lo, hi = vals
            y = len(rows) - 1 - i
            c = gcol[groups[i]]
            ax.plot([lo, hi], [y, y], color=c, lw=1.8, solid_capstyle="butt")
            excl = lo > 0 or hi < 0
            ax.plot(est, y, "o", color=c if excl else "white", mec=c, ms=5.5, mew=1.3, zorder=3)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(heads[j].replace("\n", ", "), fontsize=8.5)
        ax.set_xlim(-0.13, 0.23)
        ax.grid(axis="x", lw=0.4, alpha=0.5)
        ax.set_yticks(range(len(rows)), labels[::-1] if False else list(reversed(labels)))
        if j % 2 == 1:
            ax.tick_params(axis="y", length=0)
    for ax in axes[[0, 2]]:
        ax.tick_params(axis="y", labelsize=7.8)
    fig.supxlabel("LAR-C minus augmentation (paired mean difference, 95% confidence interval)", fontsize=8.5, y=0.005)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig_effects.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Figure: collapsed runs (raw per-seed accuracy)
def fig_collapse():
    rows = json.loads((ROOT / "lar_v9_results.json").read_text())["rows"]
    conds = [("ce_only_control", "plain\nfine-tuning"), ("aug_only", "augmentation"),
             ("lar_aligned", "LAR-A"), ("lar_pearson_l0.25", "LAR-C")]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), sharey=True)
    rng = np.random.RandomState(3)
    for ax, (phase, title) in zip(axes, [("G_imdb_medium", "IMDB 5k"), ("G_imdb_large", "IMDB 20k")]):
        for k, (c, lab) in enumerate(conds):
            acc = {}
            for r in rows:
                if r["phase"] == phase and r["condition"] == c and r["strength"] == "medium":
                    acc[r["seed"]] = r["acc_full"]
            v = np.array([acc[s] for s in sorted(acc)])
            bad = v < 0.70
            xs = k + rng.uniform(-0.17, 0.17, len(v))
            ax.scatter(xs[~bad], v[~bad], s=26, color="#0072B2", alpha=0.85, lw=0)
            ax.scatter(xs[bad], v[bad], s=34, color="#D55E00", marker="X", lw=0)
            ax.text(k, 0.405, f"{int(bad.sum())}/{len(v)}", ha="center", va="center", fontsize=8,
                    color="#D55E00" if bad.any() else "#555555")
        ax.axhline(0.70, color="#444444", lw=0.8, ls=":")
        ax.set_xticks(range(len(conds)), [c[1].replace("\n", " ") if False else c[1] for c in conds], fontsize=7.5)
        ax.set_title(title)
        ax.set_ylim(0.37, 0.90)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[0].set_ylabel("clean accuracy per seed")
    axes[0].text(-0.45, 0.715, "0.70", fontsize=7.5, color="#444444", va="bottom")
    fig.text(0.5, 0.005, "numbers show runs below 0.70 accuracy (crosses) out of ten", ha="center", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_collapse.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Condensed main table
def clean_cell(c):
    return c


def table_main():
    stab = {r[0]: r for r in rows_of("tab_stab_all.tex")}
    con = {r[0]: r for r in rows_of("tab_contrast_all.tex")}
    cols = [2, 3, 4, 5]                                    # SST-2 large, RT, IMDB 5k, IMDB 20k
    names = ["SST-2", "Rotten Tomatoes", "IMDB 5k", "IMDB 20k"]
    la, lc = "LAR-A (cos, aligned)", "LAR-C, $\\lambda{=}0.25$"
    a = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Dataset & Plain & Augm. & LAR-A & LAR-C \\\\", "\\midrule"]
    b = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Dataset & LAR-A $-$ Augm. & LAR-C $-$ Augm. \\\\", "\\midrule"]
    for nm, c in zip(names, cols):
        a.append(f"{nm} & {stab['CE only'][c]} & {stab['Augmentation'][c]} & {stab[la][c]} & {stab[lc][c]} \\\\")
        b.append(f"{nm} & {con[la][c]} & {con[lc][c]} \\\\")
    a += ["\\bottomrule", "\\end{tabular}"]
    b += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_main_stab_a.tex").write_text("\n".join(a) + "\n")
    (TAB / "tab_main_stab_b.tex").write_text("\n".join(b) + "\n")


def table_lambda():
    """Lambda sweep without the column that was not recorded for every row."""
    rows = [r for r in rows_of("tab_lambda.tex")]
    keep = [0, 1, 3, 4]
    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "Condition & $\\rho$ (medium) & Clean accuracy & Compreh.\\ gain \\\\", "\\midrule"]
    for r in rows[1:]:
        lines.append(" & ".join(r[i] for i in keep) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_lambda_short.tex").write_text("\n".join(lines) + "\n")


def table_data():
    ds = {r[0] + "|" + r[1]: r for r in rows_of("tab_datasets_measured.tex")}
    bank = {r[0]: r for r in rows_of("tab_bank_train.tex")}
    spec = [("SST-2", "SST-2|train (used)", "SST-2|eval", "SST-2", "64"),
            ("Rotten Tomatoes", "Rotten Tomatoes|train (used)", "Rotten Tomatoes|eval", "Rotten Tomatoes", "64"),
            ("IMDB 5k", "IMDB (5k)|train (used)", "IMDB (both)|eval", "IMDB (5k)", "128"),
            ("IMDB 20k", "IMDB (20k)|train (used)", "IMDB (both)|eval", "IMDB (20k)", "128")]
    lines = ["\\begin{tabular}{@{}lrrcccc@{}}", "\\toprule",
             "Setting & Train & Eval & Words (tr./ev.) & Max len. & Eval, eligible (\\%) & Train, edited (\\%) \\\\", "\\midrule"]
    for name, tr, ev, bk, ml in spec:
        t, e = ds[tr], ds[ev]
        words = f"{t[4].split('/')[0]} / {e[4].split('/')[0]}"
        lines.append(f"{name} & {t[2]} & {e[2]} & {words} & {ml} & {e[8]} & {bank[bk][1]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_data_short.tex").write_text("\n".join(lines) + "\n")


def table_effects_split():
    """The full effects table split into two narrower tables so it stays readable."""
    rows = rows_of("tab_effects_all.tex")
    head, body = rows[0], rows[1:]
    for name, cols in (("a", [1, 2]), ("b", [3, 4])):
        lines = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
                 " & ".join([head[0]] + [head[c] for c in cols]) + " \\\\", "\\midrule"]
        for r in body:
            lines.append(" & ".join([r[0]] + [r[c] for c in cols]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (TAB / f"tab_effects_{name}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_pairing()
    fig_strength()
    fig_forest()
    fig_collapse()
    table_main()
    table_lambda()
    table_data()
    table_effects_split()
    print("done:", sorted(p.name for p in FIG.glob("*.pdf")))
