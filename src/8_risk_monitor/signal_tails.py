"""Signal layer, one signal at a time: the cross-section by quarter, the right tail marked, and
whether the tail is growing. Plus pairwise portfolio overlap between manager families, and two
preliminary confounder splits (borrower size, BDC type).

Signals (higher = riskier, so the risk tail is always the right tail):

    unit              signal                      definition
    BDC-quarter       concentration               HHI of the BDC's book across borrowers (cost
                                                  weights); effective N = 1 / HHI
    borrower-quarter  crowding                    C1: lender manager families holding the
                                                  borrower's funded debt
    BDC-quarter       dispersion                  par-weighted mean |lender's mark - consensus|
                                                  over the BDC's loans held by 2+ families: how far
                                                  this BDC's marks sit from its co-lenders'
    BDC-quarter       PIK, total                  share of the BDC's debt FV paying any PIK
    BDC-quarter       PIK, amendment-driven       share of debt FV on acquired PIK: cash-pay at
                                                  origination, PIK now (includes P2 toggles)

The tail. A within-quarter percentile always puts 10% of units in its top decile, so it cannot
show a growing tail. Each signal's tail threshold is fixed instead: the 90th percentile of the
pooled 2023Q2-2024Q4 cross-section (the baseline period). The chart counts units above it each
quarter. Units: BDCs with at least MIN_POSITIONS borrowers (concentration, PIK) or MIN_SHARED
multi-family loans (dispersion); borrowers with BDC-held debt (crowding).

Pairwise overlap (C5). For two manager families, dollar-weighted overlap is
sum over shared borrowers of min(w_A, w_B), w = the borrower's cost share of the family's book:
0 = no common borrowers, 1 = identical books. Jaccard = shared borrowers / borrowers in either.
Families with at least MIN_FAMILY_BORROWERS borrowers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from charts import SURFACE, INK, MUTED, GRID, BLUE, ORANGE, AQUA, _style
from val_dispersion import bdc_mark_distance

BASELINE = ("2023Q2", "2024Q4")
TAIL_Q = 0.90
MIN_POSITIONS = 20
MIN_SHARED = 5
MIN_FAMILY_BORROWERS = 20
TYPE_COLORS = {"listed": BLUE, "non-traded": ORANGE, "private": AQUA}

SIGNALS = {
    "concentration": ("BDC", "Concentration: HHI of the BDC's book across borrowers", "HHI"),
    "crowding": ("borrower", "Crowding: lender manager families per borrower", "lender families"),
    "dispersion": ("BDC", "Dispersion: BDC's mean distance from co-lender consensus mark", "|mark - consensus|, share of par"),
    "pik_total": ("BDC", "PIK, total: share of the BDC's debt paying PIK", "share of debt FV"),
    "pik_amendment": ("BDC", "PIK, amendment-driven: share on PIK acquired after origination", "share of debt FV"),
}


def _borrower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def bdc_signals(panel: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """One row per BDC-quarter: the BDC-level signals, bdc_type and size."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "COST"]).to_pandas()
    r = r[(r["COST"] > 0) & r["borrower_canonical"].notna()]
    w = r.assign(b=_borrower(r["borrower_canonical"])).groupby(["cik", "cal_q", "b"])["COST"].sum()
    share = w / w.groupby(["cik", "cal_q"]).transform("sum")
    conc = pd.DataFrame({"concentration": (share ** 2).groupby(["cik", "cal_q"]).sum(),
                         "n_borrowers": share.groupby(["cik", "cal_q"]).size()})
    conc.loc[conc["n_borrowers"] < MIN_POSITIONS, "concentration"] = np.nan

    p = panel[panel["PAR"] > 0]
    keys = ["cik", "cal_q"]
    g = p.groupby(keys)
    out = pd.DataFrame({
        "fv": g["FV"].sum(),
        "bdc_type": g["bdc_type"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
        "pik_total": p["FV"].where(p["pik_share"].fillna(0) > 0, 0).groupby([p[k] for k in keys]).sum() / g["FV"].sum(),
        "pik_amendment": p["FV"].where(p["pik_acquired"].fillna(False).astype(bool), 0)
                         .groupby([p[k] for k in keys]).sum() / g["FV"].sum(),
    })
    disp = bdc_mark_distance(p, min_shared=MIN_SHARED)
    out = out.join(conc).join(disp)
    for c in ("pik_total", "pik_amendment"):
        out.loc[out["n_borrowers"].fillna(0) < MIN_POSITIONS, c] = np.nan
    out["effective_n"] = 1 / out["concentration"]
    return out.reset_index()


def borrower_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per borrower-quarter: lender families (C1), BDC-held par and its size quintile."""
    p = panel[~panel["borrower_key"].str.startswith("ID: ") & (panel["PAR"] > 0)]
    b = p.groupby([_borrower(p["borrower_key"]).rename("borrower"), "cal_q"]).agg(
        crowding=("C1_families", "first"), par=("PAR", "sum")).reset_index()
    b["size_quintile"] = b.groupby("cal_q")["par"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    return b


def _quarters(df):
    return sorted(df["cal_q"].unique())


def tail_summary(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, float]:
    """Per quarter: percentiles, the fixed baseline tail threshold and the units above it."""
    d = df[df[col].notna()]
    base = d[(d["cal_q"] >= BASELINE[0]) & (d["cal_q"] <= BASELINE[1])]
    thr = float(base[col].quantile(TAIL_Q))
    q = d.groupby("cal_q")[col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    q["n_tail"] = d.assign(t=d[col] > thr).groupby("cal_q")["t"].sum()
    q["share_tail"] = q["n_tail"] / q["count"]
    q["threshold"] = thr
    return q.reset_index(), thr


def plot_signal_tail(df: pd.DataFrame, col: str, path) -> pd.DataFrame:
    unit, title, ylabel = SIGNALS[col]
    q, thr = tail_summary(df, col)
    d = df[df[col].notna()]
    xs = {k: i for i, k in enumerate(q["cal_q"])}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    x = np.arange(len(q))
    ax.fill_between(x, q["10%"], q["90%"], color=BLUE, alpha=0.12, linewidth=0, label="10th-90th percentile")
    ax.fill_between(x, q["25%"], q["75%"], color=BLUE, alpha=0.25, linewidth=0, label="25th-75th percentile")
    ax.plot(x, q["50%"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Median")
    tail = d[d[col] > thr]
    if unit == "BDC":
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(tail))
        ax.scatter(tail["cal_q"].map(xs) + jitter, tail[col], s=12, color=ORANGE, alpha=0.7, linewidths=0,
                   label=f"BDCs in the tail", zorder=3)
    ax.axhline(thr, color=ORANGE, linewidth=1.2, linestyle="--", label=f"Tail threshold {thr:.3g} (90th pct, 2023-24)")
    top = max(q["90%"].max(), thr) * 1.6 if unit == "BDC" else None
    if top and tail[col].max() > top:
        ax.set_ylim(top=top)
        ax.text(0.99, 0.98, "a few tail points above the axis are clipped", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=MUTED)
    _style(ax, title, ylabel)
    ax.set_xticks(x, q["cal_q"], rotation=45)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.bar(x, 100 * q["share_tail"], color=ORANGE, width=0.6)
    for i, (n, c) in enumerate(zip(q["n_tail"], q["count"])):
        ax.text(i, 100 * q["share_tail"].iat[i], f"{int(n)}/{int(c)}", ha="center", va="bottom", fontsize=6.5,
                color=MUTED, rotation=90 if c >= 1000 else 0)
    base = d[(d["cal_q"] >= BASELINE[0]) & (d["cal_q"] <= BASELINE[1])]
    base_share = float((base[col] > thr).mean())
    ax.axhline(100 * base_share, color=MUTED, linewidth=1, linestyle=":")
    _style(ax, f"{'BDCs' if unit == 'BDC' else 'Borrowers'} above the tail threshold", f"% of {unit.lower()}s")
    ax.set_xticks(x, q["cal_q"], rotation=45)
    fig.text(0.01, 0.01, f"Dotted line: {base_share:.1%}, the share above the threshold in the pooled 2023Q2-2024Q4 baseline. "
             "Labels: units in the tail / units with the signal.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return q.assign(signal=col, unit=unit)


# ------------------------------------------------------------------ pairwise overlap (C5)

def family_books(rows: pl.DataFrame) -> pd.DataFrame:
    r = rows.select(["family", "cal_q", "borrower_canonical", "COST"]).to_pandas()
    r = r[(r["COST"] > 0) & r["borrower_canonical"].notna() & r["family"].notna()]
    w = r.assign(b=_borrower(r["borrower_canonical"])).groupby(["cal_q", "family", "b"])["COST"].sum().reset_index()
    w["w"] = w["COST"] / w.groupby(["cal_q", "family"])["COST"].transform("sum")
    w["n_b"] = w.groupby(["cal_q", "family"])["b"].transform("size")
    return w[w["n_b"] >= MIN_FAMILY_BORROWERS]


def pairwise_overlap(rows: pl.DataFrame) -> pd.DataFrame:
    """Every family pair per quarter with at least one shared borrower: dollar-weighted
    overlap and Jaccard. Pairs with no shared borrower have overlap 0 and are not listed;
    n_pairs_total gives the denominator."""
    books = family_books(rows)
    out = []
    for q, g in books.groupby("cal_q"):
        sizes = g.groupby("family")["b"].size()
        m = g.merge(g, on=["cal_q", "b"], suffixes=("_a", "_b"))
        m = m[m["family_a"] < m["family_b"]]
        m = m.assign(w_min=np.minimum(m["w_a"], m["w_b"]))
        pr = m.groupby(["family_a", "family_b"]).agg(shared=("b", "size"), overlap=("w_min", "sum")).reset_index()
        pr["jaccard"] = pr["shared"] / (pr["family_a"].map(sizes) + pr["family_b"].map(sizes) - pr["shared"])
        pr["cal_q"] = q
        pr["n_pairs_total"] = len(sizes) * (len(sizes) - 1) // 2
        out.append(pr)
    return pd.concat(out, ignore_index=True)


def plot_overlap(pairs: pd.DataFrame, rows: pl.DataFrame, path, quarter: str = "2026Q2", top: int = 15):
    """Left: heatmap of dollar-weighted overlap among the largest families at `quarter`.
    Right: per quarter, the share of family pairs whose books overlap by more than 10% and 20%."""
    from matplotlib.colors import LinearSegmentedColormap
    books = family_books(rows)
    size = books[books["cal_q"] == quarter].groupby("family")["COST"].sum().sort_values(ascending=False)
    fams = list(size.index[:top])
    p = pairs[(pairs["cal_q"] == quarter) & pairs["family_a"].isin(fams) & pairs["family_b"].isin(fams)]
    mat = pd.DataFrame(0.0, index=fams, columns=fams)
    for r in p.itertuples():
        mat.loc[r.family_a, r.family_b] = mat.loc[r.family_b, r.family_a] = r.overlap
    np.fill_diagonal(mat.values, np.nan)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=max(0.3, np.nanmax(mat.values)))
    short = [f[:22] for f in fams]
    ax.set_xticks(range(len(fams)), short, rotation=60, ha="right", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(fams)), short, fontsize=7, color=MUTED)
    for i in range(len(fams)):
        for j in range(len(fams)):
            v = mat.values[i, j]
            if not np.isnan(v) and v >= 0.05:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.18 else INK)
    ax.set_title(f"Dollar-weighted portfolio overlap, {top} largest manager families, {quarter}",
                 loc="left", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    ax = axes[1]
    s = pairs.groupby("cal_q").agg(n_total=("n_pairs_total", "first"),
                                   over10=("overlap", lambda v: (v > 0.10).sum()),
                                   over20=("overlap", lambda v: (v > 0.20).sum()),
                                   any_shared=("overlap", "size")).reset_index()
    x = np.arange(len(s))
    ax.plot(x, 100 * s["over10"] / s["n_total"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Overlap > 10%")
    ax.plot(x, 100 * s["over20"] / s["n_total"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Overlap > 20%")
    _style(ax, "Family pairs with overlapping books", "% of family pairs")
    ax.set_xticks(x, s["cal_q"], rotation=45)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return mat, s


# ------------------------------------------------------------------ confounder splits

def size_split(bs: pd.DataFrame, path) -> pd.DataFrame:
    """Are crowded borrowers just big borrowers? Mean lender families and the share in the
    crowding tail by size quintile (BDC-held par within the quarter), pooled over quarters,
    plus the within-quarter Spearman correlation of size and lender families."""
    _, thr = tail_summary(bs, "crowding")
    d = bs[bs["crowding"].notna()].assign(tail=lambda x: x["crowding"] > thr)
    t = d.groupby("size_quintile").agg(borrowers=("crowding", "size"), mean_families=("crowding", "mean"),
                                       share_in_tail=("tail", "mean"), median_par_musd=("par", lambda s: s.median() / 1e6)
                                       ).reset_index()
    rho = d.groupby("cal_q").apply(lambda g: g["crowding"].corr(g["par"], method="spearman"))
    t["spearman_size_vs_families_median_q"] = float(rho.median())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    lab = [f"Q{int(k)}\n${v:,.0f}M" for k, v in zip(t["size_quintile"], t["median_par_musd"])]
    axes[0].bar(lab, t["mean_families"], color=BLUE, width=0.6)
    for i, v in enumerate(t["mean_families"]):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=MUTED)
    _style(axes[0], "Mean lender families by borrower size quintile", "lender families")
    axes[1].bar(lab, 100 * t["share_in_tail"], color=ORANGE, width=0.6)
    for i, v in enumerate(t["share_in_tail"]):
        axes[1].text(i, 100 * v, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=MUTED)
    _style(axes[1], f"Share of borrowers in the crowding tail (> {thr:.0f} families)", "% of borrowers")
    fig.suptitle(f"Crowding vs size: Spearman correlation of BDC-held par and lender families, median quarter: "
                 f"{rho.median():.2f}. Size quintile labels show median BDC-held par.",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return t


def bdc_type_split(bdc: pd.DataFrame, path) -> pd.DataFrame:
    """Do signals look different by BDC type? Median BDC per type and quarter, one small
    multiple per BDC-level signal, plus the share of each type's BDCs in the tail."""
    cols = ["concentration", "dispersion", "pik_total", "pik_amendment"]
    rows = []
    for c in cols:
        _, thr = tail_summary(bdc, c)
        d = bdc[bdc[c].notna()]
        g = d.assign(tail=d[c] > thr).groupby(["cal_q", "bdc_type"]).agg(
            median=(c, "median"), n=(c, "size"), share_in_tail=("tail", "mean")).reset_index()
        rows.append(g.assign(signal=c, threshold=thr))
    t = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), facecolor=SURFACE)
    for j, c in enumerate(cols):
        for i, (metric, ylab) in enumerate([("median", SIGNALS[c][2]), ("share_in_tail", "% of BDCs in the tail")]):
            ax = axes[i, j]
            for typ, color in TYPE_COLORS.items():
                g = t[(t["signal"] == c) & (t["bdc_type"] == typ)].sort_values("cal_q")
                y = g[metric] * (100 if metric == "share_in_tail" else 1)
                ax.plot(g["cal_q"], y, color=color, linewidth=2, marker="o", markersize=3.5, label=typ)
                if len(g):
                    ax.annotate(typ, (len(g) - 1, y.iat[-1]), xytext=(4, 0), textcoords="offset points",
                                fontsize=7, color=MUTED, va="center")
            _style(ax, (SIGNALS[c][1].split(":")[0] + (" - median BDC" if i == 0 else " - share in tail")), ylab)
            ax.tick_params(axis="x", rotation=60, labelsize=6.5)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Signals by BDC type: median BDC (top) and share of BDCs above the 2023-24 tail threshold (bottom)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return t
