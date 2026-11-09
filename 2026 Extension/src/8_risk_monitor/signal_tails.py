"""Warning signs per borrower-quarter, ranked within the quarter (crowding within size groups,
disagreement on value, PIK half / switch), their distributions (slides 12-14), and each BDC's exposure
to borrowers with a distress sign (slide 15). Definitions in docs/12_risk_monitor.md."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import polars as pl

from charts import SURFACE, INK, MUTED, GRID, BLUE, ORANGE
from paths import REFERENCE_DIR

FULL_HISTORY_FROM = "2024Q1"   # first quarter whose past-year switch window is wholly inside the data
SWITCH_WINDOW = 4              # quarters: this one and the three before
TOP_SHARE = 0.10               # flag: in the top 10% of the distribution
SIZE_GROUPS = 5                # crowding is ranked within fifths of BDC-held par
MANAGER_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 10 ** 6, "6+")]
PIK_HALF = 0.5
MIN_POSITIONS = 20
EXPOSURE_LINE = 0.20      # "more than 20% of the portfolio in flagged borrowers"
SINGLE_NAME_LINE = 0.05   # "one flagged borrower worth more than 5% of the portfolio"
BUCKET_COLORS = ["#c9ddf5", "#86b4ea", "#3987e5", "#0d366b"]   # 2, 3, 4-5, 6+ managers (light to dark)

SIGNS = {
    "crowding": "More managers than 9 in 10 borrowers of its size",
    "disagreement": "Its lenders disagree more than for 9 in 10 shared borrowers",
    "pik_half": "More than half of its interest is paid in kind",
    "pik_switch_q": "One of its loans switched to PIK this quarter",
    "pik_switched": "One of its loans switched to PIK in the past year",
}
DISTRESS_SIGNS = ("disagreement", "pik_switched")


def _borrower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _flag(value: pd.Series, mask: pd.Series) -> pd.Series:
    """1.0 / 0.0 where the signal exists, NaN where it does not."""
    return mask.astype(float).where(value.notna())


def _upper_share(value: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """For each row: the share of rows in its group whose value is at least as high (1/n for
    the top row). A row is in the top 10% when this is at most 0.10; ties are never split, so
    whole-number counts flag a little under 10%."""
    d = pd.DataFrame({"v": value}).assign(**{f"g{i}": g for i, g in enumerate(groups)})
    d = d[d["v"].notna()]
    keys = [c for c in d.columns if c.startswith("g")]
    r_min = d.groupby(keys)["v"].rank(method="min")
    n = d.groupby(keys)["v"].transform("size")
    return ((n - r_min + 1) / n).reindex(value.index)


def size_groups(b: pd.DataFrame) -> pd.Series:
    """Fifth of BDC-held par within the quarter (1 = smallest)."""
    return b.groupby("cal_q")["par"].transform(
        lambda s: pd.qcut(s.rank(method="first"), SIZE_GROUPS, labels=False) + 1)


# ------------------------------------------------------------------ borrower warning signs

def warning_flags(b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """b: the borrower-quarter panel. One row per borrower-quarter with a 0/1 column per sign
    (NaN where the sign cannot be measured), the upper-tail shares behind the two ranked signs,
    and `distress` (any distress sign). Returns the frame and the rules."""
    thr = {"crowding": TOP_SHARE, "disagreement": TOP_SHARE,
           "pik_half": PIK_HALF, "pik_switch_q": 0.0, "pik_switched": 0.0}
    f = b[["borrower", "cal_q", "sector", "par", "fv", "price", "C1_families", "V1_range",
           "pik_share", "pik_par_share", "toggle_par_share"]].copy()
    f["size_group"] = size_groups(b)
    f["crowding_upper_share"] = _upper_share(b["C1_families"], [b["cal_q"], f["size_group"]])
    f["disagreement_upper_share"] = _upper_share(b["V1_range"], [b["cal_q"]])
    f["crowding"] = _flag(f["crowding_upper_share"], f["crowding_upper_share"] <= TOP_SHARE)
    f["disagreement"] = _flag(f["disagreement_upper_share"], f["disagreement_upper_share"] <= TOP_SHARE)
    f["pik_half"] = _flag(b["pik_share"], b["pik_share"] > thr["pik_half"])
    first_q = b["cal_q"].min()
    f["pik_switch_q"] = (b["toggle_par_share"].fillna(0) > 0).astype(float).where(b["cal_q"] > first_q)
    f["pik_switched"] = _switched_within(b, SWITCH_WINDOW).where(b["cal_q"] >= FULL_HISTORY_FROM)
    f["distress"] = f[list(DISTRESS_SIGNS)].fillna(0).max(axis=1)
    return f, thr


def _switched_within(b: pd.DataFrame, quarters: int) -> pd.Series:
    """1.0 if one of the borrower's loans switched to PIK in this calendar quarter or the
    `quarters` - 1 before it (a quarter the borrower is absent counts as no switch)."""
    tog = b[["borrower", "qsort"]].assign(t=(b["toggle_par_share"].fillna(0) > 0).astype(float))
    hit = np.zeros(len(b))
    for lag in range(quarters):
        m = b[["borrower", "qsort"]].merge(tog.assign(qsort=tog["qsort"] + lag), on=["borrower", "qsort"], how="left")
        hit = np.maximum(hit, m["t"].fillna(0).to_numpy())
    return pd.Series(hit, index=b.index)


def warning_tail(f: pd.DataFrame) -> pd.DataFrame:
    """Per sign and quarter: borrowers where the sign is measured, borrowers flagged, their
    share, and the share of BDC-held par they carry."""
    out = []
    for s in list(SIGNS) + ["distress"]:
        d = f[f[s].notna()]
        g = d.groupby("cal_q")
        t = pd.DataFrame({"borrowers": g.size(), "flagged": g[s].sum(),
                          "par_flagged": (d[s] * d["par"]).groupby(d["cal_q"]).sum(), "par": g["par"].sum()})
        t["share"] = t["flagged"] / t["borrowers"]
        t["par_share"] = t["par_flagged"] / t["par"]
        out.append(t.reset_index().assign(sign=s))
    return pd.concat(out, ignore_index=True)


def _big_axes(ax, title, subtitle, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=15, color=INK, pad=30, weight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=11, color=MUTED, va="bottom")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
    ax.tick_params(colors=MUTED, labelsize=10.5, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _pct_axis(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def _bucket(n: pd.Series) -> pd.Series:
    """Manager-count bucket label (MANAGER_BUCKETS) for each value."""
    out = pd.Series(pd.NA, index=n.index, dtype="object")
    for lo, hi, lab in MANAGER_BUCKETS:
        out[(n >= lo) & (n <= hi)] = lab
    return out


def crowding_distribution(b: pd.DataFrame) -> pd.DataFrame:
    """Per quarter and manager-count bucket (1, 2, 3, 4-5, 6+): borrowers and their share of all
    borrowers, BDC-held par and its share of all BDC-held par."""
    d = b[b["C1_families"].notna() & (b["par"] > 0)].assign(bucket=lambda x: _bucket(x["C1_families"]))
    g = d.groupby(["cal_q", "bucket"]).agg(borrowers=("borrower", "size"), par=("par", "sum")).reset_index()
    g = g.join(d.groupby("cal_q").agg(all_borrowers=("borrower", "size"), all_par=("par", "sum")), on="cal_q")
    g["share_borrowers"] = g["borrowers"] / g["all_borrowers"]
    g["share_par"] = g["par"] / g["all_par"]
    return g


def plot_crowding_distribution(dist: pd.DataFrame, path):
    """Two panels, one bar per quarter stacked by number of managers (2, 3, 4-5, 6+; borrowers
    with one manager are the rest): left as a share of borrowers, right as a share of BDC-held
    debt. No cutoff: the whole tail is shown."""
    labels = [lab for _, _, lab in MANAGER_BUCKETS[1:]]
    quarters = sorted(dist["cal_q"].unique())
    x = np.arange(len(quarters))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=SURFACE)
    for ax, col, title, ylabel in [(axes[0], "share_borrowers", "By number of borrowers", "% of borrowers"),
                                   (axes[1], "share_par", "By dollars of BDC-held debt", "% of BDC-held debt")]:
        p = 100 * dist.pivot(index="cal_q", columns="bucket", values=col).reindex(quarters).fillna(0)
        bottom = np.zeros(len(quarters))
        for lab, color in zip(labels, BUCKET_COLORS):
            ax.bar(x, p[lab], bottom=bottom, width=0.7, color=color, edgecolor=SURFACE, linewidth=1.2,
                   label=f"{lab} managers")
            bottom += p[lab].to_numpy()
        for i in (0, len(x) - 1):
            ax.text(i, bottom[i], f"{bottom[i]:.0f}%", ha="center", va="bottom", fontsize=11.5, color=INK,
                    weight="bold" if i else "normal")
        six = p["6+"]
        _big_axes(ax, title, f"6+ managers: {six.iat[0]:.1f}% in {quarters[0]}, {six.iat[-1]:.1f}% in {quarters[-1]}",
                  ylabel)
        ax.title.set_fontsize(14)
        _pct_axis(ax)
        ax.set_ylim(0, bottom.max() * 1.18)
        ax.set_xticks(x, quarters, rotation=45)
    handles, labs = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labs[::-1], frameon=False, fontsize=10.5, loc="upper left", labelcolor=INK)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.15, wspace=0.16)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _weighted_quantile(x: pd.Series, w: pd.Series, q: float) -> float:
    o = np.argsort(x.to_numpy())
    xs, ws = x.to_numpy()[o], w.to_numpy()[o]
    return float(xs[np.searchsorted(np.cumsum(ws) / ws.sum(), q)])


def disagreement_percentiles(b: pd.DataFrame) -> pd.DataFrame:
    """Per quarter, over borrowers held by 2+ manager families: the median, 90th and 95th
    percentile of the gap between lenders' values (V1_range, share of par), across borrowers
    and across dollars of BDC-held debt (par-weighted: q of the dollars sit with a smaller gap)."""
    d = b[b["V1_range"].notna() & (b["par"] > 0)]
    rows = []
    for q, g in d.groupby("cal_q"):
        r = {"cal_q": q, "borrowers": len(g), "par": g["par"].sum()}
        for p in (0.5, 0.9, 0.95):
            r[f"p{int(100 * p)}_borrowers"] = float(g["V1_range"].quantile(p))
            r[f"p{int(100 * p)}_dollars"] = _weighted_quantile(g["V1_range"], g["par"], p)
        rows.append(r)
    return pd.DataFrame(rows)


def plot_disagreement_percentiles(t: pd.DataFrame, path):
    """One panel: the median and 90th percentile of the gap each quarter, in cents per dollar
    owed, across borrowers (blue) and across dollars of BDC-held debt (orange). Labelled at the
    line ends; no legend."""
    t = t.sort_values("cal_q").reset_index(drop=True)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    lines = [("p90_dollars", ORANGE, 2.8, "90th percentile, dollars"),
             ("p90_borrowers", BLUE, 2.8, "90th percentile, borrowers"),
             ("p50_dollars", ORANGE, 1.4, "Median, dollars"),
             ("p50_borrowers", BLUE, 1.4, "Median, borrowers")]
    ends = []
    for col, color, lw, label in lines:
        y = 100 * t[col]
        ax.plot(x, y, color=color, linewidth=lw, marker="o", markersize=5 if lw > 2 else 3.5,
                alpha=1 if lw > 2 else 0.75, zorder=3)
        ends.append((y.iat[-1], f"{label}  {y.iat[-1]:.1f}¢", color, lw > 2))
    _big_axes(ax, "How far apart lenders value the same loan",
              "Gap between the highest and lowest value, cents per dollar owed; borrowers held by 2+ managers",
              "cents per dollar owed")
    ax.set_ylim(0, 100 * t[["p90_borrowers", "p90_dollars"]].to_numpy().max() * 1.15)
    lo, hi = ax.get_ylim()
    ys = sorted(range(len(ends)), key=lambda i: ends[i][0])
    pos = [e[0] for e in ends]
    for a, b_ in zip(ys, ys[1:]):          # keep end labels at least 6% of the axis apart
        if pos[b_] - pos[a] < (hi - lo) * 0.06:
            pos[b_] = pos[a] + (hi - lo) * 0.06
    for (yv, text, color, bold), yl in zip(ends, pos):
        ax.text(len(x) - 1 + 0.35, yl, text, va="center", fontsize=10.5, color=INK, weight="bold" if bold else "normal")
        ax.plot([len(x) - 1 + 0.08, len(x) - 1 + 0.28], [yv, yl], color=color, linewidth=1.2)
    ax.set_xlim(-0.4, len(x) + 6.0)
    ax.set_xticks(x, t["cal_q"], rotation=45)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}¢")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.16)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_pik_tail(tail: pd.DataFrame, path) -> pd.DataFrame:
    """Two PIK series on one axis: borrowers paying more than half their interest in kind (all
    quarters), and borrowers with a loan that switched to PIK that quarter (from the second
    quarter of data: a switch is seen against the quarter before)."""
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    rows = []
    quarters = sorted(tail["cal_q"].unique())
    pos = {q: i for i, q in enumerate(quarters)}
    for sign, color, label in [("pik_half", BLUE, "Pays more than half its interest in kind"),
                               ("pik_switch_q", ORANGE, "Switched to PIK this quarter")]:
        t = tail[tail["sign"] == sign].sort_values("cal_q")
        xs, ys = t["cal_q"].map(pos), 100 * t["share"]
        ax.plot(xs, ys, color=color, linewidth=2.4, marker="o", markersize=6, label=label)
        ax.annotate(f"{ys.iat[-1]:.1f}%", (xs.iat[-1], ys.iat[-1]), xytext=(8, 0), textcoords="offset points",
                    fontsize=11, color=INK, weight="bold", va="center")
        ax.annotate(f"{ys.iat[0]:.1f}%", (xs.iat[0], ys.iat[0]), xytext=(0, 10), textcoords="offset points",
                    fontsize=10.5, color=MUTED, ha="center")
        rows.append(t)
    _big_axes(ax, "Borrowers paying interest with more debt",
              "% of borrowers with BDC debt, each quarter", "% of borrowers")
    _pct_axis(ax)
    ax.set_ylim(0, 6)
    ax.set_xlim(-0.5, len(quarters) - 0.2)
    ax.set_xticks(range(len(quarters)), quarters, rotation=45)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right", labelcolor=INK)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.16)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ BDC exposure

def bdc_names(path=REFERENCE_DIR / "bdc_type.csv") -> pd.Series:
    try:
        return pd.read_csv(path, usecols=["cik", "name"]).drop_duplicates("cik").set_index("cik")["name"]
    except (FileNotFoundError, ValueError):
        return pd.Series(dtype=str)


def bdc_exposure(rows: pl.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """One row per BDC-quarter: share of portfolio fair value in flagged borrowers (per sign and
    for any distress sign), the largest distress-flagged borrower and its weight, concentration
    (cost HHI), bdc_type and name."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "FV", "COST", "bdc_type"]).to_pandas()
    r = r[r["FV"] > 0]
    total = r.groupby(["cik", "cal_q"])["FV"].sum().rename("portfolio_fv")
    btype = r.groupby(["cik", "cal_q"])["bdc_type"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    r = r[r["borrower_canonical"].notna()].assign(borrower=lambda d: _borrower(d["borrower_canonical"]))
    w = r.groupby(["cik", "cal_q", "borrower"]).agg(fv=("FV", "sum"), cost=("COST", "sum")).reset_index()
    signs = list(SIGNS) + ["distress"]
    w = w.merge(f[["borrower", "cal_q"] + signs], on=["borrower", "cal_q"], how="left")
    w["weight"] = w["fv"] / w.set_index(["cik", "cal_q"]).index.map(total).to_numpy()
    g = w.groupby(["cik", "cal_q"])
    e = pd.DataFrame({"n_borrowers": g["borrower"].nunique()})
    for s in signs:
        e[f"exposure_{s}"] = (w["weight"] * w[s].fillna(0)).groupby([w["cik"], w["cal_q"]]).sum()
    flagged = w[w["distress"] == 1].sort_values("weight", ascending=False).drop_duplicates(["cik", "cal_q"])
    top = flagged.set_index(["cik", "cal_q"])
    e["top_flagged_weight"] = top["weight"].reindex(e.index).fillna(0.0)
    e["top_flagged"] = top["borrower"].reindex(e.index)
    c = w[w["cost"] > 0]
    share = c["cost"] / c.groupby(["cik", "cal_q"])["cost"].transform("sum")
    e["concentration"] = (share ** 2).groupby([c["cik"], c["cal_q"]]).sum()
    e["effective_n"] = 1 / e["concentration"]
    e = e.join(total).join(btype.rename("bdc_type")).reset_index()
    e["name"] = e["cik"].map(bdc_names())
    return e[e["n_borrowers"] >= MIN_POSITIONS].reset_index(drop=True)


def exposure_by_quarter(e: pd.DataFrame) -> pd.DataFrame:
    """Per quarter: BDCs, the median exposure, the share of BDCs above EXPOSURE_LINE, and the
    BDCs whose largest flagged borrower is above SINGLE_NAME_LINE."""
    return e.groupby("cal_q").agg(
        bdcs=("cik", "size"), median_exposure=("exposure_distress", "median"),
        over_line=("exposure_distress", lambda s: int((s > EXPOSURE_LINE).sum())),
        share_over_line=("exposure_distress", lambda s: (s > EXPOSURE_LINE).mean()),
        single_name_over=("top_flagged_weight", lambda s: int((s > SINGLE_NAME_LINE).sum())),
        share_single_name_over=("top_flagged_weight", lambda s: (s > SINGLE_NAME_LINE).mean()),
    ).reset_index()


def plot_exposure(e: pd.DataFrame, path) -> pd.DataFrame:
    """Left: share of BDCs with more than EXPOSURE_LINE of the portfolio in borrowers with a
    distress sign. Right: BDCs whose single largest flagged borrower is above SINGLE_NAME_LINE.
    From FULL_HISTORY_FROM, when both distress signs can be measured."""
    s = exposure_by_quarter(e)
    s = s[s["cal_q"] >= FULL_HISTORY_FROM].reset_index(drop=True)
    x = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    panels = [(s["over_line"], "count", f"BDCs with over {EXPOSURE_LINE:.0%} of portfolio flagged",
               "BDCs", lambda v: f"{v:.0f}"),
              (s["single_name_over"], "count", f"BDCs with one flagged borrower over {SINGLE_NAME_LINE:.0%}",
               "BDCs", lambda v: f"{v:.0f}")]
    for ax, (y, kind, title, ylabel, fmt) in zip(axes, panels):
        ax.bar(x, y, width=0.68, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        for i in (0, len(s) - 1):
            ax.text(i, y.iat[i], fmt(y.iat[i]), ha="center", va="bottom", fontsize=11.5, color=INK,
                    weight="bold" if i else "normal")
        _big_axes(ax, title, f"BDCs with {MIN_POSITIONS}+ borrowers, {s['cal_q'].iat[0]}–{s['cal_q'].iat[-1]}", ylabel)
        if kind == "share":
            _pct_axis(ax)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_ylim(0, y.max() * 1.2)
        ax.set_xticks(x, s["cal_q"], rotation=45)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.15, wspace=0.18)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return s
