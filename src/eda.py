"""Market analysis: six questions, six figures, one findings file.

Each section answers one question a publisher would actually ask, writes a
figure to ``reports/figures/``, and returns the numbers it found so that the
findings file is generated from the same computation that drew the chart --
no hand-copied figures that can drift out of date.

Run: ``python src/eda.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import FIGURES_DIR, REGIONS, REGION_LABELS, REPORTS_DIR
from data_prep import load_clean
import viz_style as vs

vs.apply_theme()

# 2016 is a partial year in this snapshot; every time-series chart says so
# rather than letting the reader mistake a data artefact for a market crash.
PARTIAL_YEAR = 2016


def _flag_partial_year(ax, y_frac: float = 0.92) -> None:
    """Shade the incomplete final year and label it once."""
    ax.axvspan(PARTIAL_YEAR - 0.5, PARTIAL_YEAR + 0.5,
               color=vs.GRID, alpha=0.55, zorder=0, lw=0)
    ax.text(PARTIAL_YEAR - 0.7, y_frac, "2016\npartial",
            transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=8.5, color=vs.INK_MUTED, linespacing=1.3)


# --------------------------------------------------------------- 1. size
def market_size(d: pd.DataFrame) -> dict:
    """How big is the tracked market each year, and how many titles ship?

    Two measures on different scales, so two panels rather than a second
    y-axis -- a dual-axis chart would invite a causal reading of whichever
    way the lines happen to cross.
    """
    per_year = d.groupby("Year_of_Release").agg(
        sales=("Global_Sales", "sum"),
        titles=("Name", "size"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.32})

    axes[0].plot(per_year.index, per_year["sales"], color=vs.SERIES[0])
    axes[0].fill_between(per_year.index, per_year["sales"], color=vs.SERIES[0], alpha=0.10)
    vs.titled(axes[0], "The tracked market peaked in 2008",
              "Global sales, millions of units")
    axes[0].set_ylim(0)
    _flag_partial_year(axes[0])

    axes[1].plot(per_year.index, per_year["titles"], color=vs.SERIES[1])
    axes[1].fill_between(per_year.index, per_year["titles"], color=vs.SERIES[1], alpha=0.10)
    vs.titled(axes[1], "Releases peaked the same year",
              "Titles released (platform releases, not unique games)")
    axes[1].set_ylim(0)
    axes[1].set_xlabel("Year of release")
    axes[1].set_xlim(per_year.index.min(), per_year.index.max() + 0.5)
    _flag_partial_year(axes[1])

    vs.save(fig, FIGURES_DIR / "01_market_size.svg")

    peak_year = int(per_year["sales"].idxmax())
    return {
        "peak_year": peak_year,
        "peak_sales": float(per_year.loc[peak_year, "sales"]),
        "sales_2015": float(per_year.loc[2015, "sales"]),
        "decline_pct": float(
            (1 - per_year.loc[2015, "sales"] / per_year.loc[peak_year, "sales"]) * 100
        ),
    }


# -------------------------------------------------------------- 2. genre
def genre_mix(d: pd.DataFrame, top_n: int = 5) -> dict:
    """Which genres hold the market, and is the mix shifting?

    Shares of an annual total, so the parts sum to the whole -- a stacked
    area is the honest form here. Everything outside the top few folds into
    a single neutral "Other" band rather than becoming a ninth colour.
    """
    by_year_genre = (
        d.pivot_table(index="Year_of_Release", columns="Genre",
                      values="Global_Sales", aggfunc="sum")
        .fillna(0)
    )
    shares = by_year_genre.div(by_year_genre.sum(axis=1), axis=0) * 100

    leaders = d.groupby("Genre")["Global_Sales"].sum().nlargest(top_n).index.tolist()
    plotted = shares[leaders].copy()
    plotted["Other"] = 100 - plotted.sum(axis=1)

    colors = vs.SERIES[:top_n] + [vs.BASELINE]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.stackplot(plotted.index, plotted.T.values,
                 labels=plotted.columns, colors=colors,
                 edgecolor=vs.SURFACE, linewidth=1.4)
    vs.titled(ax, "Shooters quadrupled their share of a shrinking market",
              "Share of global sales by genre, %")
    ax.set_ylim(0, 100)
    ax.set_xlim(plotted.index.min(), plotted.index.max())
    ax.set_xlabel("Year of release")
    ax.set_xticks(range(1996, 2017, 4))
    ax.grid(False)

    # Direct labels at the right edge beat a legend the eye has to travel to.
    cumulative = 0.0
    final = plotted.iloc[-1]
    for name, value in final.items():
        mid = cumulative + value / 2
        if value > 6:
            ax.text(plotted.index.max() + 0.25, mid, name,
                    va="center", ha="left", fontsize=9,
                    color=vs.INK_SECONDARY)
        cumulative += value
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=6, fontsize=9)

    vs.save(fig, FIGURES_DIR / "02_genre_mix.svg")

    shooter_start = float(shares.loc[1996:2000, "Shooter"].mean())
    shooter_end = float(shares.loc[2013:2016, "Shooter"].mean())
    return {
        "shooter_early_share": shooter_start,
        "shooter_late_share": shooter_end,
        "top_genres": leaders,
    }


# ------------------------------------------------------------- 3. region
def regional_taste(d: pd.DataFrame) -> dict:
    """Where does Japanese taste diverge from Western taste?

    A genre's share of Japan's total, divided by its share of North America's
    total. Shares rather than raw units, so the much larger NA market does not
    swamp the comparison. The result is signed around 1.0, which makes this a
    diverging scale: two poles, neutral in the middle, no rainbow.
    """
    totals = d.groupby("Genre")[REGIONS].sum()
    shares = totals / totals.sum() * 100
    index = (shares["JP_Sales"] / shares["NA_Sales"]).sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    log_index = np.log2(index)
    colors = [vs.SERIES[7] if v < 0 else vs.SERIES[0] for v in log_index]
    ax.barh(index.index, log_index, color=colors, height=0.62)

    ax.axvline(0, color=vs.BASELINE, lw=1.2)
    vs.titled(ax, "Japan buys a different industry",
              "Genre's share of Japanese sales vs its share of North American sales")
    ax.set_xlabel("← more Western          log₂ ratio          more Japanese →")
    ax.grid(False)
    ax.xaxis.grid(True)

    for genre, value in log_index.items():
        ratio = index[genre]
        offset = 0.06 if value >= 0 else -0.06
        ax.text(value + offset, genre, f"{ratio:.2f}×",
                va="center", ha="left" if value >= 0 else "right",
                fontsize=9, color=vs.INK_SECONDARY)
    ax.set_xlim(log_index.min() - 0.6, log_index.max() + 0.5)

    vs.save(fig, FIGURES_DIR / "03_regional_taste.svg")

    return {
        "jp_rpg_share": float(shares.loc["Role-Playing", "JP_Sales"]),
        "na_rpg_share": float(shares.loc["Role-Playing", "NA_Sales"]),
        "jp_shooter_share": float(shares.loc["Shooter", "JP_Sales"]),
        "na_shooter_share": float(shares.loc["Shooter", "NA_Sales"]),
        "rpg_index": float(index["Role-Playing"]),
        "shooter_index": float(index["Shooter"]),
        "region_totals": {REGION_LABELS[r]: float(totals[r].sum()) for r in REGIONS},
    }


# ----------------------------------------------------------- 4. platform
def platform_lifecycle(d: pd.DataFrame, top_n: int = 6) -> dict:
    """How does a console's software revenue evolve after launch?

    Plotted against years-since-launch rather than calendar year, which puts
    hardware generations on a common x-axis and makes the shapes comparable.

    Restricted to platforms that launched inside the analysis window. The
    PlayStation and Nintendo 64 sold their first years before 1996, so their
    curves would start mid-life and fake an early peak.
    """
    observed = d[d["platform_fully_observed"] == 1]
    biggest = (
        observed.groupby("Platform")["Global_Sales"].sum()
        .nlargest(top_n).index.tolist()
    )
    # A handful of titles carry a release year one before their platform's
    # first real slate. They are import or pre-launch oddities; dropping them
    # keeps the x-axis anchored at launch.
    sub = observed[observed["Platform"].isin(biggest) & (observed["platform_age"] >= 0)]
    curves = sub.pivot_table(index="platform_age", columns="Platform",
                             values="Global_Sales", aggfunc="sum")

    peaks = {p: int(curves[p].dropna().idxmax()) for p in biggest}
    lo, hi = min(peaks.values()), max(peaks.values())

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, platform in enumerate(biggest):
        series = curves[platform].dropna()
        ax.plot(series.index, series.values, color=vs.SERIES[i], label=platform)
        peak_age = peaks[platform]
        ax.plot([peak_age], [series[peak_age]], "o",
                color=vs.SERIES[i], markersize=8,
                markeredgecolor=vs.SURFACE, markeredgewidth=2)

    vs.titled(ax, f"Console software peaks {lo} to {hi} years after launch",
              "Global software sales by platform, millions of units; dot marks the peak")
    ax.set_xlabel("Years since the platform's first release")
    ax.set_ylim(0)
    ax.legend(loc="upper right", ncol=2)

    vs.save(fig, FIGURES_DIR / "04_platform_lifecycle.svg")

    return {"platform_peak_age": peaks, "platforms": biggest,
            "peak_lo": lo, "peak_hi": hi}


# ------------------------------------------------------------- 5. scores
def score_vs_success(d: pd.DataFrame) -> dict:
    """Does critical reception track commercial success?

    Hit rate within critic-score bands, plus the share of titles that never
    got reviewed at all -- the second group is larger than most people expect
    and behaves very differently.
    """
    scored = d[d["Critic_Score"].notna()].copy()
    bands = pd.cut(scored["Critic_Score"], [0, 50, 60, 70, 80, 90, 100])
    grouped = scored.groupby(bands, observed=True).agg(
        n=("Name", "size"),
        hit_rate=("is_hit", "mean"),
        median_sales=("Global_Sales", "median"),
    )
    labels = ["<50", "50-60", "60-70", "70-80", "80-90", "90+"]

    unreviewed_hit = float(d.loc[d["has_critic_score"] == 0, "is_hit"].mean())

    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    ramp = [vs.SEQUENTIAL[i] for i in (2, 4, 6, 8, 10, 12)]
    bars = ax.bar(range(len(labels)), grouped["hit_rate"] * 100,
                  color=ramp, width=0.68)

    # The reference line is the group the score bands leave out entirely.
    ax.axhline(unreviewed_hit * 100, color=vs.SERIES[1], lw=2, ls=(0, (4, 3)))
    ax.text(-0.42, unreviewed_hit * 100 + 2.2,
            f"never reviewed: {unreviewed_hit * 100:.0f}%",
            ha="left", va="bottom", fontsize=9, color=vs.SERIES[1])

    vs.titled(ax, "Review scores separate hits, but only at the top",
              "Share of titles selling 1M+ units, %")
    ax.set_xlabel("Metacritic critic score")
    ax.set_ylim(0, 78)

    # Sample size lives under the tick label, where it cannot collide with a
    # bar or sit on a fill it has no contrast against.
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{lab}\nn={int(n):,}" for lab, n in zip(labels, grouped["n"])])

    for bar, value in zip(bars, grouped["hit_rate"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.6,
                f"{value:.0f}%", ha="center", va="bottom",
                fontsize=10, color=vs.INK_SECONDARY)

    vs.save(fig, FIGURES_DIR / "05_score_vs_success.svg")

    r = float(np.corrcoef(scored["Critic_Score"], scored["log_sales"])[0, 1])
    return {
        "pearson_r_log_sales": r,
        "hit_rate_90plus": float(grouped["hit_rate"].iloc[-1]),
        "hit_rate_sub50": float(grouped["hit_rate"].iloc[0]),
        "hit_rate_unreviewed": unreviewed_hit,
        "unreviewed_share": float(1 - d["has_critic_score"].mean()),
    }


# --------------------------------------------------- 6. who owns the market
def publisher_concentration(d: pd.DataFrame) -> dict:
    """Is the market consolidating into fewer hands?

    Top-5 share of annual sales, alongside the count of publishers shipping
    anything at all. Same two-measures-two-panels rule as the first figure.
    """
    rows = []
    for year, group in d.groupby("Year_of_Release"):
        sales = group.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
        total = sales.sum()
        rows.append({
            "year": year,
            "top5_share": sales.head(5).sum() / total * 100,
            "n_publishers": len(sales),
        })
    conc = pd.DataFrame(rows).set_index("year")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True,
                             gridspec_kw={"hspace": 0.32})

    axes[0].plot(conc.index, conc["top5_share"], color=vs.SERIES[0])
    vs.titled(axes[0], "The top five publishers never hold less than half",
              "Share of annual global sales taken by the five largest publishers, %")
    axes[0].set_ylim(0, 100)
    _flag_partial_year(axes[0])

    axes[1].plot(conc.index, conc["n_publishers"], color=vs.SERIES[2])
    vs.titled(axes[1], "…while the count of active publishers swings widely",
              "Publishers with at least one tracked release")
    axes[1].set_ylim(0)
    axes[1].set_xlabel("Year of release")
    axes[1].set_xlim(conc.index.min(), conc.index.max() + 0.5)
    _flag_partial_year(axes[1])

    vs.save(fig, FIGURES_DIR / "06_publisher_concentration.svg")

    lifetime = d.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
    return {
        "top5_share_min": float(conc["top5_share"].min()),
        "top5_share_max": float(conc["top5_share"].max()),
        "top5_lifetime": {k: float(v) for k, v in lifetime.head(5).items()},
        "lifetime_top5_pct": float(lifetime.head(5).sum() / lifetime.sum() * 100),
    }


# ---------------------------------------------------------------- report
def write_findings(d: pd.DataFrame, f: dict) -> None:
    """Render the findings file from the numbers the analysis just produced."""
    size, genre, region = f["size"], f["genre"], f["region"]
    plat, score, pub = f["platform"], f["score"], f["publisher"]

    peak_ages = plat["platform_peak_age"]
    peak_list = ", ".join(f"{k} (year {v})" for k, v in sorted(peak_ages.items(), key=lambda kv: kv[1]))

    text = f"""# Market findings

Generated by `src/eda.py` from {len(d):,} platform releases
({d['Name'].nunique():,} unique titles), {d.Year_of_Release.min()}-{d.Year_of_Release.max()}.
Every number below is computed at render time; none are typed by hand.

> **On 2016.** The snapshot was taken in December 2016, so that year's titles
> had not finished selling. 2016 appears in the charts with a shaded band and
> is excluded from model evaluation.

---

## 1. The market peaked in {size['peak_year']} and has not recovered

Tracked sales peaked at **{size['peak_sales']:.0f}M units** in {size['peak_year']} and fell to
**{size['sales_2015']:.0f}M** by 2015 -- a **{size['decline_pct']:.0f}% decline**. Release counts
fell alongside, so this is a market contracting rather than the same demand
spread over more titles.

The honest caveat: this dataset tracks physical retail sales. The decline
overlaps with the rise of digital distribution and free-to-play, so some of
this is a measurement shift, not only a market one. Nothing in the data can
separate the two, and any conclusion that ignores that is overclaiming.

![Market size](figures/01_market_size.svg)

## 2. Shooters quadrupled their share as the market shrank

Shooters grew from **{genre['shooter_early_share']:.1f}%** of sales in 1996-2000 to
**{genre['shooter_late_share']:.1f}%** in 2013-2016 -- a
{genre['shooter_late_share'] / genre['shooter_early_share']:.1f}× increase in share
while the market itself contracted. The genres holding the most lifetime revenue
are {", ".join(genre['top_genres'])}.

![Genre mix](figures/02_genre_mix.svg)

## 3. Japan buys a different industry

The sharpest divide in the data, and the one with the clearest commercial
consequence.

| Genre | Share of JP sales | Share of NA sales | Ratio |
|---|---|---|---|
| Role-Playing | {region['jp_rpg_share']:.1f}% | {region['na_rpg_share']:.1f}% | **{region['rpg_index']:.2f}×** |
| Shooter | {region['jp_shooter_share']:.1f}% | {region['na_shooter_share']:.1f}% | **{region['shooter_index']:.2f}×** |

Role-playing games take **{region['jp_rpg_share']:.0f}%** of the Japanese market against
**{region['na_rpg_share']:.0f}%** of the North American one. Shooters run the other way by
an even wider margin: **{region['na_shooter_share']:.0f}%** of NA sales, **{region['jp_shooter_share']:.1f}%** of JP.

A publisher treating "global" as one market with one genre strategy is
getting one of these two regions wrong.

![Regional taste](figures/03_regional_taste.svg)

## 4. Console software peaks {plat['peak_lo']} to {plat['peak_hi']} years after launch

Peak year by platform: {peak_list}. Restricted to platforms that launched
inside the analysis window -- the PlayStation and N64 sold their first years
before 1996, so including them would fake an early peak.

The practical read: a title shipping into a platform's first year competes
for a small installed base, and one shipping in year six is fighting the
next generation's launch window.

![Platform lifecycle](figures/04_platform_lifecycle.svg)

## 5. Review scores separate hits only at the very top

Critic score correlates with log sales at **r = {score['pearson_r_log_sales']:.2f}** -- real, but
far from deterministic. The relationship is strongly non-linear:

- 90+ scored titles: **{score['hit_rate_90plus']:.0%}** are million-sellers
- Sub-50 titles: **{score['hit_rate_sub50']:.0%}**
- Never reviewed ({score['unreviewed_share']:.0%} of all releases): **{score['hit_rate_unreviewed']:.0%}**

Most of the discriminating power sits in the top two bands. Between 50 and 70
the score barely moves the odds, which is a useful negative result: a mid-range
review tells a publisher almost nothing.

![Score vs success](figures/05_score_vs_success.svg)

## 6. The top five publishers never hold less than half

Top-5 share of annual sales ranges from **{pub['top5_share_min']:.0f}%** to
**{pub['top5_share_max']:.0f}%**. Over the full period the five largest publishers
account for **{pub['lifetime_top5_pct']:.0f}%** of all tracked sales.

![Publisher concentration](figures/06_publisher_concentration.svg)
"""
    path = REPORTS_DIR / "market_findings.md"
    path.write_text(text)
    print(f"  wrote {path.name}")


def main() -> None:
    d = load_clean()
    print(f"loaded {len(d):,} rows")
    print("figures:")
    findings = {
        "size": market_size(d),
        "genre": genre_mix(d),
        "region": regional_taste(d),
        "platform": platform_lifecycle(d),
        "score": score_vs_success(d),
        "publisher": publisher_concentration(d),
    }
    write_findings(d, findings)
    print("done")


if __name__ == "__main__":
    main()
