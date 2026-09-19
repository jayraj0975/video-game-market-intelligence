"""Hit prediction: can a million-seller be called before it ships?

The task is framed the way a publisher would face it. Given what is knowable
at greenlight -- platform, genre, rating, the studio's track record, the
franchise's history, and early critic reception -- rank a release slate by the
probability each title clears one million units.

Evaluation is deliberately unforgiving:

* **Temporal split.** Trained on 1996-2013, tested on the 2014-15 slate. The
  model never sees a year it is scored on.
* **Precision-recall, not accuracy.** Hits are 12% of releases, so a model
  that predicts "no" every time is 88% accurate and completely useless.
* **A decision metric.** ROC-AUC does not tell a publisher anything. "Fund the
  top 10% of the slate and you capture N% of the actual hits" does.

A leakage demonstration is included at the end: the same model with one
post-release feature added, to show how large and how silent the inflation is.

Run: ``python src/train.py``
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import (
    FIGURES_DIR,
    HIT_THRESHOLD_MUNITS,
    MODEL_MAX_YEAR,
    RANDOM_STATE,
    REPORTS_DIR,
    TRAIN_END_YEAR,
)
from data_prep import load_clean
from features import CATEGORICAL, FEATURES, NUMERIC, build_features, split_temporal
import viz_style as vs

vs.apply_theme()

# The share of the slate a publisher can actually fund. The headline decision
# metric is measured at this budget.
BUDGET_FRACTION = 0.10


# ------------------------------------------------------------- estimators
def logistic_pipeline(numeric: list[str]) -> Pipeline:
    """One-hot categoricals, median-imputed and scaled numerics.

    The imputer adds a missingness indicator rather than silently filling:
    "this game was never reviewed" is a real signal and the linear model
    should get to use it.
    """
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAL),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def gradient_boosting_pipeline(numeric: list[str]) -> Pipeline:
    """Gradient-boosted trees over ordinal-coded categoricals.

    Histogram gradient boosting handles missing values natively, so the
    unreviewed half of the data goes in untouched -- no imputation choice to
    defend, and the split finder decides for itself which side missing belongs
    on.
    """
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        ), CATEGORICAL),
        ("num", "passthrough", numeric),
    ])
    mask = [True] * len(CATEGORICAL) + [False] * len(numeric)
    return Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(
            categorical_features=mask,
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------- metrics
def capture_at_budget(y_true: np.ndarray, scores: np.ndarray,
                      fraction: float = BUDGET_FRACTION) -> dict:
    """What a fixed budget buys you.

    Rank the slate by predicted probability, take the top ``fraction``, and
    report how many of the real hits that selection contains -- against what
    picking at random would have given.
    """
    n_select = max(1, int(round(len(scores) * fraction)))
    # Ties are broken at random rather than by row order. Without this, a
    # constant-score baseline would be scored on how the rows happen to be
    # sorted in the file, which flatters or punishes it arbitrarily.
    jitter = np.random.default_rng(RANDOM_STATE).random(len(scores))
    top = np.lexsort((jitter, scores))[::-1][:n_select]
    hits_caught = int(y_true[top].sum())
    hits_total = int(y_true.sum())
    return {
        "n_selected": n_select,
        "hits_caught": hits_caught,
        "hits_total": hits_total,
        "recall": hits_caught / hits_total if hits_total else 0.0,
        "precision": hits_caught / n_select,
        "lift": (hits_caught / n_select) / y_true.mean() if y_true.mean() else 0.0,
    }


def evaluate(name: str, y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Every metric the report quotes, for one model."""
    cap = capture_at_budget(y_true, scores)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
        **{f"top{int(BUDGET_FRACTION * 100)}_{k}": v for k, v in cap.items()},
    }


# ---------------------------------------------------------------- figures
def plot_curves(y_test: np.ndarray, scored: dict[str, np.ndarray],
                base_rate: float) -> None:
    """Precision-recall beside ROC, with the no-skill line on both."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"wspace": 0.26})

    # The constant-score baseline is drawn as the dashed no-skill line, not as
    # a curve. Passed through `precision_recall_curve` it comes back as a
    # clean diagonal, which reads like a real model and is purely an artefact
    # of there being a single threshold.
    curves = {k: v for k, v in scored.items() if k != "Base rate"}

    for i, (name, scores) in enumerate(curves.items()):
        precision, recall, _ = precision_recall_curve(y_test, scores)
        axes[0].plot(recall, precision, color=vs.SERIES[i],
                     label=f"{name} (AP {average_precision_score(y_test, scores):.3f})")
        fpr, tpr, _ = roc_curve(y_test, scores)
        axes[1].plot(fpr, tpr, color=vs.SERIES[i],
                     label=f"{name} (AUC {roc_auc_score(y_test, scores):.3f})")

    axes[0].axhline(base_rate, color=vs.BASELINE, lw=1.5, ls=(0, (4, 3)))
    axes[0].text(0.98, base_rate + 0.015, f"no skill ({base_rate:.0%})",
                 ha="right", va="bottom", fontsize=9, color=vs.INK_MUTED)
    vs.titled(axes[0], "Precision-recall",
              f"Hits are {base_rate:.0%} of the slate, so this is the curve that matters")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right")

    axes[1].plot([0, 1], [0, 1], color=vs.BASELINE, lw=1.5, ls=(0, (4, 3)))
    vs.titled(axes[1], "ROC", "Shown for comparability; flattering on imbalanced data")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].legend(loc="lower right")

    vs.save(fig, FIGURES_DIR / "07_model_curves.svg")


def plot_importance(model: Pipeline, X_test: pd.DataFrame,
                    y_test: np.ndarray, top_n: int = 12) -> pd.DataFrame:
    """Permutation importance, measured on the held-out slate.

    Permutation rather than a tree's internal gain: gain is biased toward
    high-cardinality features, and measuring on the test set answers the
    question that matters -- which features carry weight on data the model
    has never seen.
    """
    result = permutation_importance(
        model, X_test, y_test,
        scoring="average_precision",
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"],
            color=vs.SERIES[0], height=0.62,
            error_kw={"ecolor": vs.INK_MUTED, "elinewidth": 1.2, "capsize": 3})
    vs.titled(ax, "Franchise history and press attention lead",
              "Drop in average precision when the feature is shuffled (test slate, 15 repeats)")
    ax.set_xlabel("Importance")
    ax.grid(False)
    ax.xaxis.grid(True)

    vs.save(fig, FIGURES_DIR / "08_feature_importance.svg")
    return imp.iloc[::-1]


def plot_calibration_and_capture(y_test: np.ndarray, scores: np.ndarray,
                                 calibrated: np.ndarray) -> dict:
    """Are the probabilities honest, and what do they buy?

    Left: predicted probability against observed frequency, before and after
    isotonic calibration. Right: the share of real hits captured as the budget
    widens -- the decision curve a publisher would actually read.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"wspace": 0.26})

    limit = 0.0
    for i, (label, s) in enumerate([("As trained", scores),
                                    ("Isotonic-calibrated", calibrated)]):
        prob_true, prob_pred = calibration_curve(y_test, s, n_bins=8, strategy="quantile")
        axes[0].plot(prob_pred, prob_true, "o-", color=vs.SERIES[i], label=label)
        limit = max(limit, float(prob_pred.max()), float(prob_true.max()))

    limit *= 1.05
    axes[0].plot([0, limit], [0, limit], color=vs.BASELINE, lw=1.5, ls=(0, (4, 3)),
                 zorder=0)
    axes[0].text(limit, limit, "perfect  ", ha="right", va="top",
                 fontsize=9, color=vs.INK_MUTED, rotation=45,
                 rotation_mode="anchor")
    vs.titled(axes[0], "Calibration",
              "Predicted probability vs observed hit frequency, 8 quantile bins")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left")

    fractions = np.arange(0.02, 1.01, 0.02)
    recalls = [capture_at_budget(y_test, scores, f)["recall"] for f in fractions]
    axes[1].plot(fractions * 100, np.array(recalls) * 100, color=vs.SERIES[0],
                 label="Model ranking")
    axes[1].plot([0, 100], [0, 100], color=vs.BASELINE, lw=1.5, ls=(0, (4, 3)),
                 label="Random selection")

    budget = capture_at_budget(y_test, scores)
    axes[1].plot([BUDGET_FRACTION * 100], [budget["recall"] * 100], "o",
                 color=vs.SERIES[1], markersize=9,
                 markeredgecolor=vs.SURFACE, markeredgewidth=2)
    axes[1].annotate(
        f"top {BUDGET_FRACTION:.0%} of the slate\ncaptures {budget['recall']:.0%} of hits",
        xy=(BUDGET_FRACTION * 100, budget["recall"] * 100),
        xytext=(BUDGET_FRACTION * 100 + 12, budget["recall"] * 100 - 6),
        fontsize=9.5, color=vs.INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": vs.INK_MUTED, "lw": 1},
    )

    vs.titled(axes[1], "What a budget buys",
              "Share of the 2014-15 slate funded vs share of real million-sellers caught, %")
    axes[1].set_xlabel("Share of slate funded, %")
    axes[1].set_ylabel("Hits captured, %")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 102)
    axes[1].legend(loc="lower right")

    vs.save(fig, FIGURES_DIR / "09_calibration_and_capture.svg")
    return budget


# ----------------------------------------------------------------- report
def write_report(rows: list[dict], leak: list[dict], ablation: list[dict],
                 imp: pd.DataFrame, budget: dict, n_train: int, n_test: int,
                 base_rate: float, best_name: str,
                 brier_before: float, brier_after: float) -> None:
    """Render the model report from the metrics just computed."""
    results = pd.DataFrame(rows)
    pct = int(BUDGET_FRACTION * 100)

    table = results[["model", "roc_auc", "pr_auc", "brier",
                     f"top{pct}_precision", f"top{pct}_recall", f"top{pct}_lift"]]
    table = table.rename(columns={
        "model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        "brier": "Brier", f"top{pct}_precision": f"Precision@{pct}%",
        f"top{pct}_recall": f"Recall@{pct}%", f"top{pct}_lift": "Lift",
    })

    best = results.loc[results["pr_auc"].idxmax()]
    leak_df = pd.DataFrame(leak)
    honest = leak_df.loc[leak_df["model"].str.contains("launch-time"), "pr_auc"].iloc[0]
    leaked = leak_df.loc[leak_df["model"].str.contains("post-release"), "pr_auc"].iloc[0]

    leak_table = (
        leak_df[["model", "roc_auc", "pr_auc"]]
        .rename(columns={"model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        .to_markdown(index=False, floatfmt=".3f")
    )

    abl_df = pd.DataFrame(ablation)
    pct_ = int(BUDGET_FRACTION * 100)
    abl_table = (
        abl_df[["model", "pr_auc", f"top{pct_}_recall"]]
        .rename(columns={"model": "Model", "pr_auc": "PR-AUC",
                         f"top{pct_}_recall": f"Recall@{pct_}%"})
        .to_markdown(index=False, floatfmt=".3f")
    )
    abl_full, abl_strict = abl_df["pr_auc"].iloc[0], abl_df["pr_auc"].iloc[1]

    top_features = ", ".join(f"`{f}`" for f in imp["feature"].head(5))

    text = f"""# Model report

Generated by `src/train.py`.

**Task.** Given only what is knowable before release, predict whether a title
will sell {HIT_THRESHOLD_MUNITS:.0f}M+ units.

**Split.** Train on 1996-{TRAIN_END_YEAR} ({n_train:,} releases), test on
{TRAIN_END_YEAR + 1}-{MODEL_MAX_YEAR} ({n_test:,} releases). Base rate in the test
slate: **{base_rate:.1%}**.

---

## Results

{table.to_markdown(index=False, floatfmt=".3f")}

*Lift is precision in the top {pct}% divided by the base rate: how many times better
than picking at random.*

The best model by PR-AUC is **{best['model']}** at **{best['pr_auc']:.3f}**, against a
no-skill floor of {base_rate:.3f}. ROC-AUC of {best['roc_auc']:.3f} looks more impressive
than the model is; on a slate that is {base_rate:.0%} hits, PR-AUC is the honest number.

![Model curves](figures/07_model_curves.svg)

## What it buys

Ranking the {n_test:,}-title slate and funding the top {pct}% -- {budget['n_selected']}
titles -- captures **{budget['hits_caught']} of the {budget['hits_total']} actual
million-sellers**, or **{budget['recall']:.0%}**. Picking {pct}% at random would have
returned about {BUDGET_FRACTION * budget['hits_total']:.0f}. That is a
**{budget['lift']:.1f}× lift**, with {budget['precision']:.0%} of the funded titles
turning out to be hits.

![Calibration and capture](figures/09_calibration_and_capture.svg)

## The probabilities needed fixing; the ranking did not

The left panel above shows the model as trained sitting well below the diagonal
at every point: it consistently claims a far higher chance than the observed
frequency. That is not a bug, it is the direct consequence of
`class_weight="balanced"`, which trains as though hits were half the slate. The
scores are good *rankings* and bad *probabilities*.

Isotonic regression fixes it without disturbing the order, so every decision
metric above is untouched while the Brier score falls from **{brier_before:.4f}**
to **{brier_after:.4f}**. The distinction matters in use: "rank the slate" and
"tell me this game's chance of clearing a million units" are different asks, and
only the second one needs the calibrated model.

## What drives it

{top_features}.

![Feature importance](figures/08_feature_importance.svg)

Two things stand out.

**Franchise history dominates.** How many hits the series has already produced
matters more than anything intrinsic to the game. The unglamorous truth of the
industry: a sequel to a million-seller is the safest bet on the board.

**How many critics reviewed it beats what they said.** `Critic_Count` scores
roughly {imp.set_index('feature').loc['Critic_Count', 'importance'] / imp.set_index('feature').loc['Critic_Score', 'importance']:.0f}× the importance of `Critic_Score`.
The count is not a quality measure -- it is a proxy for how much press attention
the title commanded, which tracks marketing spend and distribution. Being
reviewed widely predicts sales better than being reviewed well.

Genre and release year sit near zero. Both matter a great deal to *how much* a
game sells, and almost not at all to *whether* it crosses a million.

## Ablation: the model without the critic features

The critic columns are the least defensible part of the feature set, so here is
what the model is worth with them removed entirely -- track record, platform,
genre, rating and release scale only:

{abl_table}

PR-AUC falls from **{abl_full:.3f}** to **{abl_strict:.3f}**, still
{abl_strict / base_rate:.1f}× the no-skill floor of {base_rate:.3f}. The critic
features are doing real work, and the model is not merely a repackaging of them.

## The leakage check

The point of this section is that the inflation is large and produces no error
message. Adding `User_Count` -- the number of Metacritic user reviews, which
only exists *after* people have bought and played the game -- moves PR-AUC from
**{honest:.3f}** to **{leaked:.3f}**.

{leak_table}

Nothing about that second row looks wrong in isolation. It is a better model by
every metric on the page, and it is worthless, because at greenlight the feature
is a column of nulls. This is the single most common way a portfolio model
reports a score it could never reproduce in use.

## Honest limitations

1. **Physical retail only.** VGChartz tracks boxed sales. Digital, mobile and
   free-to-play are absent, so "hit" here means "retail hit", and the market
   decline in the descriptive analysis is partly a measurement artefact.
2. **Critic features are launch-window, not greenlight.** Review embargoes lift
   within days of release, so `Critic_Score` and `Critic_Count` are knowable at
   launch but not two years earlier when the money is actually committed. They
   are defensible for a launch-window forecast and a stretch for a greenlight
   one. `Critic_Count` in particular keeps accruing for weeks afterwards, so it
   is the feature in this set closest to the leakage line -- which makes its
   high importance score worth reading with suspicion rather than pride.
3. **Survivorship in the publisher features.** A publisher only has a track
   record if it survived long enough to build one.
4. **The threshold is a business choice.** 1M units is the industry's own
   shorthand, not a natural break in the distribution. Moving it moves every
   number here.
"""
    path = REPORTS_DIR / "model_report.md"
    path.write_text(text)
    print(f"  wrote {path.name}")


def main() -> None:
    d = build_features(load_clean())
    train, test = split_temporal(d, TRAIN_END_YEAR, MODEL_MAX_YEAR)

    X_train, y_train = train[FEATURES], train["is_hit"].to_numpy()
    X_test, y_test = test[FEATURES], test["is_hit"].to_numpy()
    base_rate = float(y_test.mean())

    print(f"train {len(train):,} rows ({y_train.mean():.1%} hits)")
    print(f"test  {len(test):,} rows ({base_rate:.1%} hits)")

    models = {
        "Base rate": DummyClassifier(strategy="prior"),
        "Logistic regression": logistic_pipeline(NUMERIC),
        "Gradient boosting": gradient_boosting_pipeline(NUMERIC),
    }

    rows, scored = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        scored[name] = scores
        rows.append(evaluate(name, y_test, scores))
        print(f"  {name:22s} PR-AUC {rows[-1]['pr_auc']:.3f}  ROC-AUC {rows[-1]['roc_auc']:.3f}")

    # The winner is whichever model actually scores best on the held-out
    # slate, not whichever one is fashionable.
    best_name = max(
        (r for r in rows if r["model"] != "Base rate"),
        key=lambda r: r["pr_auc"],
    )["model"]
    best_model = models[best_name]
    print(f"  best by PR-AUC: {best_name}")

    # ---- leakage demonstration ------------------------------------------
    # Same architecture, one post-release column added. Nothing else changes.
    # The count is log-scaled so the linear model can use it on equal terms
    # with the trees -- the point is to show the inflation, not to handicap it.
    leak_col = "log_user_count"
    train_leaky = train.assign(**{leak_col: np.log1p(train["User_Count"])})
    test_leaky = test.assign(**{leak_col: np.log1p(test["User_Count"])})
    leaky_cols = FEATURES + [leak_col]

    leaky = (logistic_pipeline if "Logistic" in best_name else gradient_boosting_pipeline)(
        NUMERIC + [leak_col]
    )
    leaky.fit(train_leaky[leaky_cols], y_train)
    leaky_scores = leaky.predict_proba(test_leaky[leaky_cols])[:, 1]
    leak = [
        evaluate(f"{best_name} (launch-time features only)", y_test, scored[best_name]),
        evaluate(f"{best_name} (+ post-release User_Count)", y_test, leaky_scores),
    ]
    print(f"  leakage check: PR-AUC {leak[0]['pr_auc']:.3f} -> {leak[1]['pr_auc']:.3f}")

    # ---- ablation: how much of the model is the critic features? ---------
    # The critic columns are the least defensible thing in the feature set
    # (see the limitations section), so the honest thing is to measure what
    # the model is worth without them rather than assert it survives.
    strict_numeric = [c for c in NUMERIC
                      if c not in ("Critic_Score", "Critic_Count", "has_critic_score")]
    strict_cols = CATEGORICAL + strict_numeric
    strict = logistic_pipeline(strict_numeric) if "Logistic" in best_name \
        else gradient_boosting_pipeline(strict_numeric)
    strict.fit(train[strict_cols], y_train)
    strict_scores = strict.predict_proba(test[strict_cols])[:, 1]
    ablation = [
        evaluate(f"{best_name} (all launch-time features)", y_test, scored[best_name]),
        evaluate(f"{best_name} (no critic features)", y_test, strict_scores),
    ]
    print(f"  ablation: PR-AUC {ablation[0]['pr_auc']:.3f} -> {ablation[1]['pr_auc']:.3f}")

    # ---- calibration -----------------------------------------------------
    # `class_weight="balanced"` buys ranking quality at the cost of honest
    # probabilities: it trains as though hits were half the slate, so every
    # predicted probability comes out far too high. Isotonic regression maps
    # the scores back onto observed frequencies without touching their order,
    # so the ranking -- and therefore every decision metric above -- is
    # unchanged.
    calibrated_model = CalibratedClassifierCV(
        clone(best_model), method="isotonic", cv=5,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_scores = calibrated_model.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, scored[best_name])
    brier_after = brier_score_loss(y_test, calibrated_scores)
    print(f"  calibration: Brier {brier_before:.4f} -> {brier_after:.4f}")

    print("figures:")
    plot_curves(y_test, scored, base_rate)
    imp = plot_importance(best_model, X_test, y_test)
    budget = plot_calibration_and_capture(y_test, scored[best_name], calibrated_scores)

    write_report(rows, leak, ablation, imp, budget, len(train), len(test),
                 base_rate, best_name, brier_before, brier_after)

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(
        {"models": rows, "ablation": ablation, "leakage_check": leak,
         "calibration": {"brier_before": brier_before, "brier_after": brier_after},
         "budget": budget, "best_model": best_name, "base_rate": base_rate,
         "n_train": len(train), "n_test": len(test)},
        indent=2,
    ))
    print(f"  wrote metrics.json")
    print("done")


if __name__ == "__main__":
    main()
