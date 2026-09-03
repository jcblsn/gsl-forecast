"""The committed record of one cross-validation run.

The experiment tracker database is a working file that `.gitignore` excludes, so it cannot be
the citation for a published number. `gsl-cv` writes a snapshot of the run under
`data/results/`, which the repository keeps. This module reads that snapshot back into the 2
tables the README and `docs/model-spec.md` are rendered from.

The snapshot is written by the tracker, not here, so it also carries the commit, the tree
state and the command line that produced the numbers.
"""

import json
import os

import pandas as pd

RESULTS_DIR = os.path.join("data", "results")
EXPERIMENT = "experiment.json"
METRICS = "metrics.csv"
RUNS = "runs.csv"

TABLE_LEADS = (1, 3, 6, 9, 12, 18, 24)
INTERVAL_LEADS = (6, 12)
HEADLINE_ROWS = (
    ("peak", "Spring peak", ("jan", "feb", "mar", "apr", "may")),
    ("wy_end", "Water-year end", ("jan", "apr", "jun", "jul", "aug")),
)

LEAD_METRICS = ("mae", "rmse", "mae_ratio", "crps", "cov90")


def _dims(frame: pd.DataFrame) -> pd.DataFrame:
    """Spread the dims column of a snapshot into real columns."""
    parsed = frame["dims"].apply(json.loads)
    for key in sorted({k for row in parsed for k in row}):
        frame[key] = parsed.apply(lambda row, key=key: row.get(key))
    return frame


def read_results(results_dir: str = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The lead table, the headline table, and the run description.

    The snapshot stores one row per metric and dims. The 2 tables are pivots of it: the lead
    table is keyed by model and lead, the headline table by model, issue month and target.
    """
    metrics = _dims(pd.read_csv(os.path.join(results_dir, METRICS)))
    with open(os.path.join(results_dir, EXPERIMENT)) as stream:
        record = json.load(stream)

    meta = {**record, **record.get("meta", {}), **record.get("tags", {})}
    meta["run_label"] = record["name"]

    by_lead = metrics[metrics.get("h").notna()] if "h" in metrics else metrics.iloc[:0]
    summary = (
        by_lead.pivot_table(
            index=["run_name", "h"], columns="metric", values="value", aggfunc="first"
        )
        .reset_index()
        .rename(columns={"run_name": "model"})
    )
    summary["h"] = summary["h"].astype(int)
    summary.columns.name = None
    for column in LEAD_METRICS:
        if column not in summary.columns:
            summary[column] = float("nan")

    if "target" in metrics:
        headline_rows = metrics[metrics["target"].notna()]
    else:
        headline_rows = metrics.iloc[:0]
    headline = (
        headline_rows.pivot_table(
            index=["run_name", "issue", "target"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"run_name": "model"})
    )
    headline.columns.name = None
    if "n" in headline.columns:
        headline["n"] = headline["n"].astype(int)
    if headline.empty:
        headline = pd.DataFrame(columns=["model", "issue", "target", "mae", "n"])
    return summary, headline, meta


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _order_models(summary: pd.DataFrame, models: list[str] | None, baseline: str) -> list[str]:
    """The column order: the requested models, else every model ranked by lead-6 MAE.

    The baseline goes last in both cases, because every ratio is measured against it.
    """
    if models is None:
        at6 = summary[summary["h"] == 6].sort_values("mae")
        models = [m for m in at6["model"] if m in set(summary["model"])]
    ordered = [m for m in models if m != baseline]
    if baseline in set(summary["model"]):
        ordered.append(baseline)
    return ordered


def mae_table(summary: pd.DataFrame, models: list[str], headline_model: str) -> str:
    """MAE by lead, with the ratio of the headline model to the baseline."""
    mae = summary.pivot(index="h", columns="model", values="mae")
    ratio = summary.pivot(index="h", columns="model", values="mae_ratio")
    leads = [h for h in TABLE_LEADS if h in mae.index]
    columns = [m for m in models if m in mae.columns]
    lines = [
        _row(["Lead", *columns, f"Ratio ({headline_model})"]),
        _row(["---"] * (len(columns) + 2)),
    ]
    for h in leads:
        cells = [f"{mae.loc[h, m]:.2f}" for m in columns]
        share = ratio.loc[h, headline_model] if headline_model in ratio.columns else float("nan")
        lines.append(_row([str(h), *cells, f"{share:.2f}"]))
    return "\n".join(lines)


def interval_table(summary: pd.DataFrame, models: list[str]) -> str:
    """CRPS and 90% coverage at the leads the README reports."""
    if "crps" not in summary.columns:
        return ""
    crps = summary.pivot(index="h", columns="model", values="crps")
    cov = summary.pivot(index="h", columns="model", values="cov90")
    columns = [m for m in models if m in crps.columns]
    lines = [_row(["Lead", *columns]), _row(["---"] * (len(columns) + 1))]
    for h in INTERVAL_LEADS:
        if h not in crps.index:
            continue
        lines.append(
            _row([str(h), *[f"{crps.loc[h, m]:.2f} / {cov.loc[h, m]:.2f}" for m in columns]])
        )
    return "\n".join(lines)


def headline_table(headline: pd.DataFrame, models: list[str]) -> str:
    """The 2 headline scalars by the date the forecast goes out."""
    pivot = headline.pivot_table(index=["target", "issue"], columns="model", values="mae")
    columns = [m for m in models if m in pivot.columns]
    lines = [_row(["Target", "Issue", *columns]), _row(["---"] * (len(columns) + 2))]
    month_names = {
        "jan": "Jan 1",
        "feb": "Feb 1",
        "mar": "Mar 1",
        "apr": "Apr 1",
        "may": "May 1",
        "jun": "Jun 1",
        "jul": "Jul 1",
        "aug": "Aug 1",
    }
    for target, label, issues in HEADLINE_ROWS:
        for issue in issues:
            if (target, issue) not in pivot.index:
                continue
            row = pivot.loc[(target, issue)]
            lines.append(_row([label, month_names[issue], *[f"{row[m]:.2f}" for m in columns]]))
    return "\n".join(lines)


def render_tables(
    summary: pd.DataFrame,
    headline: pd.DataFrame,
    meta: dict,
    models: list[str] | None = None,
    baseline: str = "naive_last",
) -> str:
    """The markdown for the README section "Current results", from the committed files."""
    columns = _order_models(summary, models, baseline)
    headline_model = meta.get("headline_model") or columns[0]
    commit = (meta.get("git_commit") or "")[:12]
    parts = [
        f"Run `{meta.get('run_label', 'unknown')}`, commit `{commit}`.",
        f"{meta.get('n_cutoffs', '?')} cutoffs from {meta.get('first_cutoff', '?')} to "
        f"{meta.get('last_cutoff', '?')}, {meta.get('horizon', '?')}-month horizon, "
        f"training from {meta.get('train_start', '?')}, data through "
        f"{meta.get('data_max', '?')}.",
        "",
        "MAE by lead (ft):",
        "",
        mae_table(summary, columns, headline_model),
    ]
    intervals = interval_table(summary, columns)
    if intervals:
        parts += ["", "CRPS and 90% coverage:", "", intervals]
    if not headline.empty:
        parts += [
            "",
            "Headline scalars by issue date (MAE, ft):",
            "",
            headline_table(headline, columns),
        ]
    return "\n".join(parts)
