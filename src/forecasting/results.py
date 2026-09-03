"""The committed record of one cross-validation run.

The experiment tracker database is a working scratchpad, and `.gitignore` excludes it. So it
cannot be the citation for a published number. This module writes the same numbers to small
files under `data/results/`, which the repository keeps, and renders the markdown tables in
the README and in `docs/model-spec.md` from those files.
"""

import json
import os
import subprocess

import pandas as pd

RESULTS_DIR = os.path.join("data", "results")
CV_SUMMARY = "cv_summary.csv"
HEADLINE_SUMMARY = "headline_summary.csv"
META = "cv_summary.meta.json"

TABLE_LEADS = (1, 3, 6, 9, 12, 18, 24)
INTERVAL_LEADS = (6, 12)
HEADLINE_ROWS = (
    ("peak", "Spring peak", ("jan", "feb", "mar", "apr", "may")),
    ("wy_end", "Water-year end", ("jan", "apr", "jun", "jul", "aug")),
)


def git_commit() -> str:
    """The commit that produced the run, or an empty string outside a work tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def write_results(
    summary: pd.DataFrame,
    headline_summary: pd.DataFrame,
    meta: dict,
    results_dir: str = RESULTS_DIR,
) -> str:
    """Write the 2 summary tables and the run description. Returns the directory."""
    os.makedirs(results_dir, exist_ok=True)
    summary.sort_values(["model", "h"]).to_csv(
        os.path.join(results_dir, CV_SUMMARY), index=False, float_format="%.6f"
    )
    headline_summary.sort_values(["target", "issue", "model"]).to_csv(
        os.path.join(results_dir, HEADLINE_SUMMARY), index=False, float_format="%.6f"
    )
    with open(os.path.join(results_dir, META), "w") as stream:
        json.dump(meta, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return results_dir


def read_results(results_dir: str = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The 2 summary tables and the run description, as written by `write_results`."""
    summary = pd.read_csv(os.path.join(results_dir, CV_SUMMARY))
    headline = pd.read_csv(os.path.join(results_dir, HEADLINE_SUMMARY))
    with open(os.path.join(results_dir, META)) as stream:
        meta = json.load(stream)
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
