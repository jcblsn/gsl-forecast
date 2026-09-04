"""Read, verify, and display the committed cross-validation summary.

The experiment database and row-level predictions are local working files. `gsl-cv` can export
a compact summary to `data/results/`. Its manifest records file hashes, the source commit, the
working-tree state, and the command that produced the summary.
"""

import csv
import hashlib
import json
import os

import pandas as pd

from src.forecasting.headline import (
    APR_JUN_MONTHLY_MEAN_MAX,
    SEPTEMBER_MONTHLY_MEAN,
    TARGET_LABELS,
)

RESULTS_DIR = os.path.join("data", "results")
EXPERIMENT = "experiment.json"
METRICS = "metrics.csv"
RUNS = "runs.csv"
MANIFEST = "manifest.json"

TABLE_LEADS = (1, 3, 6, 9, 12, 18, 24)
INTERVAL_LEADS = (6, 12)
HEADLINE_ROWS = (
    (
        APR_JUN_MONTHLY_MEAN_MAX,
        TARGET_LABELS[APR_JUN_MONTHLY_MEAN_MAX],
        ("jan", "feb", "mar", "apr", "may"),
    ),
    (
        SEPTEMBER_MONTHLY_MEAN,
        TARGET_LABELS[SEPTEMBER_MONTHLY_MEAN],
        ("jan", "apr", "jun", "jul", "aug"),
    ),
)

LEAD_METRICS = ("mae", "rmse", "mae_ratio", "mean_pinball_loss", "wis", "cov90")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_values_sha256(path: str) -> tuple[int, str]:
    with open(path, newline="") as stream:
        values = [row["value"] for row in csv.DictReader(stream)]
    payload = "".join(f"{value}\n" for value in values).encode()
    return len(values), hashlib.sha256(payload).hexdigest()


SNAPSHOT_FILES = (EXPERIMENT, METRICS, RUNS)
DEFAULT_LIMITATIONS = (
    "This repeatedly used development cohort is unsuitable as untouched test evidence.",
    "The snapshot does not contain row-level predictions, so its summary metrics cannot be "
    "recomputed from this directory.",
    "Historical cross-validation uses revised data rather than issue-vintage inputs.",
)


def write_manifest(
    results_dir: str,
    evaluation_policy_version: str,
    evaluation_split: str,
    limitations: tuple[str, ...] = DEFAULT_LIMITATIONS,
    note: str | None = None,
) -> dict:
    """Hash a snapshot so a published number cannot drift away from the run behind it.

    The tracker writes the snapshot files. It cannot know the evaluation policy the run
    followed, and it does not hash what it wrote, so this function completes the record.
    `gsl-results --verify-manifest` reads it back, and continuous integration runs that.
    """
    with open(os.path.join(results_dir, EXPERIMENT)) as stream:
        experiment = json.load(stream)
    count, digest = _numeric_values_sha256(os.path.join(results_dir, METRICS))
    manifest = {
        "schema_version": 1,
        "snapshot_status": "frozen_development_only",
        "evaluation_policy_version": evaluation_policy_version,
        "evaluation_split": evaluation_split,
        "source_run": experiment.get("name"),
        "source_commit": experiment.get("git_commit"),
        "limitations": list(limitations),
        "numeric_value_count": count,
        "numeric_values_sha256": digest,
        "files": {
            name: {"sha256": _sha256(os.path.join(results_dir, name))}
            for name in SNAPSHOT_FILES
            if os.path.exists(os.path.join(results_dir, name))
        },
    }
    if note:
        manifest["source_note"] = note
    with open(os.path.join(results_dir, MANIFEST), "w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def verify_manifest(results_dir: str = RESULTS_DIR) -> dict:
    """Verify the immutable development snapshot against its checked-in manifest."""
    path = os.path.join(results_dir, MANIFEST)
    with open(path) as stream:
        manifest = json.load(stream)
    for filename, expected in manifest["files"].items():
        actual = _sha256(os.path.join(results_dir, filename))
        if actual != expected["sha256"]:
            raise ValueError(f"Frozen results hash mismatch for {filename}: {actual}")
    count, digest = _numeric_values_sha256(os.path.join(results_dir, METRICS))
    if count != manifest["numeric_value_count"] or digest != manifest["numeric_values_sha256"]:
        raise ValueError("Frozen metric numeric values do not match the manifest")
    return manifest


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
    """Weighted interval score and 90% coverage at selected reporting leads.

    The snapshot also holds the mean pinball loss, which is exactly half the weighted
    interval score for this symmetric quantile set. The table prints the score that has a
    recognized name.
    """
    column = "wis" if "wis" in summary.columns else "mean_pinball_loss"
    if column not in summary.columns:
        return ""
    loss = summary.pivot(index="h", columns="model", values=column)
    cov = summary.pivot(index="h", columns="model", values="cov90")
    columns = [m for m in models if m in loss.columns]
    lines = [_row(["Lead", *columns]), _row(["---"] * (len(columns) + 1))]
    for h in INTERVAL_LEADS:
        if h not in loss.index:
            continue
        lines.append(
            _row([str(h), *[f"{loss.loc[h, m]:.2f} / {cov.loc[h, m]:.2f}" for m in columns]])
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
    """Render Markdown inspection tables from the frozen development snapshot."""
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
        parts += ["", "Weighted interval score and nominal central-90% coverage:", "", intervals]
    if not headline.empty:
        parts += [
            "",
            "Headline scalars by issue date (MAE, ft):",
            "",
            headline_table(headline, columns),
        ]
    return "\n".join(parts)
