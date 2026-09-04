"""Close the south-arm water balance and report what does not close.

This answers one question: is the data good enough to conserve volume? It converts the
end-of-month elevation to storage, subtracts the measured fluxes, and reports what is left.

The residual is `net_unmeasured_flux`. It is not evaporation and it is not consumptive use.
It holds the causeway exchange, the error in the gauged-to-delivered inflow ratio, the error
in the bathymetry, and the part of evaporation the temperature term does not explain.

A residual that is stable across elevation bands and across held-out periods says the
measurement system is consistent. A residual that moves with the lake says a term that
matters is missing, and that no amount of model fitting will repair it.
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

from src.config import load_config
from src.forecasting.cutoffs import issue_season
from src.forecasting.data import load_monthly_data
from src.forecasting.multivariate.water_balance import (
    EOM_COL,
    INFLOW_COL,
    WaterBalanceForecaster,
)

ELEVATION_BANDS = [(-np.inf, 4192.0), (4192.0, 4196.0), (4196.0, 4200.0), (4200.0, np.inf)]


def band_label(low: float, high: float) -> str:
    if np.isneginf(low):
        return f"below {high:.0f}"
    if np.isposinf(high):
        return f"above {low:.0f}"
    return f"{low:.0f} to {high:.0f}"


def closure_frame(data: pd.DataFrame, model: WaterBalanceForecaster) -> pd.DataFrame:
    """One row per closed monthly step, with the residual in kaf and in feet."""
    df = model._frame(data)
    ok = model._usable(df)
    rows = df[ok].copy()
    X, y = model._design(rows, model._coefficients["salt_coefficient"])
    residual = y - X @ model._beta
    rows["residual_kaf"] = residual
    rows["residual_ft"] = residual * 1000.0 / rows["area_prev"].to_numpy(dtype=float)
    rows["elevation"] = rows[EOM_COL]
    rows["water_year"] = rows[TIME := "month"].dt.year + (rows[TIME].dt.month >= 10).astype(int)
    rows["season"] = [issue_season(m) for m in rows[TIME]]
    rows["band"] = pd.cut(
        rows["elevation"],
        bins=[b[0] for b in ELEVATION_BANDS] + [ELEVATION_BANDS[-1][1]],
        labels=[band_label(*b) for b in ELEVATION_BANDS],
    )
    return rows


def summarize(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    grouped = rows.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean_kaf": grouped["residual_kaf"].mean(),
            "sd_kaf": grouped["residual_kaf"].std(),
            "sd_ft": grouped["residual_ft"].std(),
        }
    )
    return out.reset_index()


def gauged_closure(rows: pd.DataFrame) -> pd.DataFrame:
    """What the balance looks like with the gauged inflow alone and nothing else.

    This is the starting point the project was at: no evaporation term, no lake
    precipitation, no delivery correction. It is reported so the value of each added term is
    visible rather than asserted.
    """
    naive = rows["delta_v"] - rows[INFLOW_COL]
    depth = naive * 1000.0 / rows["area_prev"]
    return pd.DataFrame(
        {
            "specification": ["gauged inflow alone", "full balance"],
            "sd_kaf": [float(naive.std()), float(rows["residual_kaf"].std())],
            "sd_ft": [float(depth.std()), float(rows["residual_ft"].std())],
        }
    )


def holdout_stability(data: pd.DataFrame, folds: int = 3) -> pd.DataFrame:
    """Fit on all but one block of years and report the residual on the block held out.

    A closure that only holds where it was fitted is a fitted constant, not a measurement.
    """
    df = data.sort_values("month").reset_index(drop=True)
    years = df["month"].dt.year
    edges = np.array_split(np.array(sorted(years.unique())), folds)
    rows = []
    for block in edges:
        held = years.isin(block)
        if int((~held).sum()) < 60 or int(held.sum()) < 12:
            continue
        model = WaterBalanceForecaster().fit(df[~held])
        frame = closure_frame(df, model)
        inside = frame["month"].dt.year.isin(block)
        rows.append(
            {
                "held_out": f"{block.min()}-{block.max()}",
                "n": int(inside.sum()),
                "sd_ft_in_fit": float(frame.loc[~inside, "residual_ft"].std()),
                "sd_ft_held_out": float(frame.loc[inside, "residual_ft"].std()),
                "mean_kaf_held_out": float(frame.loc[inside, "residual_kaf"].mean()),
            }
        )
    return pd.DataFrame(rows)


def render(data: pd.DataFrame) -> tuple[str, dict]:
    model = WaterBalanceForecaster().fit(data)
    rows = closure_frame(data, model)
    metrics = model.get_metrics()
    lines = [
        "Great Salt Lake south-arm water balance: closure audit",
        "",
        f"Closed monthly steps: {len(rows)} from {rows['month'].min():%Y-%m} "
        f"to {rows['month'].max():%Y-%m}",
        "",
        "Fitted terms",
        f"  evaporation scale on Hargreaves    {metrics['evap_scale']}",
        f"  precipitation scale on KSLC        {metrics['precip_scale']}",
        f"  salinity suppression coefficient   {metrics['salt_coefficient']}",
        f"  gauged inflow to delivered ratio   {metrics['fitted_delivery_ratio']}",
        "",
        "What each term is worth",
        gauged_closure(rows).round(4).to_string(index=False),
        "",
        "Residual by calendar month",
        summarize(rows, "month_number").round(3).to_string(index=False),
        "",
        "Residual by issue season",
        summarize(rows, "season").round(3).to_string(index=False),
        "",
        "Residual by elevation band (ft NGVD29)",
        summarize(rows, "band").round(3).to_string(index=False),
        "",
        "Stability on held-out years",
        holdout_stability(data).round(4).to_string(index=False),
    ]
    payload = {
        "n_steps": int(len(rows)),
        "first_month": str(rows["month"].min().date()),
        "last_month": str(rows["month"].max().date()),
        "residual_sd_kaf": float(rows["residual_kaf"].std()),
        "residual_sd_ft": float(rows["residual_ft"].std()),
        "fitted": {k: v for k, v in metrics.items() if isinstance(v, int | float | str | bool)},
        "by_band": summarize(rows, "band").to_dict("records"),
        "by_season": summarize(rows, "season").to_dict("records"),
        "holdout": holdout_stability(data).to_dict("records"),
    }
    return "\n".join(lines), payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the closure of the south-arm balance")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--json", help="Write the audit payload to this path")
    parser.add_argument(
        "--validate-salinity",
        action="store_true",
        help="Also check the reconstructed salinity against the HydroShare record",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = load_config(args.config)
    data = load_monthly_data(
        config["database"]["path"], train_start=config["forecasting"]["train_start"]
    )
    text, payload = render(data)
    print(text)
    if args.validate_salinity:
        from src.forecasting.validate_salinity import render as render_salinity

        print()
        print(render_salinity(config["database"]["path"]))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        logging.info(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
