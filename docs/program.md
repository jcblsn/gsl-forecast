# Autoresearch program for gsl-forecast

This file is the research strategy for an agent-driven improvement loop, in the pattern of
karpathy/autoresearch: one editable asset, one scalar metric, one fixed harness. The loop
fits the multivariate layer and not the univariate baselines, whose design space is too
small to search. The data pipeline is not part of the loop: a new source is a reviewed
pipeline change first, and only then a feature the loop may use.

## The three primitives

- Editable asset: files under `src/forecasting/multivariate/` and the entries that register
  them in `src/forecasting/registry.py`. Nothing else may be edited by the loop.
- Metric: `peak_mae_feb` for the model under study, as logged by `gsl-cv` (spring-peak MAE
  in feet from cutoffs whose data end in January, i.e. the February 1 outlook). Secondary
  metrics to report but not optimise: `mae_h6`, `crps_h6`, `wyend_mae_apr`.
- Harness: `uv run gsl-cv --no-plots --models <model>,ets_damped_s12` with config defaults
  (24-month horizon, cutoffs from the last 15 years, training from 1960). The harness, the
  data pipeline, the univariate baselines and the scoring code are fixed.

## Loop

1. Read the latest experiment's metrics: `uv run gsl-results <experiment_id>`.
2. Propose one change to one multivariate model (features, lags, regularisation, fallback
   rules, the stage-two step in `inflow_chain`, a new model file).
3. Run the harness. Compare `peak_mae_feb` and `mae_h6` against the previous run and
   against `ets_damped_s12`.
4. Keep the change (commit) if `peak_mae_feb` improves without `mae_h6` getting worse by
   more than 0.05 ft; otherwise revert.
5. Log one line per experiment in `docs/autoresearch.log`: date, model, change, metrics.

## Guardrails

- A model may only use columns dated at or before the cutoff. The leakage test in
  `tests/test_multivariate.py` must keep passing; add a case for any new feature.
- Do not touch `train_start`, the cutoff policy, or the horizon.
- Do not add models that need data outside `monthly_elevation` and `monthly_covariates`
  without first extending the pipeline in a separate, reviewed change.

## Ideas queue

Results so far are in `docs/autoresearch.log`. Reservoir storage as a plain extra regressor
lowers skill. The cause is not a small sample: each fit has 32 rows at a 2011 cutoff and 47
rows at a 2026 cutoff, against 4 parameters. The cause is collinearity, because storage
moves with the lake level and with the same long trend. Storage needs a form that carries
new information, such as the deficit below capacity. The head difference between the arms
helps the peak and is registered as `swe_head`.

Columns already in `monthly_covariates` (see the README data section):

- A season-aware feature set. `swe_pct_median_*` is NULL from June to September, because the
  median of the site sum is 0 in those months. So the `swe_pct` run in the log did not gain
  skill after June from percent of median; it gained skill from the level-only fallback that
  the NULL triggered. Test the schedule directly: snowpack from October to May, and soil
  moisture or year-to-date inflow from June to September.
- `prec_pct_median_*` in place of raw inches (the site roster grew from 18 to 55 sites, so
  raw means drift). This column has no summer gap.
- Separate Bear/Weber/Provo-Jordan terms instead of the pooled index.
- Reservoir storage in a form that does not add a free coefficient per month: deficit
  below the long-run maximum, or a single pooled regression across calendar months.
- `sms_eom_gsl` (8-inch soil moisture, from 1999) for autumn cutoffs.
- `head_diff_ft` in the `inflow_chain` bucket step (it already helps `swe_regression`).
- Feature sets with 3 or more terms. The estimator standardises the design and picks `alpha`
  per fit by generalised cross-validation, so a wide feature set is now shrunk rather than
  fitted at full scale.
- Stacking with more than the 2 components in `blend`, or a residual form that regresses
  `ets_damped_s12` residuals at each lead on snowpack anomalies.
- `swe_head` in place of `swe_regression` as the snowpack component of `blend`. It wins at
  leads 5 to 12 and loses badly past lead 15, which is the shape the blend is built to
  handle.

Needs pipeline work first:

- The issued NRCS inflow forecast (`nrcs_inflow_forecasts`) as an in-season anchor for
  `inflow_chain` stage one. There are 15 publication dates so far, which is too few to fit
  a coefficient on.
- Climate indices (NINO4, PDO, PNA, SOI) for leads beyond 12 months.

Done and available:

- Bathymetry, as `inflow_chain_area`. It scores within 0.02 ft of `inflow_chain`.
- Climate division temperature and precipitation, as `tavg_f_gsl_lag1` and `prcp_in_gsl_lag1`.
  Use the lagged columns only. NOAA releases a month around the 8th of the next month, so the
  unlagged column is never available at issue time and would leak in cross-validation.
