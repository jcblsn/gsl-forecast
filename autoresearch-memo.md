# Autoresearch for GSL Forecasting

## Background

Karpathy's autoresearch (github.com/karpathy/autoresearch) automates the ML research loop: an AI coding agent reads a plain-English research strategy (`program.md`), edits a single file (`train.py`), evaluates against a fixed metric (`val_bpb`), and keeps or reverts the change. The loop runs ~12 experiments/hour, ~100 overnight. Git serves as memory. The human's job shifts from writing model code to writing the research strategy.

The pattern generalizes to any domain with three primitives:
1. An editable asset (one file the agent can change)
2. A scalar metric (measurable, not subjective)
3. A fixed constraint (evaluation harness the agent cannot modify)

Deedy Das applied this to a chess engine (github.com/deedy/chess), optimizing Elo rating by letting the agent modify search/evaluation code and benchmarking against Stockfish.

## How This Applies to GSL Forecasting

### Direction 1: Univariate model improvement

Weak fit. The current univariate forecasters (naive, MA, drift, Holt-Winters, Theta) are clean implementations of well-understood statistical methods with small parameter spaces. There isn't a large design space for an agent to explore -- adding the next method (ARIMA, Prophet) is more "implement from a textbook" than "search a design space." The overhead of setting up autoresearch likely exceeds the value here.

### Direction 2: Multivariate forecasting with external data

Strong fit, but only if the problem is decomposed correctly.

GSL level is driven by many factors beyond its own history:
- Tributary streamflow (Bear, Weber, Logan rivers)
- Snowpack and snowmelt (SNOTEL stations)
- Direct precipitation on the lake
- Evaporation (temperature, wind)
- Pacific climate teleconnections (Quasi-Decadal Oscillation, 10-15 year cycles linked to central Pacific SST)
- Multi-decadal drought cycles (30-50 year oscillation)
- Tree-ring proxies capturing 576 years of low-frequency variability (see Gillies et al. 2015)
- Human diversions and consumption

Gillies et al. (2015) demonstrated that adding tree-ring reconstructed GSL data as an exogenous variable reduced 5-year forecast RMSE from 40.2 to 32.1 cm/yr. The implication: external data sources meaningfully improve GSL forecasts, especially at longer horizons where low-frequency climate cycles dominate.

### The right decomposition

The data engineering -- fetching, cleaning, aligning external data sources to the monthly grain -- does not fit autoresearch. It's not a "try, measure, keep/revert" loop. You can't evaluate whether a data source is useful until it's been prepared and made available.

But once a feature store is built (all candidate external series cleaned and available in DuckDB), autoresearch maps cleanly:

- Editable asset: a single forecasting file that defines which features to use, how to lag/transform them, and what model to fit
- Scalar metric: MAE or RMSE at a target horizon (e.g. `mae_h6`) via walk-forward CV
- Fixed constraint: the CV harness (`gsl-cv`) and the prepared dataset, both locked from agent modification

The agent could then explore: which feature subsets help at which horizons, what lag structures capture teleconnection effects, whether VAR/ARIMAX/ridge regression beats univariate ETS when given external regressors, whether Pacific SST anomalies improve 12-month forecasts even if they don't help at 1-month, etc. That's a legitimately large search space where autonomous iteration adds value.

## Proposed Approach

1. Build a multivariate feature store: fetch USGS streamflow, SNOTEL snowpack, NOAA precipitation, Pacific SST indices, and any available tree-ring data. Land it all in clean columns in `gsl.db`.
2. Write a single `multivariate_forecast.py` that the agent can edit freely -- feature selection, lag structures, transformations, model architecture.
3. Lock the CV harness and data pipeline so the agent can't cheat.
4. Write a `program.md` research strategy and let autoresearch run.

## References

- Karpathy, A. (2026). autoresearch. github.com/karpathy/autoresearch
- Das, D. (2026). chess. github.com/deedy/chess
- Gillies, R.R., Chung, O.-Y., Wang, S.-Y., DeRose, R.J., Sun, Y. (2015). Added value from 576 years of tree-ring records in the prediction of the Great Salt Lake level. Journal of Hydrology, 529, 962-968.
