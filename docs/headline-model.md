# Current headline model

The current headline model is `swe_ets_blend_v1`.

For cutoff month $t$, calendar month $m$, and lead $h$, the snow model is:

$$
S_{t,h}=Y_t+\beta_{0,m,h}+\beta_{Y,m,h}Y_t
+\beta_{S,m,h}SWE_t+\beta_{P,m,h}P_t+\beta_{D,m,h}D_t.
$$

$Y_t$ is the south-arm monthly mean elevation. $SWE_t$ is the basin snow water equivalent.
$P_t$ is the water-year precipitation. $D_t$ is the south-minus-north arm head difference.

In this version, the snow component fits one direct ridge regression for each cutoff month
and lead. The training rows
use past cutoffs from the same calendar month. The response is the elevation change through
the target month. The ridge penalty does not apply to the intercept.

This version combines the snow component with the damped seasonal ETS component $E_{t,h}$:

$$
F_{t,h}=w_{q(t),h}S_{t,h}+[1-w_{q(t),h}]E_{t,h}.
$$

$q(t)$ is the issue season. The accumulation season is November through March. The melt
season is April through June. The recession season is July through October.

Each monthly run learns 24 weights for each season from walk-forward forecasts. It searches
weights from 0.00 through 1.00 in steps of 0.01. It selects the sequence with the lowest
total absolute error. The snow-model weight cannot increase with lead. An equal-loss choice
uses the smaller snow-model weight.

The reported blend scores use held-out years. The weight fit for one year excludes all
cutoffs from that year. The prediction intervals use the held-out blend errors.

For input $j$, the contribution on one target date is:

$$
C_{j,t,h}=w_{q(t),h}\beta_{j,m,h}(x_{j,t}-\bar{x}_{j,m,h}).
$$

The current-level term also includes the direct $Y_t$ term outside the regression. The
reference path contains the centered snow-model reference and the ETS share. The reference
path plus all input contributions equals the point forecast.

These contributions describe terms in the fitted model. They do not estimate causal effects.
