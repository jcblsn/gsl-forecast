import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_bw,
)


def base_theme(figure_size=(10, 5)):
    return theme_bw() + theme(
        figure_size=figure_size,
        plot_title=element_text(size=13, face="bold"),
        plot_subtitle=element_text(size=10, color="#555555"),
        panel_grid_minor=element_blank(),
    )


def plot_cv_mae(summary: pd.DataFrame, output_path: str, subtitle: str) -> str:
    horizon = int(summary["h"].max())
    plot = (
        ggplot(summary, aes(x="h", y="mae", color="model", group="model"))
        + geom_line(size=0.8)
        + geom_point(size=1.5)
        + scale_x_continuous(breaks=list(range(1, horizon + 1)))
        + labs(
            title="Walk-forward cross-validation: mean absolute error by lead",
            subtitle=subtitle,
            x="Lead (monthly steps after cutoff)",
            y="Mean absolute error (ft)",
            color="Model",
        )
        + base_theme()
    )
    plot.save(output_path, dpi=150, verbose=False)
    return output_path


def plot_cv_ratio(summary: pd.DataFrame, output_path: str, subtitle: str) -> str:
    horizon = int(summary["h"].max())
    plot = (
        ggplot(summary, aes(x="h", y="mae_ratio", color="model", group="model"))
        + geom_hline(yintercept=1.0, linetype="dashed", color="#888888")
        + geom_line(size=0.8)
        + geom_point(size=1.5)
        + scale_x_continuous(breaks=list(range(1, horizon + 1)))
        + labs(
            title="MAE relative to persistence (values below 1 are lower)",
            subtitle=subtitle,
            x="Lead (monthly steps after cutoff)",
            y="MAE / MAE(naive_last)",
            color="Model",
        )
        + base_theme()
    )
    plot.save(output_path, dpi=150, verbose=False)
    return output_path
