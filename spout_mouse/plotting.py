from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_ribbon,
    geom_boxplot,
    geom_point,
    facet_wrap,
    labs,
    theme_bw,
    theme,
    scale_color_manual,
    scale_fill_manual,
    element_text,
    element_rect
)
from .config import LIQUIDS_COLORS, DAY_COLORS


def plot_lick_rate(lick_data_grouped):
    """
    Plot lick rate over time with confidence intervals.

    Parameters:
        lick_data_grouped (pd.DataFrame): DataFrame containing lick rate data.

    Returns:
        plotnine.ggplot: The generated plot.
    """
    plot = (
        ggplot(
            lick_data_grouped,
            aes(x='time_ms_binned', y='lick_avg_all', color='spout_name')
        )
        + geom_line(size=0.5)
        + geom_ribbon(
            aes(ymin='lick_avg_all - sem', ymax='lick_avg_all + sem', fill='spout_name'),
            alpha=0.2
        )
        + facet_wrap('~group', scales='fixed')
        + scale_color_manual(values=LIQUIDS_COLORS)
        + scale_fill_manual(values=LIQUIDS_COLORS)
        + labs(
            title='Lick Rate Over Time',
            x='Time (ms)',
            y='Lick Rate (Hz)',
            color='Spout Name',
            fill='Spout Name'
        )
        + theme_bw()
        + theme(
            figure_size=(10, 6),
            axis_text_x=element_text(rotation=45, hjust=1),
            legend_title=element_text(size=10),
            plot_title=element_text(size=12),
            strip_background=element_rect(fill="#f0f0f0"),
        )
    )
    return plot


def plot_total_licks(licks_per_spout):
    """
    Plot total licks per spout using boxplots.

    Parameters:
        licks_per_spout (pd.DataFrame): DataFrame containing total licks per spout.

    Returns:
        plotnine.ggplot: The generated plot.
    """
    plot = (
        ggplot(
            licks_per_spout,
            aes(x='spout_name', y='lick_count_total', color='spout_name')
        )
        + geom_boxplot()
        + geom_point(position='jitter', alpha=0.5)
        + facet_wrap('~group')
        + scale_color_manual(values=LIQUIDS_COLORS)
        + labs(
            title='Total Licks per Spout',
            x='Spout Name',
            y='Total Licks',
            color='Spout Name'
        )
        + theme_bw()
        + theme(
            figure_size=(10, 6),
            axis_text_x=element_text(rotation=45, hjust=1),
            legend_title=element_text(size=10),
            plot_title=element_text(size=12),
            strip_background=element_rect(fill="#f0f0f0"),
        )
    )
    return plot
