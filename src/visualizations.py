from functools import partial

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from src.analyze_data import draw_plots


def create_season_table_plot(tables_df):
    fig, (bar_ax, line_ax) = plt.subplots(2)

    fig: plt.Figure
    bar_ax: plt.Axes
    line_ax: plt.Axes

    final_round_num = max(tables_df['round'])

    draw_plots(final_round_num, total_num_rounds=final_round_num, tables_df=tables_df, bar_graph_ax=bar_ax,
               line_graph_ax=line_ax)

    plt.subplots_adjust(bottom=0.2)

    slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])

    round_slider = Slider(
        ax=slider_ax,
        label='Round',
        valmin=1,
        valmax=final_round_num,
        valinit=final_round_num,
        valstep=1
    )

    draw_plots_partial = partial(draw_plots, total_num_rounds=final_round_num, tables_df=tables_df,
                                 bar_graph_ax=bar_ax, line_graph_ax=line_ax)

    round_slider.on_changed(draw_plots_partial)

    plt.show()