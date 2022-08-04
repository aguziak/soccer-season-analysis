import pandas as pd
import numpy as np
import math

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LabelSet, VBar, Legend
from bokeh.layouts import row, column

from functools import partial

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from src.analyze_data import draw_plots


def create_bokeh_plot_for_round(tables_df: pd.DataFrame, round_num: int):
    round_df = tables_df.loc[tables_df['round'] == round_num].reset_index(drop=True)

    round_df = round_df.sort_values(by='team_id')

    round_df = round_df.sort_values(by='pts', ascending=True)

    line_graph = figure(title='Season Points Progression',
                        x_range=(0, max(tables_df['round'])),
                        y_range=(0, max(tables_df['pts']) + 10),
                        x_axis_label='Round',
                        y_axis_label='Points')

    line_graph.toolbar.active_drag = None
    line_graph.toolbar.active_scroll = None
    line_graph.toolbar.active_tap = None

    line_graph.y_range.start = 0

    line_graph.xgrid.grid_line_color = None

    bar_graph = figure(y_axis_label='Points',
                       x_range=round_df['team_abbr'],
                       toolbar_location=None)

    bar_graph.toolbar.active_drag = None
    bar_graph.toolbar.active_scroll = None
    bar_graph.toolbar.active_tap = None

    bar_graph.y_range.start = 0

    bar_graph.xgrid.grid_line_color = None

    bar_graph.xaxis.major_label_orientation = math.pi / 4.

    bar_data_source = ColumnDataSource(
        dict(
            x=round_df['team_abbr'],
            y=[0] * len(round_df),
            top=round_df['pts'],
            color=round_df['team_color'],
            label=round_df['team_abbr']
        )
    )

    legend_items = list()

    for team_abbr in round_df['team_abbr']:
        tables_till_current_round_df = tables_df.loc[tables_df['round'] <= round_num]
        team_df = tables_till_current_round_df.loc[tables_till_current_round_df['team_abbr'] == team_abbr]
        team_color = team_df['team_color'].iloc[0]

        legend_items.append(
            [team_abbr,
             [line_graph.line(team_df['round'], team_df['pts'], line_color=team_color, line_width=2)]
             ])
        bar_graph.vbar(x='x', top='top', color='color', source=bar_data_source)

    line_graph.add_layout(Legend(items=legend_items), 'left')
    show(row(line_graph, bar_graph))


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
