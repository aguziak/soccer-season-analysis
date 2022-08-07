import pandas as pd
import math

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import row, column

from functools import partial

from matplotlib import pyplot as plt

from src.analyze_data import draw_plots


def create_bokeh_plot_for_round(tables_df: pd.DataFrame, round_num: int, plot_size=400):
    final_round_df = tables_df.loc[tables_df['round'] == max(tables_df['round'])] \
        .sort_values(by=['pts', 'gd_full'], ascending=True)
    final_round_df['final_position'] = final_round_df['pts'].rank(method='first').astype(int)
    final_round_df = final_round_df.reset_index(drop=True)

    tables_df = tables_df.reset_index(drop=True)
    tables_df = tables_df.merge(final_round_df[['team_id', 'final_position']], on='team_id')

    tables_df = tables_df.sort_values(by=['round', 'pts', 'team_id'], ascending=[True, False, True])

    round_df = tables_df.loc[tables_df['round'] == round_num].reset_index(drop=True)
    round_df = round_df.sort_values(by='pts', ascending=True)

    line_graph = figure(x_range=(0, max(tables_df['round'])),
                        y_range=(0, max(tables_df['pts']) + 10),
                        x_axis_label='Round',
                        y_axis_label='Points',
                        toolbar_location=None,
                        plot_width=plot_size,
                        plot_height=plot_size)

    line_graph.toolbar.active_drag = None
    line_graph.toolbar.active_scroll = None
    line_graph.toolbar.active_tap = None

    line_graph.y_range.start = 0

    line_graph.xgrid.grid_line_color = None

    bar_graph = figure(title='Season Points Progression',
                       y_axis_label='Points',
                       x_range=final_round_df['team_name'],
                       y_range=(0, max(tables_df['pts']) + 10),
                       toolbar_location=None,
                       plot_width=plot_size,
                       plot_height=plot_size)

    bar_graph.toolbar.active_drag = None
    bar_graph.toolbar.active_scroll = None
    bar_graph.toolbar.active_tap = None

    bar_graph.y_range.start = 0

    bar_graph.xgrid.grid_line_color = None

    bar_graph.xaxis.major_label_orientation = math.pi / 4.

    bar_data_source = ColumnDataSource(
        dict(
            x=round_df['team_name'],
            y=[0] * len(round_df),
            top=round_df['pts'],
            color=round_df['team_color'],
            label=round_df['team_name']
        )
    )

    tables_till_current_round_df = tables_df.loc[tables_df['round'] <= round_num]

    xs = list()
    ys = list()
    team_colors = list()
    team_name_list = list()

    for team_name in round_df['team_name']:
        round_df = round_df.sort_values(by='final_position')
        team_df = tables_till_current_round_df.loc[tables_till_current_round_df['team_name'] == team_name]
        team_color = team_df['team_color'].iloc[0]

        xs.append(team_df['round'])
        ys.append(team_df['pts'])
        team_colors.append(team_color)
        team_name_list.append(team_name)

    lines_data_source = ColumnDataSource(
        dict(
            xs=xs,
            ys=ys,
            team_colors=team_colors,
            team_name=team_name_list
        )
    )

    bar_graph.vbar(x='x', top='top', color='color', source=bar_data_source)
    line_graph.multi_line(xs='xs', ys='ys', line_color='team_colors', line_width=2, source=lines_data_source)

    round_slider = Slider(start=1, end=max(tables_df['round']), value=1, step=1, title='Round')

    def slider_update(_, __, new):
        update_round_df = tables_df.loc[tables_df['round'] == new].reset_index(drop=True)
        update_round_df = update_round_df.sort_values(by='final_position')

        bar_data_source.data['x'] = update_round_df['team_name']
        bar_data_source.data['y'] = [0] * len(update_round_df)
        bar_data_source.data['top'] = update_round_df['pts']
        bar_data_source.data['color'] = update_round_df['team_color']
        bar_data_source.data['label'] = update_round_df['team_name']

        update_tables_till_current_round_df = tables_df.loc[tables_df['round'] <= new]

        new_xs = list()
        new_ys = list()
        new_team_colors = list()
        new_team_name = list()

        for update_team_name in round_df['team_name']:
            update_team_df = update_tables_till_current_round_df.loc[
                update_tables_till_current_round_df['team_name'] == update_team_name]
            update_team_color = update_team_df['team_color'].iloc[0]

            new_xs.append(update_team_df['round'])
            new_ys.append(update_team_df['pts'])

            new_team_colors.append(update_team_color)
            new_team_name.append(update_team_name)

        lines_data_source.data['xs'] = new_xs
        lines_data_source.data['ys'] = new_ys
        lines_data_source.data['team_colors'] = new_team_colors
        lines_data_source.data['team_name'] = new_team_name

    round_slider.on_change('value', slider_update)

    return column(row(bar_graph, line_graph), round_slider)


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
