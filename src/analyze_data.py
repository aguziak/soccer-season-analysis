from src.get_season_data import get_season_data
from constants import team_three_letter_codes, team_colors
from matplotlib.widgets import Slider
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
import src.network_analysis as network_analysis


def draw_plots(round_num: int, total_num_rounds: int, tables_df: pd.DataFrame, bar_graph_ax: plt.Axes,
               line_graph_ax: plt.Axes):
    bar_graph_ax.clear()
    line_graph_ax.clear()

    ylim_pts = 110

    line_graph_ax.set_xlim(left=1, right=total_num_rounds)
    line_graph_ax.set_ylim(bottom=0, top=ylim_pts)

    bar_graph_ax.set_ylim(bottom=0, top=ylim_pts)

    round_df = tables_df.loc[tables_df['round'] == round_num]

    round_df = round_df.sort_values(by='team_id')
    tables_till_current_round_df = tables_df.loc[tables_df['round'] <= round_num]

    for team_abbr in round_df['team_abbr']:
        team_df = tables_till_current_round_df.loc[tables_till_current_round_df['team_abbr'] == team_abbr]
        team_color = team_df['team_color'].iloc[0]

        only_last_marker_mask = [False] * (len(team_df['pts']) - 1) + [True]

        line_graph_ax.plot(team_df['round'], team_df['pts'], label=team_abbr, color=team_color, marker='o',
                           markersize=4, markevery=only_last_marker_mask)
        line_graph_ax.set_ylabel('Points')

    round_df = round_df.sort_values(by='pts')

    bar_graph_ax.bar(x=round_df['team_abbr'], height=round_df['pts'], color=round_df['team_color'])
    bar_graph_ax.set_ylabel('Points')


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


def create_tables_df(matches_df: pd.DataFrame) -> pd.DataFrame:
    matches_df['round'] = matches_df['league-round'].apply(lambda s: int(s.split('Regular Season - ')[1]))

    matches_df['home_win'] = matches_df['score-fulltime-home'] > matches_df['score-fulltime-away']
    matches_df['away_win'] = matches_df['score-fulltime-home'] < matches_df['score-fulltime-away']
    matches_df['draw'] = matches_df['score-fulltime-home'] == matches_df['score-fulltime-away']

    matches_df['home_pts'] = matches_df['home_win'] * 3 + matches_df['draw'] * 1
    matches_df['away_pts'] = matches_df['away_win'] * 3 + matches_df['draw'] * 1

    matches_df['home_gd_half_time'] = matches_df['score-halftime-home'] - matches_df['score-halftime-away']
    matches_df['away_gd_half_time'] = -matches_df['home_gd_half_time']

    matches_df['home_gd_full_time'] = matches_df['score-fulltime-home'] - matches_df['score-fulltime-away']
    matches_df['away_gd_full_time'] = -matches_df['home_gd_full_time']

    relevant_cols = ['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                     'score-halftime-home', 'score-halftime-away', 'score-fulltime-home', 'score-fulltime-away',
                     'home_win', 'away_win', 'draw', 'home_pts', 'away_pts', 'home_gd_half_time', 'away_gd_half_time',
                     'home_gd_full_time', 'away_gd_full_time']
    matches_df = matches_df[relevant_cols]

    home_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'home_win', 'draw', 'home_pts', 'home_gd_half_time', 'home_gd_full_time']] \
        .rename({'teams-home-id': 'team_id', 'teams-home-name': 'team_name', 'teams-away-id': 'opp_team_id',
                 'teams-away-name': 'opp_team_name', 'home_win': 'match_win', 'home_pts': 'match_pts',
                 'home_gd_half_time': 'match_gd_half', 'home_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw'}, axis='columns')
    home_match_results_df['home'] = True

    away_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'away_win', 'draw', 'away_pts', 'away_gd_half_time', 'away_gd_full_time']] \
        .rename({'teams-away-id': 'team_id', 'teams-away-name': 'team_name', 'teams-home-id': 'opp_team_id',
                 'teams-home-name': 'opp_team_name', 'away_win': 'match_win', 'away_pts': 'match_pts',
                 'away_gd_half_time': 'match_gd_half', 'away_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw'}, axis='columns')
    away_match_results_df['home'] = False

    tables_df = pd.concat([home_match_results_df, away_match_results_df], axis='rows') \
        .sort_values(by='round') \
        .reset_index(drop=True)

    cumulative_stats = tables_df[['team_id', 'match_win', 'match_draw', 'match_pts', 'match_gd_half', 'match_gd_full']] \
        .groupby(by='team_id').cumsum() \
        .rename({'match_win': 'wins', 'match_draw': 'draws', 'match_pts': 'pts', 'match_gd_half': 'gd_half',
                 'match_gd_full': 'gd_full'}, axis='columns')

    tables_df = pd.concat([tables_df, cumulative_stats], axis='columns')
    tables_df['losses'] = tables_df['round'] - tables_df['wins'] - tables_df['draws']

    tables_df['team_abbr'] = tables_df.apply(lambda row: team_three_letter_codes[row['team_id']], axis='columns')
    tables_df['opp_team_abbr'] = tables_df.apply(lambda row: team_three_letter_codes[row['opp_team_id']], axis='columns')

    tables_df['team_color'] = tables_df.apply(lambda row: team_colors[row['team_id']], axis='columns')
    tables_df['opp_team_color'] = tables_df.apply(lambda row: team_colors[row['opp_team_id']], axis='columns')

    return tables_df


if __name__ == '__main__':
    epl_league_id = 39
    epl_matches_df = get_season_data(season=2018, league_id=epl_league_id, use_cache=True)
    epl_tables_df = create_tables_df(epl_matches_df)

    # create_season_table_plot(epl_tables_df)

    fig, ax = plt.subplots()
    network_analysis.create_network_from_table_data(table_df=epl_tables_df, round_num=38, network_graph_ax=ax)

    plt.show()
