from src.get_season_data import get_season_data, get_team_colors_from_api
from constants import team_three_letter_codes, team_colors
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    for team_name in round_df['team_name']:
        team_df = tables_till_current_round_df.loc[tables_till_current_round_df['team_name'] == team_name]
        team_color = team_df['team_color'].iloc[0]

        only_last_marker_mask = [False] * (len(team_df['pts']) - 1) + [True]

        line_graph_ax.plot(team_df['round'], team_df['pts'], label=team_name, color=team_color, marker='o',
                           markersize=4, markevery=only_last_marker_mask)
        line_graph_ax.set_ylabel('Points')

    round_df = round_df.sort_values(by='pts')

    bar_graph_ax.bar(x=round_df['team_name'], height=round_df['pts'], color=round_df['team_color'])
    bar_graph_ax.set_ylabel('Points')


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
                     'home_gd_full_time', 'away_gd_full_time', 'league-id', 'league-season', 'teams-home-logo']
    matches_df = matches_df[relevant_cols]

    home_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'home_win', 'draw', 'home_pts', 'home_gd_half_time', 'home_gd_full_time',
                                        'league-id', 'league-season']] \
        .rename({'teams-home-id': 'team_id', 'teams-home-name': 'team_name', 'teams-away-id': 'opp_team_id',
                 'teams-away-name': 'opp_team_name', 'home_win': 'match_win', 'home_pts': 'match_pts',
                 'home_gd_half_time': 'match_gd_half', 'home_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw', 'league-id': 'league_id', 'league-season': 'league_season'}, axis='columns')
    home_match_results_df['home'] = True

    away_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'away_win', 'draw', 'away_pts', 'away_gd_half_time', 'away_gd_full_time',
                                        'league-id', 'league-season']] \
        .rename({'teams-away-id': 'team_id', 'teams-away-name': 'team_name', 'teams-home-id': 'opp_team_id',
                 'teams-home-name': 'opp_team_name', 'away_win': 'match_win', 'away_pts': 'match_pts',
                 'away_gd_half_time': 'match_gd_half', 'away_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw', 'league-id': 'league_id', 'league-season': 'league_season'}, axis='columns')
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

    team_colors_from_api = get_team_colors_from_api(matches_df, use_cache=True, league_id=tables_df['league_id'][0],
                                                    season=tables_df['league_season'][0])

    def get_team_color(row: pd.Series, row_team_id_col_name: str) -> str:
        if row[row_team_id_col_name] in team_colors:
            return team_colors[row[row_team_id_col_name]]
        else:
            return team_colors_from_api.loc[row[row_team_id_col_name]]['team_color']

    tables_df['team_color'] = tables_df.apply(get_team_color, row_team_id_col_name='team_id', axis='columns')
    tables_df['opp_team_color'] = tables_df.apply(get_team_color, row_team_id_col_name='opp_team_id', axis='columns')

    return tables_df


def _create_rolling_avg_match_pts(group: pd.DataFrame, period: int):
    """
    Creates two new columns in the group named prev_{period}_match_ppg_avg and next_{period}_match_ppg_avg where
        {period} is the second positional argument provided and represents the size of the rolling average to use.

        The new columns represent the backwards and forwards looking moving averages of the points earned per game

    Args:
        group (DataFrame): team_id group of DataFrame generated by create_tables_df
        period (int): The size of the rolling window

    Returns:
        DataFrame: DataFrame with the two new columns

    """
    group = group.sort_values(by='round')

    group[f'prev_{period}_match_ppg_avg'] = group['match_pts'] \
        .shift(1) \
        .rolling(window=period, min_periods=period) \
        .mean()

    group[f'next_{period}_match_ppg_avg'] = group[::-1]['match_pts'] \
        .shift(1) \
        .rolling(window=period, min_periods=period) \
        .mean()
    return group


def perform_analysis():
    epl_league_id = 39
    epl_matches_df = get_season_data(season=2019, league_id=epl_league_id, use_cache=True)
    epl_tables_df = create_tables_df(epl_matches_df)
    epl_tables_df = epl_tables_df.groupby('team_id').apply(_create_rolling_avg_match_pts, period=3)

    epl_tables_df['3_match_ppg_diff'] = epl_tables_df['prev_3_match_ppg_avg'] - epl_tables_df['next_3_match_ppg_avg']

    winning_goal_diff_series = epl_tables_df.loc[epl_tables_df['match_win']]['match_gd_full']
    mean_winning_gd = winning_goal_diff_series.mean()
    std_winning_gd = winning_goal_diff_series.std()

    big_win_goal_diff_thresh = np.round(mean_winning_gd + std_winning_gd)

    epl_tables_df['big_win'] = epl_tables_df['match_gd_full'] >= big_win_goal_diff_thresh
    epl_tables_df['bad_loss'] = epl_tables_df['match_gd_full'] <= -big_win_goal_diff_thresh

    big_win_group = epl_tables_df.loc[epl_tables_df['big_win']]
    bad_loss_group = epl_tables_df.loc[epl_tables_df['bad_loss']]
    control_group = epl_tables_df.loc[(~epl_tables_df['big_win']) & (~epl_tables_df['bad_loss'])]

    kw_h_stat, p_value = stats.kruskal(bad_loss_group['3_match_ppg_diff'],
                                       control_group['3_match_ppg_diff'],
                                       nan_policy='omit')

    fig, ax = plt.subplots()

    fig: plt.Figure
    ax: plt.Axes

    # ax.hist(epl_tables_df['next_3_match_ppg_avg'], bins=10)
    ax.hist(big_win_group['3_match_ppg_diff'], bins=10)
    ax.hist(bad_loss_group['3_match_ppg_diff'], bins=10)

    plt.show()


if __name__ == '__main__':
    perform_analysis()
