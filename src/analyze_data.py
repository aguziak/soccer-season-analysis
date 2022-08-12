from src.get_season_data import get_season_data, get_team_colors_from_api
from constants import team_colors
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import src.visualizations


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
                     'home_gd_full_time', 'away_gd_full_time', 'league-id', 'league-season', 'teams-home-logo',
                     'fixture-date']
    matches_df = matches_df[relevant_cols]

    home_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'home_win', 'draw', 'home_pts', 'home_gd_half_time', 'home_gd_full_time',
                                        'league-id', 'league-season', 'fixture-date']] \
        .rename({'teams-home-id': 'team_id', 'teams-home-name': 'team_name', 'teams-away-id': 'opp_team_id',
                 'teams-away-name': 'opp_team_name', 'home_win': 'match_win', 'home_pts': 'match_pts',
                 'home_gd_half_time': 'match_gd_half', 'home_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw', 'league-id': 'league_id', 'league-season': 'league_season',
                 'fixture-date': 'fixture_date'}, axis='columns')
    home_match_results_df['home'] = True

    away_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                                        'away_win', 'draw', 'away_pts', 'away_gd_half_time', 'away_gd_full_time',
                                        'league-id', 'league-season', 'fixture-date']] \
        .rename({'teams-away-id': 'team_id', 'teams-away-name': 'team_name', 'teams-home-id': 'opp_team_id',
                 'teams-home-name': 'opp_team_name', 'away_win': 'match_win', 'away_pts': 'match_pts',
                 'away_gd_half_time': 'match_gd_half', 'away_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw', 'league-id': 'league_id', 'league-season': 'league_season',
                 'fixture-date': 'fixture_date'}, axis='columns')
    away_match_results_df['home'] = False

    tables_df = pd.concat([home_match_results_df, away_match_results_df], axis='rows') \
        .sort_values(by='round') \
        .reset_index(drop=True)

    cumulative_stats = tables_df[['team_id', 'match_win', 'match_draw', 'match_pts', 'match_gd_half',
                                  'match_gd_full']] \
        .groupby(by='team_id').cumsum() \
        .rename({'match_win': 'wins', 'match_draw': 'draws', 'match_pts': 'pts', 'match_gd_half': 'gd_half',
                 'match_gd_full': 'gd_full'}, axis='columns')

    tables_df = pd.concat([tables_df, cumulative_stats], axis='columns')
    tables_df['losses'] = tables_df['round'] - tables_df['wins'] - tables_df['draws']

    tables_df['fixture_date'] = tables_df['fixture_date'].apply(datetime.datetime.fromisoformat)

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


def validate_tables_df(tables_df: pd.DataFrame):
    n_rounds = max(tables_df['round'])
    n_teams = tables_df['team_id'].nunique(dropna=True)

    n_rows_expected = n_rounds * n_teams

    if len(tables_df) != n_rows_expected:
        return False

    return True


def create_multi_season_tables_df(start_year: int, end_year: int, league_id: int, throw_on_invalid=True,
                                  use_cache=True):
    """
    Gets all the season data for a range of years and return the results as one big DataFrame

    Args:
        start_year: Start of the range inclusive
        end_year: End of the range inclusive
        league_id: Id for the league
        throw_on_invalid: If true will throw an error if any season data fails validation. If false will omit
            the data and still return
        use_cache: Whether or not to prioritize the local cache when getting data

    Returns:
        DataFrame: Pandas DataFrame with multiple seasons of league table data

    """
    multi_season_table_df = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        current_season_df = create_tables_df(get_season_data(season=year, league_id=league_id, use_cache=use_cache))
        is_season_valid = validate_tables_df(current_season_df)
        if not is_season_valid and throw_on_invalid:
            raise RuntimeError(f'Season {year} league {league_id} failed validation!')
        elif is_season_valid:
            multi_season_table_df = pd.concat([multi_season_table_df, current_season_df])

    def create_final_position_column(season_df: pd.DataFrame):
        final_round_df = season_df.loc[season_df['round'] == max(season_df['round'])] \
            .sort_values(by=['pts', 'gd_full'], ascending=True)
        final_round_df['final_position'] = final_round_df['pts'].rank(method='first').astype(int)
        final_round_df = final_round_df.reset_index(drop=True)

        season_df = season_df.reset_index(drop=True)
        season_df = season_df.merge(final_round_df[['team_id', 'final_position']], on='team_id')

        return season_df

    multi_season_table_df = multi_season_table_df.groupby('league_season').apply(create_final_position_column)

    return multi_season_table_df.reset_index(drop=True)


def perform_analysis():
    epl_league_id = 39
    epl_tables_df = create_multi_season_tables_df(2011, 2021, epl_league_id)

    # src.visualizations.create_bokeh_plot_for_round(epl_tables_df, 1)

    metric_test_range = range(3, 11)

    gd_res_by_team = pd.concat([run_gd_statistical_test_for_period(epl_tables_df, period=period_length, by_team=True)
                                for period_length in metric_test_range], axis=0).dropna().reset_index(drop=True)
    n_teams = len(gd_res_by_team['team_name'].unique())

    team_dfs = [group for group in gd_res_by_team.groupby(by=['team_name'])]

    gd_res = {period_length: run_gd_statistical_test_for_period(epl_tables_df, period=period_length, by_team=True)
              for period_length in range(3, 11)}
    tod_res = run_tod_statistical_test_for_period(epl_tables_df)
    stoke_res = run_stoke_statistical_test(epl_tables_df)

    print(gd_res)
    print(tod_res)
    print(stoke_res)


def run_stoke_statistical_test(epl_tables_df, late_game_hour_cutoff=16, cold_season_start_month=10,
                               cold_season_end_month=4):
    epl_tables_df['fixture_hour'] = epl_tables_df['fixture_date'].apply(lambda date: date.hour)
    epl_tables_df['fixture_month'] = epl_tables_df['fixture_date'].apply(lambda date: date.month)

    epl_tables_df['late_game'] = epl_tables_df['fixture_hour'] >= late_game_hour_cutoff
    epl_tables_df['cold_game'] = (epl_tables_df['fixture_month'] >= cold_season_start_month) | \
                                 (epl_tables_df['fixture_month'] < cold_season_end_month)

    stoke_like = {'Blackburn': True,
                  'Manchester United': False,
                  'Chelsea': False,
                  'Arsenal': False,
                  'Wolves': False,
                  'Norwich': False,
                  'Bolton': True,
                  'Sunderland': True,
                  'Aston Villa': False,
                  'Everton': False,
                  'Swansea': False,
                  'QPR': False,
                  'West Brom': False,
                  'Stoke City': True,
                  'Newcastle': True,
                  'Wigan': True,
                  'Liverpool': False,
                  'Fulham': False,
                  'Manchester City': False,
                  'Tottenham': False,
                  'Southampton': False,
                  'Reading': False,
                  'West Ham': False,
                  'Cardiff': False,
                  'Hull City': True,
                  'Crystal Palace': False,
                  'Leicester': False,
                  'Burnley': True,
                  'Watford': False,
                  'Bournemouth': False,
                  'Middlesbrough': True,
                  'Brighton': False,
                  'Huddersfield': True,
                  'Sheffield Utd': True,
                  'Leeds': False,
                  'Brentford': False}

    epl_tables_df['stoke_like_opposition'] = epl_tables_df['opp_team_name'].apply(lambda name: stoke_like[name])

    cold_rainy_nights_at_stoke_df = epl_tables_df.loc[
        epl_tables_df['cold_game'] &
        epl_tables_df['late_game'] &
        epl_tables_df['stoke_like_opposition'] &
        ~epl_tables_df['home']
        ]

    control_group = epl_tables_df.loc[
        ~(epl_tables_df['cold_game'] &
          epl_tables_df['late_game'] &
          epl_tables_df['stoke_like_opposition']) &
        ~epl_tables_df['home']
        ]

    cold_rainy_nights_outcome = cold_rainy_nights_at_stoke_df.groupby('match_pts').apply(len).tolist()
    control_game_outcomes = control_group.groupby('match_pts').apply(len).tolist()

    g, p_value, dof, expected = stats.chi2_contingency(
        np.array([cold_rainy_nights_outcome, control_game_outcomes])
    )

    return p_value


def run_tod_statistical_test_for_period(epl_tables_df, late_game_hour_cutoff=18):
    epl_tables_df['fixture_hour'] = epl_tables_df['fixture_date'].apply(lambda date: date.hour)
    epl_tables_df['late_game'] = epl_tables_df['fixture_hour'] >= late_game_hour_cutoff

    res = dict()
    for team_name in epl_tables_df['team_name'].unique():
        late_game_group = epl_tables_df.loc[epl_tables_df['late_game'] & (epl_tables_df['team_name'] == team_name)]
        control_group = epl_tables_df.loc[(~epl_tables_df['late_game']) & (epl_tables_df['team_name'] == team_name)]

        late_game_outcomes = late_game_group.groupby('match_pts').apply(len).tolist()
        control_game_outcomes = control_group.groupby('match_pts').apply(len).tolist()

        if len(late_game_outcomes) == 3 and len(control_game_outcomes) == 3:
            g, p_value, dof, expected = stats.chi2_contingency(
                np.array([late_game_outcomes, control_game_outcomes])
            )
        else:
            p_value = np.nan

        res.update({team_name: (len(late_game_group), p_value)})

    return res


def run_gd_statistical_test_for_period(epl_tables_df, period=3, by_team=False, big_win_goal_diff_thresh=4):
    epl_tables_df = epl_tables_df.groupby(by=['league_season', 'team_id']).apply(_create_rolling_avg_match_pts,
                                                                                 period=period)
    epl_tables_df = epl_tables_df.dropna()
    epl_tables_df[f'{period}_match_ppg_diff'] = \
        epl_tables_df[f'prev_{period}_match_ppg_avg'] - epl_tables_df[f'next_{period}_match_ppg_avg']

    epl_tables_df['big_win'] = epl_tables_df['match_gd_full'] >= big_win_goal_diff_thresh
    epl_tables_df['bad_loss'] = epl_tables_df['match_gd_full'] <= -big_win_goal_diff_thresh

    results_df = pd.DataFrame()

    if by_team:
        for team_name in epl_tables_df['team_name'].unique():
            big_win_group = epl_tables_df.loc[epl_tables_df['big_win'] & (epl_tables_df['team_name'] == team_name)]
            bad_loss_group = epl_tables_df.loc[
                epl_tables_df['bad_loss'] & (epl_tables_df['team_name'] == team_name)]
            control_group = epl_tables_df.loc[(~epl_tables_df['big_win'])
                                              & (~epl_tables_df['bad_loss'])
                                              & (epl_tables_df['team_name'] == team_name)]

            if len(bad_loss_group) != 0:
                _, bad_loss_p_value = stats.mannwhitneyu(bad_loss_group[f'{period}_match_ppg_diff'],
                                                         control_group[f'{period}_match_ppg_diff'])
            else:
                bad_loss_p_value = np.nan

            if len(big_win_group) != 0:
                _, big_win_p_value = stats.mannwhitneyu(big_win_group[f'{period}_match_ppg_diff'],
                                                        control_group[f'{period}_match_ppg_diff'])
            else:
                big_win_p_value = np.nan

            results_df = pd.concat([results_df, pd.Series({
                'team_name': team_name,
                'period': period,
                'bad_loss_n': len(bad_loss_group),
                'bad_loss_p_value': bad_loss_p_value,
                'big_win_n': len(big_win_group),
                'big_win_p_value': big_win_p_value
            })], axis=1)

        return results_df.transpose()
    else:
        big_win_group = epl_tables_df.loc[epl_tables_df['big_win']]
        bad_loss_group = epl_tables_df.loc[epl_tables_df['bad_loss']]
        control_group = epl_tables_df.loc[(~epl_tables_df['big_win']) & (~epl_tables_df['bad_loss'])]

        _, bad_loss_p_value = stats.mannwhitneyu(bad_loss_group[f'{period}_match_ppg_diff'],
                                                 control_group[f'{period}_match_ppg_diff'])
        _, big_win_p_value = stats.mannwhitneyu(big_win_group[f'{period}_match_ppg_diff'],
                                                control_group[f'{period}_match_ppg_diff'])

        results_df = pd.Series({
            'period': period,
            'bad_loss_n': len(bad_loss_group),
            'bad_loss_p_value': bad_loss_p_value,
            'big_win_n': len(big_win_group),
            'big_win_p_value': big_win_p_value
        }).to_frame()

        return results_df


if __name__ == '__main__':
    perform_analysis()
