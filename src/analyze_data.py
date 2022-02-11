from src.get_season_data import get_season_data
from constants import team_three_letter_codes, team_colors

import pandas as pd
import matplotlib.pyplot as plt


def create_season_table_plot(tables_df):
    fig, (bar_ax, line_ax) = plt.subplots(2, constrained_layout=True)

    fig: plt.Figure
    bar_ax: plt.Axes
    line_ax: plt.Axes

    final_round_num = max(tables_df['round'])
    final_round_df = tables_df.loc[tables_df['round'] == final_round_num]
    final_round_df = final_round_df.sort_values(by='pts')

    bar_ax.bar(x=final_round_df['team_abbr'], height=final_round_df['pts'], color=final_round_df['team_color'])
    bar_ax.set_ylabel('Points')

    for team_abbr in final_round_df['team_abbr']:
        team_df = tables_df.loc[tables_df['team_abbr'] == team_abbr]
        team_color = team_df['team_color'].iloc[0]
        line_ax.plot(team_df['round'], team_df['pts'], label=team_abbr, color=team_color)
        line_ax.set_xlabel('Round')
        line_ax.set_ylabel('Points')

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

    home_match_results_df = matches_df[['round', 'teams-home-id', 'teams-home-name', 'home_win', 'draw', 'home_pts',
                                        'home_gd_half_time', 'home_gd_full_time']] \
        .rename({'teams-home-id': 'team_id', 'teams-home-name': 'team_name', 'home_win': 'match_win',
                 'home_pts': 'match_pts', 'home_gd_half_time': 'match_gd_half', 'home_gd_full_time': 'match_gd_full',
                 'draw': 'match_draw'}, axis='columns')
    home_match_results_df['home'] = True

    away_match_results_df = matches_df[['round', 'teams-away-id', 'teams-away-name', 'away_win', 'draw', 'away_pts',
                                        'away_gd_half_time', 'away_gd_full_time']] \
        .rename({'teams-away-id': 'team_id', 'teams-away-name': 'team_name', 'away_win': 'match_win',
                 'away_pts': 'match_pts', 'away_gd_half_time': 'match_gd_half', 'away_gd_full_time': 'match_gd_full',
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
    tables_df['team_color'] = tables_df.apply(lambda row: team_colors[row['team_id']], axis='columns')

    return tables_df


if __name__ == '__main__':
    epl_league_id = 39
    epl_matches_df = get_season_data(season=2018, league_id=epl_league_id, use_cache=True)
    epl_tables_df = create_tables_df(epl_matches_df)

    create_season_table_plot(epl_tables_df)

    print(epl_tables_df.head())
