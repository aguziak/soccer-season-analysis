from src.get_season_data import get_season_data
import pandas as pd


def create_tables_df(matches_df: pd.DataFrame) -> pd.DataFrame:
    matches_df['round'] = matches_df['league-round'].apply(lambda s: int(s.split('Regular Season - ')[1]))

    relevant_cols = ['round', 'teams-home-id', 'teams-home-name', 'teams-away-id', 'teams-away-name',
                     'score-halftime-home', 'score-halftime-away', 'score-fulltime-home', 'score-fulltime-away']

    matches_df = matches_df[relevant_cols]

    matches_df['home_win'] = matches_df['score-fulltime-home'] > matches_df['score-fulltime-away']
    matches_df['away_win'] = matches_df['score-fulltime-home'] < matches_df['score-fulltime-away']
    matches_df['draw'] = matches_df['score-fulltime-home'] == matches_df['score-fulltime-away']

    matches_df['home_pts'] = matches_df['home_win'] * 3 + matches_df['draw'] * 1
    matches_df['away_pts'] = matches_df['away_win'] * 3 + matches_df['draw'] * 1

    matches_df['home_gd_half_time'] = matches_df['score-halftime-home'] - matches_df['score-halftime-away']
    matches_df['away_gd_half_time'] = -matches_df['home_gd_half_time']

    matches_df['home_gd_full_time'] = matches_df['score-fulltime-home'] - matches_df['score-fulltime-away']
    matches_df['away_gd_full_time'] = -matches_df['home_gd_full_time']

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

    return tables_df


if __name__ == '__main__':
    epl_league_id = 39
    epl_matches_df = get_season_data(season=2018, league_id=epl_league_id, use_cache=True)
    epl_tables_df = create_tables_df(epl_matches_df)

    print(epl_tables_df.head())
